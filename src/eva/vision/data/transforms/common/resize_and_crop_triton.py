"""GPU-accelerated resize/crop/normalize transform using Triton."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _scale_and_normalize_kernel(
        input_ptr,
        output_ptr,
        mean_ptr,
        std_ptr,
        total_elements,
        hw,
        BLOCK_SIZE: tl.constexpr,  # type: ignore[valid-type]
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements

        values = tl.load(input_ptr + offsets, mask=mask, other=0).to(tl.float32)
        channels = (offsets // hw) % 3
        mean = tl.load(mean_ptr + channels, mask=mask, other=0.0)
        std = tl.load(std_ptr + channels, mask=mask, other=1.0)

        normalized = (values / 255.0 - mean) / std
        tl.store(output_ptr + offsets, normalized, mask=mask)


class ResizeAndCropTriton:
    """Resizes, center-crops and normalizes an input image on CUDA.

    Notes:
        - Resize and center crop are performed using PyTorch operators.
        - ToDtype(scale=True) + Normalize are fused in a Triton kernel.
    """

    def __init__(
        self,
        size: int | Sequence[int] = 224,
        mean: Sequence[float] = (0.5, 0.5, 0.5),
        std: Sequence[float] = (0.5, 0.5, 0.5),
        use_triton_scale_normalize: bool = True,
    ) -> None:
        self._size = size
        self._mean = torch.tensor(mean, dtype=torch.float32)
        self._std = torch.tensor(std, dtype=torch.float32)
        self._use_triton_scale_normalize = use_triton_scale_normalize

        if self._mean.numel() != 3 or self._std.numel() != 3:
            raise ValueError("ResizeAndCropTriton currently supports exactly 3 channels.")

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Applies resize/crop and fused scale+normalize on the input image.

        Args:
            image: Input image as `C x H x W` tensor.

        Returns:
            Output image as `C x H_out x W_out` float32 tensor.
        """
        output = self.batch_forward(image.unsqueeze(0))
        return output.squeeze(0)

    def batch_forward(self, images: torch.Tensor) -> torch.Tensor:
        """Applies resize/crop and fused scale+normalize on batched images.

        Args:
            images: Input tensor as `N x C x H x W`.

        Returns:
            Output tensor as `N x C x H_out x W_out` float32 tensor.
        """
        if images.ndim != 4:
            raise ValueError(f"Expected input with shape NCHW, got {tuple(images.shape)}")
        if images.shape[1] != 3:
            raise ValueError(f"Expected 3 channels, got {images.shape[1]}")
        if not images.is_cuda:
            raise ValueError("ResizeAndCropTriton requires CUDA input tensors.")

        resized = self._resize_keep_aspect(images)
        cropped = self._center_crop(resized)
        if not cropped.is_contiguous():
            cropped = cropped.contiguous()

        mean = self._mean.to(device=cropped.device)
        std = self._std.to(device=cropped.device)

        if not self._use_triton_scale_normalize:
            return self._torch_scale_and_normalize(cropped, mean, std)

        if not _TRITON_AVAILABLE:
            raise ImportError("Triton is not installed. Install `triton` to use ResizeAndCropTriton.")

        output = torch.empty_like(cropped, dtype=torch.float32)
        total_elements = output.numel()
        hw = output.shape[-1] * output.shape[-2]

        block_size = 1024
        grid = (triton.cdiv(total_elements, block_size),)

        _scale_and_normalize_kernel[grid](
            cropped,
            output,
            mean,
            std,
            total_elements,
            hw,
            BLOCK_SIZE=block_size,
        )
        return output

    def _torch_scale_and_normalize(
        self, images: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        """Reference GPU path for scale+normalize when Triton fusion is disabled."""
        images = images.to(torch.float32)
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)
        return (images / 255.0 - mean) / std

    def _resize_keep_aspect(self, images: torch.Tensor) -> torch.Tensor:
        """Resizes NCHW images with torchvision-like short-side semantics."""
        target_h, target_w = self._target_hw()

        if isinstance(self._size, int):
            _, _, height, width = images.shape
            if height < width:
                target_h = self._size
                target_w = int(round(width * (self._size / height)))
            else:
                target_w = self._size
                target_h = int(round(height * (self._size / width)))

        images_fp32 = images.to(torch.float32)
        return F.interpolate(
            images_fp32,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).to(images.dtype)

    def _center_crop(self, images: torch.Tensor) -> torch.Tensor:
        """Center-crops NCHW images."""
        target_h, target_w = self._target_hw()
        _, _, height, width = images.shape

        if target_h > height or target_w > width:
            pad_h = max(target_h - height, 0)
            pad_w = max(target_w - width, 0)
            images = F.pad(
                images,
                [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2],
                value=0,
            )
            _, _, height, width = images.shape

        top = int(round((height - target_h) / 2.0))
        left = int(round((width - target_w) / 2.0))
        return images[:, :, top : top + target_h, left : left + target_w]

    def _target_hw(self) -> tuple[int, int]:
        if isinstance(self._size, int):
            return self._size, self._size
        if len(self._size) != 2:
            raise ValueError(f"Expected size as int or (H, W), got {self._size}")
        return int(self._size[0]), int(self._size[1])
