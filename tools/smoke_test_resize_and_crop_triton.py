"""Smoke test for ResizeAndCropTriton correctness.

Run with:
    python tools/smoke_test_resize_and_crop_triton.py
"""

from __future__ import annotations

import sys

import torch

from eva.vision.data.transforms.common.resize_and_crop import ResizeAndCrop
from eva.vision.data.transforms.common.resize_and_crop_triton import ResizeAndCropTriton


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: CUDA is not available.")
        return 0

    device = torch.device("cuda")
    batch_size = 8
    height = 1536
    width = 2048
    target_size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    reference = ResizeAndCrop(size=target_size, mean=mean, std=std)
    triton_transform = ResizeAndCropTriton(size=target_size, mean=mean, std=std)

    images = torch.randint(
        low=0,
        high=256,
        size=(batch_size, 3, height, width),
        dtype=torch.uint8,
        device=device,
    )

    with torch.no_grad():
        reference_out = torch.stack([reference(image) for image in images], dim=0).to(device)
        try:
            triton_out = triton_transform.batch_forward(images)
        except ImportError:
            print("SKIP: Triton is not installed in this environment.")
            return 0

    abs_diff = (triton_out - reference_out).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    print(f"max_abs_diff={max_diff:.8f}")
    print(f"mean_abs_diff={mean_diff:.8f}")

    tol = 5e-4
    if max_diff > tol:
        print(f"FAIL: max_abs_diff {max_diff:.8f} > tolerance {tol}")
        return 1

    print("PASS: Triton output matches reference within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
