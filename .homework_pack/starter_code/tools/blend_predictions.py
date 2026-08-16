#!/usr/bin/env python
"""Blend two recursive denoised.npy prediction trees."""

import argparse
import glob
import os
import sys

import numpy as np


def find_predictions(base_dir, filename):
    predictions = {}
    pattern = os.path.join(base_dir, "**", filename)
    for path in sorted(glob.glob(pattern, recursive=True)):
        rel_path = os.path.relpath(path, base_dir)
        predictions[rel_path] = path
    return predictions


def validate_output_dir(anchor_dir, challenger_dir, output_dir):
    anchor_dir = os.path.realpath(anchor_dir)
    challenger_dir = os.path.realpath(challenger_dir)
    output_dir = os.path.realpath(output_dir)

    for name, input_dir in (("anchor_dir", anchor_dir), ("challenger_dir", challenger_dir)):
        if os.path.commonpath([input_dir, output_dir]) == input_dir:
            raise ValueError(f"output_dir must not equal or be inside {name}")

    if os.path.exists(output_dir):
        if not os.path.isdir(output_dir):
            raise ValueError("output_dir must be a directory")
        with os.scandir(output_dir) as entries:
            if any(entries):
                raise ValueError("output_dir must be empty")


def blend_predictions(anchor_dir, challenger_dir, output_dir, weight, filename="denoised.npy"):
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must be in [0, 1], got {weight}")

    validate_output_dir(anchor_dir, challenger_dir, output_dir)

    anchor_files = find_predictions(anchor_dir, filename)
    challenger_files = find_predictions(challenger_dir, filename)
    if not anchor_files:
        raise FileNotFoundError(f"no {filename} found under {anchor_dir}")

    anchor_paths = set(anchor_files)
    challenger_paths = set(challenger_files)
    missing = sorted(anchor_paths - challenger_paths)
    extra = sorted(challenger_paths - anchor_paths)
    if missing or extra:
        details = []
        if missing:
            details.append(f"challenger missing {len(missing)} paths: {missing[:5]}")
        if extra:
            details.append(f"challenger has {len(extra)} extra paths: {extra[:5]}")
        raise ValueError("; ".join(details))

    for rel_path in sorted(anchor_paths):
        anchor = np.load(anchor_files[rel_path])
        challenger = np.load(challenger_files[rel_path])
        if anchor.shape != challenger.shape:
            raise ValueError(
                f"{rel_path}: shape mismatch anchor={anchor.shape}, "
                f"challenger={challenger.shape}"
            )
        if anchor.ndim != 2 or anchor.shape[1] != 3:
            raise ValueError(f"{rel_path}: expected shape (N, 3), got {anchor.shape}")
        if not np.isfinite(anchor).all():
            raise ValueError(f"{rel_path}: anchor contains NaN or Inf")
        if not np.isfinite(challenger).all():
            raise ValueError(f"{rel_path}: challenger contains NaN or Inf")

        blended = (
            (1.0 - weight) * anchor.astype(np.float64)
            + weight * challenger.astype(np.float64)
        ).astype(np.float32)
        if not np.isfinite(blended).all():
            raise ValueError(f"{rel_path}: blended output contains NaN or Inf")

        output_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, blended)

    return len(anchor_paths)


def main():
    parser = argparse.ArgumentParser(description="融合两套点云降噪预测结果")
    parser.add_argument("--anchor_dir", required=True)
    parser.add_argument("--challenger_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--weight", required=True, type=float)
    parser.add_argument("--filename", default="denoised.npy")
    args = parser.parse_args()

    count = blend_predictions(
        anchor_dir=args.anchor_dir,
        challenger_dir=args.challenger_dir,
        output_dir=args.output_dir,
        weight=args.weight,
        filename=args.filename,
    )

    print(
        f"blended {count} samples with weight={args.weight:g} "
        f"into {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
