#!/usr/bin/env python
"""Check prediction files before packaging result.zip."""

import argparse
import glob
import os
import sys

import numpy as np


def find_samples(base_dir, filename):
    samples = {}
    for path in sorted(glob.glob(os.path.join(base_dir, "**", filename), recursive=True)):
        rel = os.path.relpath(os.path.dirname(path), base_dir)
        samples[rel] = path
    return samples


def main():
    parser = argparse.ArgumentParser(description="检查点云降噪预测结果的提交格式")
    parser.add_argument("--pred_dir", default="./tmp_predict", help="预测结果目录")
    parser.add_argument("--noisy_dir", default="../dataset_test_noisy", help="测试 noisy 点云目录")
    parser.add_argument("--pred_filename", default="denoised.npy")
    parser.add_argument("--noisy_filename", default="noisy.npy")
    args = parser.parse_args()

    pred_samples = find_samples(args.pred_dir, args.pred_filename)
    noisy_samples = find_samples(args.noisy_dir, args.noisy_filename)

    missing = sorted(set(noisy_samples) - set(pred_samples))
    extra = sorted(set(pred_samples) - set(noisy_samples))
    errors = []

    for key in sorted(set(pred_samples) & set(noisy_samples)):
        pred = np.load(pred_samples[key])
        noisy = np.load(noisy_samples[key])
        if pred.shape != noisy.shape:
            errors.append(f"{key}: shape mismatch pred={pred.shape}, noisy={noisy.shape}")
        if pred.ndim != 2 or pred.shape[1] != 3:
            errors.append(f"{key}: expected shape (N, 3), got {pred.shape}")
        if pred.dtype != np.float32:
            errors.append(f"{key}: expected float32, got {pred.dtype}")
        if not np.isfinite(pred).all():
            errors.append(f"{key}: contains NaN or Inf")

    if missing:
        errors.append(f"missing predictions: {len(missing)}")
    if extra:
        errors.append(f"extra predictions: {len(extra)}")

    print(f"noisy samples: {len(noisy_samples)}")
    print(f"pred samples:  {len(pred_samples)}")
    print(f"matched:       {len(set(pred_samples) & set(noisy_samples))}")

    if errors:
        print("\n检查失败：")
        for item in errors[:50]:
            print(f"- {item}")
        if len(errors) > 50:
            print(f"- ... and {len(errors) - 50} more")
        return 1

    print("\n检查通过：预测文件路径、shape、dtype 和数值有效性均符合提交要求。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

