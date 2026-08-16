#!/usr/bin/env python
"""Diagnose prediction files or result.zip without running the model."""

import argparse
import csv
import io
import os
import zipfile

import numpy as np


def find_files(base_dir, filename):
    import glob

    return {
        os.path.relpath(os.path.dirname(path), base_dir): path
        for path in sorted(glob.glob(os.path.join(base_dir, "**", filename), recursive=True))
    }


def find_zip_predictions(zip_path, filename):
    samples = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in sorted(zf.namelist()):
            if name.endswith("/" + filename):
                samples[os.path.dirname(name)] = name
    return samples


def load_prediction(key, samples, zip_path=None):
    if zip_path:
        with zipfile.ZipFile(zip_path) as zf:
            return np.load(io.BytesIO(zf.read(samples[key])))
    return np.load(samples[key])


def pc_stats(pc):
    pc64 = pc.astype(np.float64)
    p_min = pc64.min(axis=0)
    p_max = pc64.max(axis=0)
    center = (p_max + p_min) / 2.0
    scale = np.sqrt(((pc64 - center) ** 2.0).sum(axis=1)).max()
    bbox = p_max - p_min
    return p_min, p_max, center, scale, bbox


def fmt_shape(shape):
    return "x".join(str(x) for x in shape)


def main():
    parser = argparse.ArgumentParser(description="Diagnose denoised prediction files.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pred_dir", help="Prediction directory containing denoised.npy files")
    group.add_argument("--pred_zip", help="result.zip containing shapenet/.../denoised.npy")
    parser.add_argument("--noisy_dir", required=True, help="Directory containing noisy.npy files")
    parser.add_argument("--pred_filename", default="denoised.npy")
    parser.add_argument("--noisy_filename", default="noisy.npy")
    parser.add_argument("--csv_path", default="", help="Optional per-sample CSV output")
    parser.add_argument("--topk", type=int, default=10, help="Number of largest-displacement samples to print")
    args = parser.parse_args()

    noisy_samples = find_files(args.noisy_dir, args.noisy_filename)
    if args.pred_zip:
        pred_samples = find_zip_predictions(args.pred_zip, args.pred_filename)
    else:
        pred_samples = find_files(args.pred_dir, args.pred_filename)

    pred_keys = set(pred_samples)
    noisy_keys = set(noisy_samples)
    common = sorted(pred_keys & noisy_keys)
    missing = sorted(noisy_keys - pred_keys)
    extra = sorted(pred_keys - noisy_keys)

    rows = []
    errors = []
    for key in common:
        pred = load_prediction(key, pred_samples, zip_path=args.pred_zip)
        noisy = np.load(noisy_samples[key])
        finite = bool(np.isfinite(pred).all())
        shape_ok = pred.shape == noisy.shape and pred.ndim == 2 and pred.shape[1] == 3
        dtype_ok = pred.dtype == np.float32
        if not shape_ok:
            errors.append(f"{key}: shape pred={pred.shape}, noisy={noisy.shape}")
        if not dtype_ok:
            errors.append(f"{key}: dtype pred={pred.dtype}")
        if not finite:
            errors.append(f"{key}: non-finite prediction")

        pred_min, pred_max, pred_center, pred_scale, pred_bbox = pc_stats(pred)
        noisy_min, noisy_max, noisy_center, noisy_scale, noisy_bbox = pc_stats(noisy)
        if shape_ok:
            disp = np.sqrt(((pred.astype(np.float64) - noisy.astype(np.float64)) ** 2.0).sum(axis=1))
            disp_mean = float(disp.mean())
            disp_p50 = float(np.percentile(disp, 50))
            disp_p95 = float(np.percentile(disp, 95))
            disp_p99 = float(np.percentile(disp, 99))
            disp_max = float(disp.max())
        else:
            disp_mean = disp_p50 = disp_p95 = disp_p99 = disp_max = float("nan")

        center_shift = float(np.linalg.norm(pred_center - noisy_center))
        row = {
            "key": key,
            "pred_shape": fmt_shape(pred.shape),
            "noisy_shape": fmt_shape(noisy.shape),
            "pred_dtype": str(pred.dtype),
            "finite": finite,
            "shape_ok": shape_ok,
            "dtype_ok": dtype_ok,
            "pred_scale": float(pred_scale),
            "noisy_scale": float(noisy_scale),
            "scale_ratio": float(pred_scale / noisy_scale) if noisy_scale > 1e-12 else "",
            "center_shift": center_shift,
            "disp_mean": disp_mean,
            "disp_p50": disp_p50,
            "disp_p95": disp_p95,
            "disp_p99": disp_p99,
            "disp_max": disp_max,
            "pred_center_x": float(pred_center[0]),
            "pred_center_y": float(pred_center[1]),
            "pred_center_z": float(pred_center[2]),
            "noisy_center_x": float(noisy_center[0]),
            "noisy_center_y": float(noisy_center[1]),
            "noisy_center_z": float(noisy_center[2]),
            "pred_bbox_x": float(pred_bbox[0]),
            "pred_bbox_y": float(pred_bbox[1]),
            "pred_bbox_z": float(pred_bbox[2]),
            "noisy_bbox_x": float(noisy_bbox[0]),
            "noisy_bbox_y": float(noisy_bbox[1]),
            "noisy_bbox_z": float(noisy_bbox[2]),
        }
        rows.append(row)

    for key in missing:
        errors.append(f"{key}: missing prediction")
    for key in extra:
        errors.append(f"{key}: extra prediction")

    print(f"noisy samples: {len(noisy_samples)}")
    print(f"pred samples:  {len(pred_samples)}")
    print(f"matched:       {len(common)}")
    print(f"missing:       {len(missing)}")
    print(f"extra:         {len(extra)}")
    if rows:
        for name in ["disp_mean", "disp_p95", "disp_max", "center_shift", "scale_ratio"]:
            vals = np.array([r[name] for r in rows if r[name] == r[name]], dtype=np.float64)
            if len(vals):
                print(f"{name}: mean={vals.mean():.8f}, p95={np.percentile(vals, 95):.8f}, max={vals.max():.8f}")
        print("\nTop displacement samples:")
        for row in sorted(rows, key=lambda r: r["disp_mean"], reverse=True)[: args.topk]:
            print(
                f"  {row['key']} disp_mean={row['disp_mean']:.8f} "
                f"scale_ratio={row['scale_ratio']:.6f} center_shift={row['center_shift']:.8f}"
            )

    if args.csv_path:
        os.makedirs(os.path.dirname(args.csv_path) or ".", exist_ok=True)
        fieldnames = list(rows[0].keys()) if rows else [
            "key", "pred_shape", "noisy_shape", "pred_dtype", "finite", "shape_ok", "dtype_ok"
        ]
        with open(args.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV written: {args.csv_path}")

    if errors:
        print("\nErrors:")
        for item in errors[:50]:
            print(f"- {item}")
        if len(errors) > 50:
            print(f"- ... and {len(errors) - 50} more")
        return 1
    print("\nPrediction diagnostics passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
