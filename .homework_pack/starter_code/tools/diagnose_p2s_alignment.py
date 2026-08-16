#!/usr/bin/env python
"""Compare P2S coordinate alignment modes on validation samples."""

import argparse
import csv
import json
import os
import sys

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evaluate import (  # noqa: E402
    find_meshes,
    find_samples,
    load_mesh_vf,
    load_pointcloud,
    metric_to_score,
    point_to_surface_distance,
)


def load_meta(meta_dir, key):
    if not meta_dir:
        return None
    path = os.path.join(meta_dir, key, "meta.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def p2s_with_mode(pc, mesh_v, mesh_f, mode, ref_pc=None, meta=None):
    if mode == "none":
        return point_to_surface_distance(pc, mesh_v, mesh_f)
    if mode == "ref_gt":
        return point_to_surface_distance(pc, mesh_v, mesh_f, normalize_ref_pc=ref_pc)
    if mode == "meta":
        if meta is None:
            return None
        return point_to_surface_distance(
            pc,
            mesh_v,
            mesh_f,
            mesh_center=meta.get("normalize_center"),
            mesh_scale=meta.get("normalize_scale"),
        )
    raise ValueError(f"unknown mode: {mode}")


def parse_modes(text):
    modes = [x.strip() for x in text.split(",") if x.strip()]
    valid = {"ref_gt", "meta", "none"}
    bad = [x for x in modes if x not in valid]
    if bad:
        raise ValueError(f"unsupported modes: {bad}")
    return modes


def main():
    parser = argparse.ArgumentParser(description="Diagnose P2S alignment modes.")
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--noisy_dir", required=True)
    parser.add_argument("--mesh_dir", required=True)
    parser.add_argument("--pred_dir", default="")
    parser.add_argument("--meta_dir", default="val_meta")
    parser.add_argument("--mesh_data_name", default="models/model_normalized.obj")
    parser.add_argument("--pred_filename", default="denoised.npy")
    parser.add_argument("--gt_filename", default="clean.npy")
    parser.add_argument("--noisy_filename", default="noisy.npy")
    parser.add_argument("--modes", default="ref_gt,meta,none")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--csv_path", default="")
    args = parser.parse_args()

    modes = parse_modes(args.modes)
    gt_samples = find_samples(args.gt_dir, args.gt_filename)
    noisy_samples = find_samples(args.noisy_dir, args.noisy_filename)
    mesh_samples = find_meshes(args.mesh_dir, args.mesh_data_name)
    pred_samples = find_samples(args.pred_dir, args.pred_filename) if args.pred_dir else {}

    keys = sorted(set(gt_samples) & set(noisy_samples) & set(mesh_samples))
    if args.limit > 0:
        keys = keys[: args.limit]
    if not keys:
        raise SystemExit("No matched gt/noisy/mesh samples found.")

    rows = []
    for key in keys:
        gt = load_pointcloud(gt_samples[key])
        noisy = load_pointcloud(noisy_samples[key])
        pred = load_pointcloud(pred_samples[key]) if key in pred_samples else None
        mv, mf = load_mesh_vf(mesh_samples[key])
        meta = load_meta(args.meta_dir, key)
        if mv is None or mf is None:
            continue

        for mode in modes:
            clean_p2s = p2s_with_mode(gt, mv, mf, mode, ref_pc=gt, meta=meta)
            noisy_p2s = p2s_with_mode(noisy, mv, mf, mode, ref_pc=gt, meta=meta)
            pred_p2s = p2s_with_mode(pred, mv, mf, mode, ref_pc=gt, meta=meta) if pred is not None else None
            row = {
                "key": key,
                "mode": mode,
                "has_meta": meta is not None,
                "clean_p2s": "" if clean_p2s is None else clean_p2s,
                "noisy_p2s": "" if noisy_p2s is None else noisy_p2s,
                "pred_p2s": "" if pred_p2s is None else pred_p2s,
                "pred_p2s_score": (
                    "" if pred_p2s is None or noisy_p2s is None else metric_to_score(pred_p2s, noisy_p2s)
                ),
                "clean_to_noisy_ratio": (
                    "" if clean_p2s is None or noisy_p2s is None or noisy_p2s < 1e-15 else clean_p2s / noisy_p2s
                ),
            }
            rows.append(row)

    print(f"samples: {len(keys)}")
    for mode in modes:
        mode_rows = [r for r in rows if r["mode"] == mode]
        print(f"\nmode={mode}")
        for name in ["clean_p2s", "noisy_p2s", "pred_p2s", "clean_to_noisy_ratio"]:
            vals = [r[name] for r in mode_rows if r[name] != ""]
            if vals:
                vals = np.array(vals, dtype=np.float64)
                print(f"  {name}: mean={vals.mean():.8f}, p95={np.percentile(vals, 95):.8f}, max={vals.max():.8f}")

    print("\nFirst rows:")
    for row in rows[: min(12, len(rows))]:
        print(row)

    if args.csv_path:
        os.makedirs(os.path.dirname(args.csv_path) or ".", exist_ok=True)
        with open(args.csv_path, "w", newline="") as f:
            fieldnames = [
                "key",
                "mode",
                "has_meta",
                "clean_p2s",
                "noisy_p2s",
                "pred_p2s",
                "pred_p2s_score",
                "clean_to_noisy_ratio",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV written: {args.csv_path}")


if __name__ == "__main__":
    main()
