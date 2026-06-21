#!/usr/bin/env python
"""Export a few samples as PLY files for visual inspection."""

import argparse
import json
import os
import sys

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evaluate import find_meshes, find_samples, load_mesh_vf, load_pointcloud  # noqa: E402


def load_meta(meta_dir, key):
    if not meta_dir:
        return None
    path = os.path.join(meta_dir, key, "meta.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def transform_mesh_vertices(vertices, mode, ref_pc=None, meta=None):
    vertices = vertices.copy()
    if mode == "none":
        return vertices
    if mode == "meta":
        if meta is None:
            return vertices
        center = np.asarray(meta["normalize_center"], dtype=np.float64)
        scale = float(meta["normalize_scale"])
        return (vertices - center) / (scale + 1e-12)
    if mode == "ref_gt":
        if ref_pc is None:
            return vertices
        center = (ref_pc.max(axis=0) + ref_pc.min(axis=0)) / 2.0
        scale = np.sqrt(((ref_pc - center) ** 2.0).sum(axis=1)).max()
        return (vertices - center) / (scale + 1e-12)
    raise ValueError(f"unknown mesh mode: {mode}")


def write_point_ply(path, points):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")


def write_mesh_ply(path, vertices, faces):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for face in faces:
            f.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def main():
    parser = argparse.ArgumentParser(description="Export debug PLY files.")
    parser.add_argument("--out_dir", default="debug_clouds")
    parser.add_argument("--noisy_dir", required=True)
    parser.add_argument("--pred_dir", default="")
    parser.add_argument("--gt_dir", default="")
    parser.add_argument("--mesh_dir", default="")
    parser.add_argument("--meta_dir", default="val_meta")
    parser.add_argument("--mesh_mode", choices=["meta", "ref_gt", "none"], default="meta")
    parser.add_argument("--keys", default="", help="Comma-separated relative keys. If empty, use first --limit keys.")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--max_points", type=int, default=50000)
    parser.add_argument("--pred_filename", default="denoised.npy")
    parser.add_argument("--gt_filename", default="clean.npy")
    parser.add_argument("--noisy_filename", default="noisy.npy")
    parser.add_argument("--mesh_data_name", default="models/model_normalized.obj")
    args = parser.parse_args()

    noisy_samples = find_samples(args.noisy_dir, args.noisy_filename)
    pred_samples = find_samples(args.pred_dir, args.pred_filename) if args.pred_dir else {}
    gt_samples = find_samples(args.gt_dir, args.gt_filename) if args.gt_dir else {}
    mesh_samples = find_meshes(args.mesh_dir, args.mesh_data_name) if args.mesh_dir else {}

    if args.keys:
        keys = [x.strip() for x in args.keys.split(",") if x.strip()]
    else:
        keys = sorted(noisy_samples)[: args.limit]

    for key in keys:
        sample_dir = os.path.join(args.out_dir, key.replace("/", "__"))
        noisy = load_pointcloud(noisy_samples[key])
        if args.max_points > 0 and len(noisy) > args.max_points:
            noisy = noisy[: args.max_points]
        write_point_ply(os.path.join(sample_dir, "noisy.ply"), noisy)

        if key in pred_samples:
            pred = load_pointcloud(pred_samples[key])
            if args.max_points > 0 and len(pred) > args.max_points:
                pred = pred[: args.max_points]
            write_point_ply(os.path.join(sample_dir, "pred.ply"), pred)

        gt = None
        if key in gt_samples:
            gt = load_pointcloud(gt_samples[key])
            if args.max_points > 0 and len(gt) > args.max_points:
                gt = gt[: args.max_points]
            write_point_ply(os.path.join(sample_dir, "clean.ply"), gt)

        if key in mesh_samples:
            mv, mf = load_mesh_vf(mesh_samples[key])
            meta = load_meta(args.meta_dir, key)
            mv = transform_mesh_vertices(mv, args.mesh_mode, ref_pc=gt, meta=meta)
            write_mesh_ply(os.path.join(sample_dir, f"mesh_{args.mesh_mode}.ply"), mv, mf)

        print(f"exported {key} -> {sample_dir}")


if __name__ == "__main__":
    main()
