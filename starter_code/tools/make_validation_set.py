#!/usr/bin/env python
"""Create a fixed noisy/clean validation set from training meshes."""

import argparse
import json
import os
import sys

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.utils import sample_vertex_groups  # noqa: E402


def load_mesh(path):
    try:
        import trimesh

        mesh = trimesh.load(path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces)
    except ImportError:
        vertices = []
        faces = []
        with open(path, "r") as f:
            for line in f:
                if line.startswith("v "):
                    vertices.append([float(x) for x in line.split()[1:4]])
                elif line.startswith("f "):
                    face = []
                    for tok in line.split()[1:4]:
                        face.append(int(tok.split("/")[0]) - 1)
                    faces.append(face)
        return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def normalize_to_unit_sphere(pc):
    p_max = pc.max(axis=0)
    p_min = pc.min(axis=0)
    center = (p_max + p_min) / 2.0
    pc_centered = pc - center
    scale = np.sqrt((pc_centered ** 2.0).sum(axis=1)).max()
    return pc_centered / (scale + 1e-12), center, scale


def add_noise(pc, noise_std, noise_type, rng):
    if noise_type == "gaussian":
        noise = rng.normal(0.0, noise_std, size=pc.shape)
    elif noise_type == "laplace":
        noise = rng.laplace(0.0, noise_std / np.sqrt(2.0), size=pc.shape)
    else:
        raise ValueError(f"unsupported noise type: {noise_type}")
    return pc + noise


def main():
    parser = argparse.ArgumentParser(description="Build fixed validation clean/noisy npy pairs.")
    parser.add_argument("--datalist", default="datalist/validate.txt")
    parser.add_argument("--train_root", default="../dataset_train")
    parser.add_argument("--gt_dir", default="val_gt")
    parser.add_argument("--noisy_dir", default="val_noisy")
    parser.add_argument("--meta_dir", default="val_meta")
    parser.add_argument("--no_meta", action="store_true")
    parser.add_argument("--mesh_name", default="models/model_normalized.obj")
    parser.add_argument("--num_points", type=int, default=50000)
    parser.add_argument("--num_vertex_samples", type=int, default=1024)
    parser.add_argument("--noise_std_min", type=float, default=0.005)
    parser.add_argument("--noise_std_max", type=float, default=0.020)
    parser.add_argument("--noise_types", default="gaussian,laplace")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    datalist_path = args.datalist if os.path.isabs(args.datalist) else os.path.join(ROOT, args.datalist)
    train_root = args.train_root if os.path.isabs(args.train_root) else os.path.join(ROOT, args.train_root)
    gt_dir = args.gt_dir if os.path.isabs(args.gt_dir) else os.path.join(ROOT, args.gt_dir)
    noisy_dir = args.noisy_dir if os.path.isabs(args.noisy_dir) else os.path.join(ROOT, args.noisy_dir)
    meta_dir = args.meta_dir if os.path.isabs(args.meta_dir) else os.path.join(ROOT, args.meta_dir)

    with open(datalist_path, "r") as f:
        rel_paths = [line.strip() for line in f if line.strip()]

    noise_types = [x.strip() for x in args.noise_types.split(",") if x.strip()]
    if not noise_types:
        raise ValueError("noise_types cannot be empty")

    for idx, rel in enumerate(rel_paths):
        mesh_path = os.path.join(train_root, rel, args.mesh_name)
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(mesh_path)

        np.random.seed(args.seed + idx)
        rng = np.random.RandomState(args.seed + idx)
        vertices, faces = load_mesh(mesh_path)
        clean, _, _, _ = sample_vertex_groups(
            vertices=vertices,
            faces=faces,
            num_samples=args.num_points,
            num_vertex_samples=min(args.num_vertex_samples, args.num_points),
        )
        clean, center, scale = normalize_to_unit_sphere(clean)

        if len(rel_paths) > 1:
            t = idx / float(len(rel_paths) - 1)
        else:
            t = 0.0
        noise_std = args.noise_std_min + t * (args.noise_std_max - args.noise_std_min)
        noise_type = noise_types[idx % len(noise_types)]
        noisy = add_noise(clean, noise_std=noise_std, noise_type=noise_type, rng=rng)

        gt_out = os.path.join(gt_dir, rel, "clean.npy")
        noisy_out = os.path.join(noisy_dir, rel, "noisy.npy")
        os.makedirs(os.path.dirname(gt_out), exist_ok=True)
        os.makedirs(os.path.dirname(noisy_out), exist_ok=True)
        np.save(gt_out, clean.astype(np.float32))
        np.save(noisy_out, noisy.astype(np.float32))

        if not args.no_meta:
            meta_out = os.path.join(meta_dir, rel, "meta.json")
            os.makedirs(os.path.dirname(meta_out), exist_ok=True)
            with open(meta_out, "w") as f:
                json.dump(
                    {
                        "relative_key": rel,
                        "mesh_path": mesh_path,
                        "mesh_name": args.mesh_name,
                        "normalize_center": center.tolist(),
                        "normalize_scale": float(scale),
                        "noise_std": float(noise_std),
                        "noise_type": noise_type,
                        "seed": int(args.seed + idx),
                        "num_points": int(args.num_points),
                        "num_vertex_samples": int(min(args.num_vertex_samples, args.num_points)),
                    },
                    f,
                    indent=2,
                )
                f.write("\n")

    print(f"wrote {len(rel_paths)} validation samples")
    print(f"gt_dir: {gt_dir}")
    print(f"noisy_dir: {noisy_dir}")
    if not args.no_meta:
        print(f"meta_dir: {meta_dir}")


if __name__ == "__main__":
    main()
