#!/usr/bin/env python
"""Evaluate checkpoints and inference hyperparameters on the fixed validation set."""

import argparse
import csv
import glob
import os
import re
import shutil
import subprocess
import sys
from itertools import product

import yaml


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_list(text, cast):
    if text is None or text == "":
        return [None]
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def dump_yaml(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def run_cmd(cmd):
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.returncode, proc.stdout


def parse_score(output):
    patterns = [
        r"最终得分.*?([0-9]+(?:\.[0-9]+)?)\s*/\s*100",
        r"final_score\s*=\s*([0-9]+(?:\.[0-9]+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, output)
        if match:
            return float(match.group(1))
    return None


def main():
    parser = argparse.ArgumentParser(description="Grid-search validation inference settings.")
    parser.add_argument("--checkpoints", default="experiments/vm_strong/checkpoint_*.pkl")
    parser.add_argument("--base_task", default="configs/task/predict_val.yaml")
    parser.add_argument("--base_model", default="configs/model/vm_strong.yaml")
    parser.add_argument("--gt_dir", default="val_gt")
    parser.add_argument("--noisy_dir", default="val_noisy")
    parser.add_argument("--mesh_dir", default="../dataset_train")
    parser.add_argument("--csv_path", default="experiments/vm_strong/val_grid.csv")
    parser.add_argument("--step_sizes", default="0.6,0.8,1.0")
    parser.add_argument("--predict_steps", default="1,2,3")
    parser.add_argument("--inner_steps", default="4")
    parser.add_argument("--patch_sizes", default="1000,1200")
    parser.add_argument("--alpha_blends", default="0.75,1.0,1.25")
    parser.add_argument("--workers", default="8")
    parser.add_argument("--keep_predictions", action="store_true")
    args = parser.parse_args()

    checkpoints = sorted(glob.glob(os.path.join(ROOT, args.checkpoints)))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints matched {args.checkpoints}")

    base_task = load_yaml(os.path.join(ROOT, args.base_task))
    base_model = load_yaml(os.path.join(ROOT, args.base_model))

    step_sizes = parse_list(args.step_sizes, float)
    predict_steps = parse_list(args.predict_steps, int)
    inner_steps = parse_list(args.inner_steps, int)
    patch_sizes = parse_list(args.patch_sizes, int)
    alpha_blends = parse_list(args.alpha_blends, float)

    grid_dir = os.path.join(ROOT, ".grid_tmp")
    os.makedirs(grid_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.join(ROOT, args.csv_path)), exist_ok=True)

    rows = []
    run_id = 0
    for ckpt, step_size, pred_steps, inner, patch_size, alpha in product(
        checkpoints, step_sizes, predict_steps, inner_steps, patch_sizes, alpha_blends
    ):
        run_id += 1
        model_cfg = dict(base_model)
        if step_size is not None:
            model_cfg["predict_step_size"] = step_size
        if pred_steps is not None:
            model_cfg["predict_num_steps"] = pred_steps
        if inner is not None:
            model_cfg["denoise_inner_steps"] = inner
        if patch_size is not None:
            model_cfg["predict_patch_size"] = patch_size
        if alpha is not None:
            model_cfg["alpha_blend"] = alpha

        pred_dir = os.path.join(".grid_tmp", f"pred_{run_id:04d}")
        task_cfg = dict(base_task)
        task_cfg["load_ckpt"] = ckpt
        task_cfg["components"] = dict(base_task["components"])
        task_cfg["components"]["model"] = os.path.join(grid_dir, f"model_{run_id:04d}.yaml")
        task_cfg["writer"] = dict(base_task["writer"])
        task_cfg["writer"]["save_dir"] = pred_dir

        model_path = os.path.join(grid_dir, f"model_{run_id:04d}.yaml")
        task_path = os.path.join(grid_dir, f"task_{run_id:04d}.yaml")
        dump_yaml(model_path, model_cfg)
        dump_yaml(task_path, task_cfg)

        code, pred_out = run_cmd([sys.executable, "run.py", "--task", task_path])
        score = None
        eval_out = ""
        if code == 0:
            eval_cmd = [
                sys.executable,
                "evaluate.py",
                "--pred_dir",
                pred_dir,
                "--gt_dir",
                args.gt_dir,
                "--noisy_dir",
                args.noisy_dir,
                "--mesh_dir",
                args.mesh_dir,
                "--workers",
                args.workers,
            ]
            eval_code, eval_out = run_cmd(eval_cmd)
            if eval_code == 0:
                score = parse_score(eval_out)

        row = {
            "run_id": run_id,
            "checkpoint": os.path.relpath(ckpt, ROOT),
            "score": score if score is not None else "",
            "predict_step_size": model_cfg.get("predict_step_size", ""),
            "predict_num_steps": model_cfg.get("predict_num_steps", ""),
            "denoise_inner_steps": model_cfg.get("denoise_inner_steps", ""),
            "predict_patch_size": model_cfg.get("predict_patch_size", ""),
            "alpha_blend": model_cfg.get("alpha_blend", ""),
            "pred_dir": pred_dir if args.keep_predictions else "",
            "status": "ok" if score is not None else "failed",
        }
        rows.append(row)
        print(row)

        with open(os.path.join(ROOT, args.csv_path), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        if not args.keep_predictions:
            shutil.rmtree(os.path.join(ROOT, pred_dir), ignore_errors=True)

        if score is None:
            print(pred_out[-4000:])
            print(eval_out[-4000:])

    ranked = sorted([r for r in rows if r["score"] != ""], key=lambda r: float(r["score"]), reverse=True)
    if ranked:
        print("best:", ranked[0])
    print(f"wrote {args.csv_path}")


if __name__ == "__main__":
    main()
