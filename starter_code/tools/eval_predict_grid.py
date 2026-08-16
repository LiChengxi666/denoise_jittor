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


def _parse_float(output, patterns):
    for pattern in patterns:
        match = re.search(pattern, output)
        if match:
            return float(match.group(1))
    return None


def parse_metrics(output):
    return {
        "score": _parse_float(output, [
            r"最终得分.*?([0-9]+(?:\.[0-9]+)?)\s*/\s*100",
            r"final_score\s*=\s*([0-9]+(?:\.[0-9]+)?)",
        ]),
        "cd_score": _parse_float(output, [
            r"CD 得分:\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*100",
            r"CD_score\s*=\s*([0-9]+(?:\.[0-9]+)?)",
        ]),
        "p2s_score": _parse_float(output, [
            r"P2S 得分:\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*100",
            r"P2S_score\s*=\s*([0-9]+(?:\.[0-9]+)?)",
        ]),
        "mean_cd_pred": _parse_float(output, [
            r"平均 CD_pred:\s*([0-9]+(?:\.[0-9]+)?)",
            r"mean_cd_pred\s*=\s*([0-9]+(?:\.[0-9]+)?)",
        ]),
        "mean_cd_noisy": _parse_float(output, [
            r"平均 CD_noisy:\s*([0-9]+(?:\.[0-9]+)?)",
            r"mean_cd_noisy\s*=\s*([0-9]+(?:\.[0-9]+)?)",
        ]),
        "mean_cd_pred_to_gt": _parse_float(output, [
            r"平均 CD_pred_to_gt:\s*([0-9]+(?:\.[0-9]+)?)",
            r"mean_cd_pred_to_gt\s*=\s*([0-9]+(?:\.[0-9]+)?)",
        ]),
        "mean_cd_gt_to_pred": _parse_float(output, [
            r"平均 CD_gt_to_pred:\s*([0-9]+(?:\.[0-9]+)?)",
            r"mean_cd_gt_to_pred\s*=\s*([0-9]+(?:\.[0-9]+)?)",
        ]),
        "mean_cd_noisy_to_gt": _parse_float(output, [
            r"平均 CD_noisy_to_gt:\s*([0-9]+(?:\.[0-9]+)?)",
            r"mean_cd_noisy_to_gt\s*=\s*([0-9]+(?:\.[0-9]+)?)",
        ]),
        "mean_cd_gt_to_noisy": _parse_float(output, [
            r"平均 CD_gt_to_noisy:\s*([0-9]+(?:\.[0-9]+)?)",
            r"mean_cd_gt_to_noisy\s*=\s*([0-9]+(?:\.[0-9]+)?)",
        ]),
        "mean_p2s_pred": _parse_float(output, [
            r"平均 P2S_pred:\s*([0-9]+(?:\.[0-9]+)?)",
            r"mean_p2s_pred\s*=\s*([0-9]+(?:\.[0-9]+)?)",
        ]),
        "mean_p2s_noisy": _parse_float(output, [
            r"平均 P2S_noisy:\s*([0-9]+(?:\.[0-9]+)?)",
            r"mean_p2s_noisy\s*=\s*([0-9]+(?:\.[0-9]+)?)",
        ]),
    }


def tail(text, n=1200):
    return text[-n:].replace("\n", "\\n") if text else ""


def main():
    parser = argparse.ArgumentParser(description="Grid-search validation inference settings.")
    parser.add_argument("--checkpoints", default="experiments/vm_strong/checkpoint_*.pkl")
    parser.add_argument("--base_task", default="configs/task/predict_val.yaml")
    parser.add_argument("--base_model", default="configs/model/vm_strong.yaml")
    parser.add_argument("--gt_dir", default="val_gt")
    parser.add_argument("--noisy_dir", default="val_noisy")
    parser.add_argument("--mesh_dir", default="../dataset_train")
    parser.add_argument("--meta_dir", default="val_meta")
    parser.add_argument("--p2s_normalize", choices=["ref_gt", "meta", "none"], default="meta")
    parser.add_argument("--csv_path", default="experiments/vm_strong/val_grid.csv")
    parser.add_argument("--step_sizes", default="0.6,0.7,0.8,0.9")
    parser.add_argument("--predict_steps", default="1,2,3")
    parser.add_argument("--inner_steps", default="3,4,5")
    parser.add_argument("--patch_sizes", default="1200")
    parser.add_argument("--alpha_blends", default="0.85,0.95,1.0")
    parser.add_argument("--momentums", default="0.0,0.3,0.6")
    parser.add_argument("--step_decays", default="linear")
    parser.add_argument("--p2s_floor", type=float, default=80.37)
    parser.add_argument("--workers", default="8")
    parser.add_argument("--allow_cd_only", action="store_true")
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
    momentums = parse_list(args.momentums, float)
    step_decays = parse_list(args.step_decays, str)

    grid_dir = os.path.join(ROOT, ".grid_tmp")
    os.makedirs(grid_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.join(ROOT, args.csv_path)), exist_ok=True)

    rows = []
    run_id = 0
    for ckpt, step_size, pred_steps, inner, patch_size, alpha, momentum, step_decay in product(
        checkpoints, step_sizes, predict_steps, inner_steps, patch_sizes, alpha_blends, momentums, step_decays
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
        if momentum is not None:
            model_cfg["predict_momentum"] = momentum
        if step_decay is not None:
            model_cfg["predict_step_decay"] = step_decay

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
        metrics = parse_metrics("")
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
                "--p2s_normalize",
                args.p2s_normalize,
            ]
            if args.p2s_normalize == "meta":
                eval_cmd.extend(["--meta_dir", args.meta_dir])
            eval_code, eval_out = run_cmd(eval_cmd)
            if eval_code == 0:
                metrics = parse_metrics(eval_out)

        missing_required_p2s = (
            args.p2s_normalize != "none"
            and not args.allow_cd_only
            and metrics["score"] is not None
            and metrics["p2s_score"] is None
        )

        p2s_pass = (
            metrics["p2s_score"] is not None
            and float(metrics["p2s_score"]) >= float(args.p2s_floor)
        )
        cd_priority_score = ""
        if not missing_required_p2s and metrics["cd_score"] is not None and p2s_pass:
            cd_priority_score = 0.7 * metrics["cd_score"] + 0.3 * metrics["p2s_score"]

        status = "ok" if metrics["score"] is not None and not missing_required_p2s else "failed"
        if missing_required_p2s:
            failure_tail = "P2S metric missing while p2s_normalize is not none; pass --allow_cd_only to accept CD-only runs. "
            failure_tail += tail(pred_out + "\n" + eval_out)
        else:
            failure_tail = "" if status == "ok" else tail(pred_out + "\n" + eval_out)

        row = {
            "run_id": run_id,
            "checkpoint": os.path.relpath(ckpt, ROOT),
            "score": metrics["score"] if status == "ok" and metrics["score"] is not None else "",
            "cd_priority_score": cd_priority_score,
            "cd_score": metrics["cd_score"] if metrics["cd_score"] is not None else "",
            "p2s_score": metrics["p2s_score"] if metrics["p2s_score"] is not None else "",
            "mean_cd_pred": metrics["mean_cd_pred"] if metrics["mean_cd_pred"] is not None else "",
            "mean_cd_noisy": metrics["mean_cd_noisy"] if metrics["mean_cd_noisy"] is not None else "",
            "mean_cd_pred_to_gt": metrics["mean_cd_pred_to_gt"] if metrics["mean_cd_pred_to_gt"] is not None else "",
            "mean_cd_gt_to_pred": metrics["mean_cd_gt_to_pred"] if metrics["mean_cd_gt_to_pred"] is not None else "",
            "mean_cd_noisy_to_gt": metrics["mean_cd_noisy_to_gt"] if metrics["mean_cd_noisy_to_gt"] is not None else "",
            "mean_cd_gt_to_noisy": metrics["mean_cd_gt_to_noisy"] if metrics["mean_cd_gt_to_noisy"] is not None else "",
            "mean_p2s_pred": metrics["mean_p2s_pred"] if metrics["mean_p2s_pred"] is not None else "",
            "mean_p2s_noisy": metrics["mean_p2s_noisy"] if metrics["mean_p2s_noisy"] is not None else "",
            "p2s_floor": args.p2s_floor,
            "p2s_pass": p2s_pass,
            "predict_step_size": model_cfg.get("predict_step_size", ""),
            "predict_num_steps": model_cfg.get("predict_num_steps", ""),
            "denoise_inner_steps": model_cfg.get("denoise_inner_steps", ""),
            "predict_patch_size": model_cfg.get("predict_patch_size", ""),
            "alpha_blend": model_cfg.get("alpha_blend", ""),
            "predict_momentum": model_cfg.get("predict_momentum", ""),
            "predict_step_decay": model_cfg.get("predict_step_decay", ""),
            "pred_dir": pred_dir if args.keep_predictions else "",
            "status": status,
            "failure_tail": failure_tail,
        }
        rows.append(row)
        print(row)

        with open(os.path.join(ROOT, args.csv_path), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        if not args.keep_predictions:
            shutil.rmtree(os.path.join(ROOT, pred_dir), ignore_errors=True)

        if status != "ok":
            print(pred_out[-4000:])
            print(eval_out[-4000:])

    ranked = sorted([r for r in rows if r["cd_priority_score"] != ""], key=lambda r: float(r["cd_priority_score"]), reverse=True)
    if ranked:
        print("best_by_cd_priority:", ranked[0])
    ranked_final = sorted([r for r in rows if r["score"] != ""], key=lambda r: float(r["score"]), reverse=True)
    if ranked_final:
        print("best_by_final_score:", ranked_final[0])
    print(f"wrote {args.csv_path}")


if __name__ == "__main__":
    main()
