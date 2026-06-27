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

SPRINT_SETTINGS = [
    {
        "predict_step_size": 0.80,
        "predict_num_steps": 2,
        "denoise_inner_steps": 4,
        "alpha_blend": 1.00,
        "predict_momentum": 0.00,
        "predict_step_decay": "none",
    },
    {
        "predict_step_size": 0.80,
        "predict_num_steps": 2,
        "denoise_inner_steps": 4,
        "alpha_blend": 1.00,
        "predict_momentum": 0.60,
        "predict_step_decay": "linear",
    },
    {
        "predict_step_size": 0.65,
        "predict_num_steps": 2,
        "denoise_inner_steps": 4,
        "alpha_blend": 0.90,
        "predict_momentum": 0.60,
        "predict_step_decay": "linear",
    },
    {
        "predict_step_size": 0.90,
        "predict_num_steps": 2,
        "denoise_inner_steps": 4,
        "alpha_blend": 1.00,
        "predict_momentum": 0.60,
        "predict_step_decay": "linear",
    },
    {
        "predict_step_size": 0.80,
        "predict_num_steps": 1,
        "denoise_inner_steps": 4,
        "alpha_blend": 1.00,
        "predict_momentum": 0.60,
        "predict_step_decay": "linear",
    },
    {
        "predict_step_size": 0.65,
        "predict_num_steps": 3,
        "denoise_inner_steps": 4,
        "alpha_blend": 1.00,
        "predict_momentum": 0.60,
        "predict_step_decay": "linear",
    },
]


def parse_list(text, cast):
    if text is None or text == "":
        return [None]
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def dump_yaml(path, data):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
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
        "mean_p2s_pred": _parse_float(output, [
            r"平均 P2S_pred:\s*([0-9]+(?:\.[0-9]+)?)",
            r"mean_p2s_pred\s*=\s*([0-9]+(?:\.[0-9]+)?)",
        ]),
        "mean_p2s_noisy": _parse_float(output, [
            r"平均 P2S_noisy:\s*([0-9]+(?:\.[0-9]+)?)",
            r"mean_p2s_noisy\s*=\s*([0-9]+(?:\.[0-9]+)?)",
        ]),
    }


def select_score(metrics, allow_cd_only=False):
    cd_score = metrics["cd_score"]
    p2s_score = metrics["p2s_score"]
    if cd_score is None:
        return "", "", "missing_cd"
    if p2s_score is None:
        if allow_cd_only:
            return cd_score, "cd_only", None
        return "", "", "missing_p2s"
    score = metrics["score"]
    if score is None:
        score = 0.5 * (cd_score + p2s_score)
    return score, "official_equal", None


def build_sprint_settings(patch_weight_gammas):
    return [
        {**setting, "predict_patch_weight_gamma": gamma}
        for setting, gamma in product(SPRINT_SETTINGS, patch_weight_gammas)
    ]


def resolve_grid_mode(grid_mode, checkpoint_count, from_screen_csv=False):
    if grid_mode != "auto":
        return grid_mode
    if from_screen_csv:
        return "sprint"
    return "sprint" if checkpoint_count == 1 else "screen"


def build_screen_settings():
    return [{**SPRINT_SETTINGS[0], "predict_patch_weight_gamma": 1.0}]


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(ROOT, path)


def load_top_checkpoints(csv_path, top_k):
    if top_k <= 0:
        raise ValueError("screen_top_k must be positive")
    with open(resolve_path(csv_path), "r", newline="") as f:
        rows = list(csv.DictReader(f))

    ranked = sorted(
        (
            row for row in rows
            if row.get("status") == "ok"
            and row.get("score_mode") == "official_equal"
            and row.get("score") not in (None, "")
        ),
        key=lambda row: float(row["score"]),
        reverse=True,
    )
    checkpoints = []
    seen = set()
    for row in ranked:
        checkpoint = resolve_path(row["checkpoint"])
        if checkpoint in seen:
            continue
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"screened checkpoint not found: {checkpoint}")
        seen.add(checkpoint)
        checkpoints.append(checkpoint)
        if len(checkpoints) == top_k:
            break
    if not checkpoints:
        raise ValueError(f"no official-score checkpoints found in {csv_path}")
    return checkpoints


def export_selected_artifacts(
    best_row,
    model_config,
    model_path,
    checkpoint_path,
    allow_cd_only_export=False,
):
    if not model_path and not checkpoint_path:
        return
    if not model_path or not checkpoint_path:
        raise ValueError(
            "export_best_model and export_best_checkpoint must be provided together"
        )
    if best_row["score_mode"] != "official_equal" and not allow_cd_only_export:
        raise ValueError("refusing to export a CD-only result as the selected submission")

    source_checkpoint = resolve_path(best_row["checkpoint"])
    target_checkpoint = resolve_path(checkpoint_path)
    target_model = resolve_path(model_path)
    if not os.path.isfile(source_checkpoint):
        raise FileNotFoundError(f"selected checkpoint not found: {source_checkpoint}")

    checkpoint_dir = os.path.dirname(target_checkpoint)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    if os.path.realpath(source_checkpoint) != os.path.realpath(target_checkpoint):
        shutil.copy2(source_checkpoint, target_checkpoint)
    dump_yaml(target_model, model_config)
    print(f"exported_selected_checkpoint: {target_checkpoint}")
    print(f"exported_selected_model: {target_model}")


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
    parser.add_argument(
        "--grid_mode",
        choices=["auto", "screen", "sprint", "cartesian"],
        default="auto",
        help="auto uses sprint for one checkpoint and anchor-only screen for multiple checkpoints",
    )
    parser.add_argument(
        "--screen_csv",
        default="",
        help="stage-one CSV used to select the top checkpoints for a sprint grid",
    )
    parser.add_argument("--screen_top_k", type=int, default=2)
    parser.add_argument("--step_sizes", default="0.65,0.8")
    parser.add_argument("--predict_steps", default="2,3")
    parser.add_argument("--inner_steps", default="4")
    parser.add_argument("--patch_sizes", default="1200")
    parser.add_argument("--alpha_blends", default="0.9,1.0")
    parser.add_argument("--momentums", default="0.0,0.6")
    parser.add_argument("--patch_weight_gammas", default="1.0,4.0,9.0")
    parser.add_argument("--step_decays", default="linear")
    parser.add_argument("--workers", default="8")
    parser.add_argument("--allow_cd_only", action="store_true")
    parser.add_argument("--keep_predictions", action="store_true")
    parser.add_argument("--export_best_model", default="")
    parser.add_argument("--export_best_checkpoint", default="")
    parser.add_argument("--allow_cd_only_export", action="store_true")
    args = parser.parse_args()

    if args.screen_csv:
        checkpoints = load_top_checkpoints(args.screen_csv, args.screen_top_k)
    else:
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
    patch_weight_gammas = parse_list(args.patch_weight_gammas, float)
    step_decays = parse_list(args.step_decays, str)

    grid_dir = os.path.join(ROOT, ".grid_tmp")
    os.makedirs(grid_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.join(ROOT, args.csv_path)), exist_ok=True)

    rows = []
    model_configs = {}
    run_id = 0
    grid_mode = resolve_grid_mode(
        args.grid_mode,
        len(checkpoints),
        from_screen_csv=bool(args.screen_csv),
    )
    print(
        f"grid_mode={grid_mode}, checkpoints={len(checkpoints)}, "
        f"screen_csv={args.screen_csv or '-'}"
    )
    if grid_mode == "screen":
        settings = build_screen_settings()
    elif grid_mode == "sprint":
        settings = build_sprint_settings(patch_weight_gammas)
    else:
        settings = [
            {
                "predict_step_size": step_size,
                "predict_num_steps": pred_steps,
                "denoise_inner_steps": inner,
                "alpha_blend": alpha,
                "predict_momentum": momentum,
                "predict_patch_weight_gamma": gamma,
                "predict_step_decay": step_decay,
            }
            for step_size, pred_steps, inner, alpha, momentum, gamma, step_decay in product(
                step_sizes,
                predict_steps,
                inner_steps,
                alpha_blends,
                momentums,
                patch_weight_gammas,
                step_decays,
            )
        ]

    for ckpt, setting, patch_size in product(checkpoints, settings, patch_sizes):
        run_id += 1
        model_cfg = dict(base_model)
        for name, value in setting.items():
            if value is not None:
                model_cfg[name] = value
        if patch_size is not None:
            model_cfg["predict_patch_size"] = patch_size
        model_configs[run_id] = dict(model_cfg)

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

        official_score, score_mode, score_error = select_score(
            metrics,
            allow_cd_only=args.allow_cd_only or args.p2s_normalize == "none",
        )
        status = "ok" if score_error is None else "failed"
        if score_error == "missing_p2s":
            failure_tail = "P2S metric missing while p2s_normalize is not none; pass --allow_cd_only to accept CD-only runs. "
            failure_tail += tail(pred_out + "\n" + eval_out)
        else:
            failure_tail = "" if status == "ok" else tail(pred_out + "\n" + eval_out)

        row = {
            "run_id": run_id,
            "checkpoint": os.path.relpath(ckpt, ROOT),
            "grid_mode": grid_mode,
            "score": official_score if status == "ok" else "",
            "score_mode": score_mode,
            "cd_score": metrics["cd_score"] if metrics["cd_score"] is not None else "",
            "p2s_score": metrics["p2s_score"] if metrics["p2s_score"] is not None else "",
            "mean_cd_pred": metrics["mean_cd_pred"] if metrics["mean_cd_pred"] is not None else "",
            "mean_cd_noisy": metrics["mean_cd_noisy"] if metrics["mean_cd_noisy"] is not None else "",
            "mean_p2s_pred": metrics["mean_p2s_pred"] if metrics["mean_p2s_pred"] is not None else "",
            "mean_p2s_noisy": metrics["mean_p2s_noisy"] if metrics["mean_p2s_noisy"] is not None else "",
            "predict_step_size": model_cfg.get("predict_step_size", ""),
            "predict_num_steps": model_cfg.get("predict_num_steps", ""),
            "denoise_inner_steps": model_cfg.get("denoise_inner_steps", ""),
            "predict_patch_size": model_cfg.get("predict_patch_size", ""),
            "predict_patch_weight_gamma": model_cfg.get("predict_patch_weight_gamma", ""),
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

    ranked_final = sorted([r for r in rows if r["score"] != ""], key=lambda r: float(r["score"]), reverse=True)
    if ranked_final:
        rank_label = (
            "best_by_official_score"
            if ranked_final[0]["score_mode"] == "official_equal"
            else "best_by_cd_score"
        )
        print(f"{rank_label}:", ranked_final[0])
        export_selected_artifacts(
            best_row=ranked_final[0],
            model_config=model_configs[ranked_final[0]["run_id"]],
            model_path=args.export_best_model,
            checkpoint_path=args.export_best_checkpoint,
            allow_cd_only_export=args.allow_cd_only_export,
        )
    elif args.export_best_model or args.export_best_checkpoint:
        raise RuntimeError("no successful run is available for selected artifact export")
    print(f"wrote {args.csv_path}")


if __name__ == "__main__":
    main()
