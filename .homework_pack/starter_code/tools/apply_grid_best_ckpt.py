#!/usr/bin/env python3
"""Pick best grid row (p2s >= floor) and update predict task load_ckpt."""

import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="eval_predict_grid output csv")
    parser.add_argument(
        "--predict_task",
        default="configs/task/predict_vm_strong_indclean1600_ft.yaml",
    )
    parser.add_argument("--p2s_floor", type=float, default=80.37)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(ROOT, args.csv)
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    candidates = []
    for row in rows:
        if row.get("status") != "ok" or not row.get("score"):
            continue
        p2s = float(row["p2s_score"]) if row.get("p2s_score") not in (None, "") else None
        if p2s is not None and p2s < args.p2s_floor:
            continue
        candidates.append(row)

    if not candidates:
        print(f"no ok rows with p2s >= {args.p2s_floor} in {csv_path}", file=sys.stderr)
        sys.exit(1)

    best = max(candidates, key=lambda r: float(r["score"]))
    ckpt = best["checkpoint"]
    print(f"best checkpoint: {ckpt}")
    print(f"score={best['score']} cd={best.get('cd_score')} p2s={best.get('p2s_score')}")

    task_path = args.predict_task if os.path.isabs(args.predict_task) else os.path.join(ROOT, args.predict_task)
    with open(task_path, "r") as f:
        lines = f.readlines()

    out = []
    replaced = False
    for line in lines:
        if line.startswith("load_ckpt:"):
            out.append(f"load_ckpt: {ckpt}\n")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        print(f"load_ckpt: not found in {task_path}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print(f"would update {task_path}")
        return

    with open(task_path, "w") as f:
        f.writelines(out)
    print(f"updated {task_path}")


if __name__ == "__main__":
    main()
