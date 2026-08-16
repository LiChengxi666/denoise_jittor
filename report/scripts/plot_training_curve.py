#!/usr/bin/env python3
"""Build vm_strong training curve from synced logs and known checkpoint val losses."""

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EPOCH_LINE = re.compile(
    r"Epoch\s+(\d+)/\d+:\s+train_loss=([0-9.eE+-]+)\s+val_loss=([0-9.eE+-]+)"
)


def read_metrics(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty metrics file: {path}")
    return {
        "epoch": [int(float(r["epoch"])) for r in rows],
        "train": [float(r["train/loss_sum"]) for r in rows],
        "val": [float(r["val/loss_sum"]) for r in rows],
    }


def parse_train_log(path: Path):
    """Optional cross-check: parse Epoch lines from train.log."""
    if not path.exists():
        return None
    epochs, train, val = [], [], []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            match = EPOCH_LINE.search(line)
            if not match:
                continue
            epochs.append(int(match.group(1)))
            train.append(float(match.group(2)))
            val.append(float(match.group(3)))
    if not epochs:
        return None
    return {"epoch": epochs, "train": train, "val": val}


def read_known_checkpoints(path: Path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    points = []
    for row in rows:
        epoch = row.get("epoch", "").strip()
        if not epoch:
            continue
        val_key = "val_loss_sum" if "val_loss_sum" in row else "val"
        points.append({
            "epoch": int(float(epoch)),
            "val": float(row[val_key]),
            "checkpoint": row.get("checkpoint", ""),
        })
    return sorted(points, key=lambda p: p["epoch"])


def fit_monotone_decay(epochs, values, total_epochs=80):
    if len(epochs) < 2:
        return None, None
    t = np.array(epochs, dtype=float)
    y = np.array(values, dtype=float)
    t0, y0 = t[0], y[0]
    c = max(float(min(y)) * 0.92, 0.012)
    best_sse = float("inf")
    best_k = None
    for k in np.linspace(0.005, 0.08, 400):
        pred = c + (y0 - c) * np.exp(-k * (t - t0))
        sse = float(np.sum((pred - y) ** 2))
        if sse < best_sse:
            best_sse = sse
            best_k = k
    if best_k is None:
        return None, None
    fit_x = np.arange(int(min(epochs)), total_epochs + 1)
    fit_y = c + (y0 - c) * np.exp(-best_k * (fit_x - t0))
    return fit_x, np.maximum(fit_y, c)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("../starter_code/logs/vm_strong/metrics.csv"),
    )
    parser.add_argument(
        "--train-log",
        type=Path,
        default=Path("../starter_code/logs/vm_strong/train.log"),
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        default=Path("data/checkpoint_val_loss_known.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("figures/training_curve.pdf"))
    parser.add_argument("--total-epochs", type=int, default=80)
    args = parser.parse_args()

    metrics = read_metrics(args.metrics)
    log_metrics = parse_train_log(args.train_log)
    if log_metrics and len(log_metrics["epoch"]) != len(metrics["epoch"]):
        print(f"note: train.log has {len(log_metrics['epoch'])} epochs, "
              f"metrics.csv has {len(metrics['epoch'])}; using metrics.csv for plot")

    known = read_known_checkpoints(args.checkpoints)
    ckpt_only = [p for p in known if p["epoch"] >= 10]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 9.5,
        "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(7.6, 3.65))

    ax.axvspan(0.5, 5.5, color="#DDEBF7", alpha=0.65, label="warm-up (epoch 1-5)")
    ax.plot(metrics["epoch"], metrics["train"], "o-", color="#2F75B5", lw=1.8,
            markersize=4.0, label="train loss (synced log)")
    ax.plot(metrics["epoch"], metrics["val"], "s-", color="#ED7D31", lw=1.8,
            markersize=4.0, label="val loss (synced log)")

    if ckpt_only:
        ax.plot([p["epoch"] for p in ckpt_only], [p["val"] for p in ckpt_only],
                "D", color="#C00000", ms=5.5, zorder=5,
                label="known checkpoint val")

        fit_epochs = [p["epoch"] for p in ckpt_only]
        fit_vals = [p["val"] for p in ckpt_only]
        fit_x, fit_y = fit_monotone_decay(fit_epochs, fit_vals, total_epochs=args.total_epochs)
        if fit_x is not None:
            ax.plot(fit_x, fit_y, "-", color="#70AD47", lw=2.0, alpha=0.85,
                    label="fitted val trend (through known anchors)")

    best_log_idx = min(range(len(metrics["val"])), key=metrics["val"].__getitem__)
    best_log_epoch = metrics["epoch"][best_log_idx]
    best_log_value = metrics["val"][best_log_idx]
    ax.annotate(f"log best {best_log_value:.6f}\n(epoch {best_log_epoch})",
                xy=(best_log_epoch, best_log_value), xytext=(12, 0.034),
                arrowprops={"arrowstyle": "->", "color": "#17365D", "lw": 1.0},
                color="#17365D", ha="left", fontsize=8.3)

    if ckpt_only:
        best_ckpt = min(ckpt_only, key=lambda p: p["val"])
        ax.annotate(f"known min {best_ckpt['val']:.6f}\n(epoch {best_ckpt['epoch']})",
                    xy=(best_ckpt["epoch"], best_ckpt["val"]),
                    xytext=(best_ckpt["epoch"] - 28, best_ckpt["val"] + 0.0035),
                    arrowprops={"arrowstyle": "->", "color": "#C00000", "lw": 1.0},
                    color="#C00000", ha="left", fontsize=8.3)

    ax.set_yscale("log")
    ax.set_xlim(0.7, max(args.total_epochs, max(metrics["epoch"])) + 2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted loss (log scale)")
    ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.45)
    ax.legend(loc="upper right", frameon=True, fontsize=7.6, ncol=1)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
