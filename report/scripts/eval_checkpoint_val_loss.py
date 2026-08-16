#!/usr/bin/env python3
"""Evaluate vm_strong checkpoints on the training validate split and export val loss."""

import argparse
import csv
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STARTER = os.path.join(ROOT, "starter_code")
sys.path.insert(0, STARTER)

import jittor as jt
import yaml
from tqdm import tqdm

from src.data.dataset import PCDatasetModule
from src.data.transform import Transform
from src.model.parse import get_model
from src.system.parse import get_system


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def checkpoint_epoch(path: str):
    name = os.path.basename(path)
    if name.endswith("_best.pkl"):
        return None, "best"
    if name.endswith("_last.pkl"):
        return None, "last"
    match = re.fullmatch(r"checkpoint_(\d+)\.pkl", name)
    if match:
        # saved at end of epoch (index + 1)
        return int(match.group(1)) + 1, "epoch"
    return None, "unknown"


def mean_val_loss(system, dataset_module):
    system.model.eval()
    system.model.set_predict(False)
    validate_dataloader = dataset_module.validate_dataloader()
    if validate_dataloader is None:
        raise RuntimeError("validate_dataloader is None")
    system.on_validation_epoch_start()
    dataloaders = validate_dataloader.items() if isinstance(validate_dataloader, dict) else [("default", validate_dataloader)]
    for _name, dataloader in dataloaders:
        for batch in tqdm(dataloader, desc="validate", leave=False):
            system.on_validation_batch_start()
            system.validation_step(batch)
            system.on_validation_batch_end()
    mean_loss, _summary = system.on_validation_epoch_end()
    return float(mean_loss)


def build_system(ckpt_path: str):
    task = load_yaml(os.path.join(STARTER, "configs/task/train_vm.yaml"))
    components = task["components"]
    data_config = load_yaml(os.path.join(STARTER, "configs/data", f"{components['data']}.yaml"))
    transform_config = load_yaml(os.path.join(STARTER, "configs/transform", f"{components['transform']}.yaml"))
    model_config = load_yaml(os.path.join(STARTER, "configs/model", f"{components['model']}.yaml"))
    system_config = load_yaml(os.path.join(STARTER, "configs/system", f"{components['system']}.yaml"))

    from src.data.dataset import DatasetConfig

    train_dataset_config = DatasetConfig.parse(**data_config["train_dataset"])
    validate_dataset_config = DatasetConfig.parse(**data_config["validate_dataset"]).split_by_cls()
    model = get_model(model_config=model_config, transform_config=transform_config)
    model.load(ckpt_path)
    dataset_module = PCDatasetModule(
        process_fn=model._process_fn,
        train_dataset_config=train_dataset_config,
        validate_dataset_config=validate_dataset_config,
        predict_dataset_config=None,
        train_transform=model.get_train_transform(),
        validate_transform=model.get_validate_transform(),
        predict_transform=model.get_predict_transform(),
    )
    system = get_system(
        dataset_module=dataset_module,
        model=model,
        loss_config=task.get("loss"),
        optimizer_config=task.get("optimizer"),
        trainer_config=task.get("trainer"),
        writer=None,
        config_snapshot=None,
        **system_config,
    )
    return system, dataset_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = []
    for ckpt in args.checkpoints:
        ckpt = os.path.abspath(ckpt)
        if not os.path.isfile(ckpt):
            print(f"skip missing: {ckpt}")
            continue
        epoch, kind = checkpoint_epoch(ckpt)
        print(f"evaluating {ckpt} ({kind}, epoch={epoch})")
        system, dataset_module = build_system(ckpt)
        val_loss = mean_val_loss(system, dataset_module)
        rows.append({
            "checkpoint": ckpt,
            "kind": kind,
            "epoch": epoch if epoch is not None else "",
            "val_loss_sum": val_loss,
        })
        print(f"  val_loss_sum={val_loss:.6f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["checkpoint", "kind", "epoch", "val_loss_sum"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
