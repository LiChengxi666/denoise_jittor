from collections import defaultdict
from jittor import optim
from typing import Dict, List, Optional
from tqdm import tqdm

import csv
import json
import jittor as jt
import math
import os
import time

from ..data.asset import Asset
from ..data.dataset import PCDatasetModule
from ..model.spec import ModelSpec

def _get_item(x):
    if isinstance(x, jt.Var):
        return x.item()
    return x

def _mean(values):
    return sum(values) / len(values) if values else None

def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)

def get_optimizer(optimizer_config, model):
    optimizer_config = dict(optimizer_config)
    __target__ = optimizer_config.pop('__target__')
    MAPPING = {
        'sgd': optim.SGD,
        'adam': optim.Adam,
    }
    if __target__ not in MAPPING:
        raise ValueError(f"unsupported optimizer: {__target__}")
    OptimizerClass = MAPPING[__target__]
    optimizer = OptimizerClass(model.parameters(), **optimizer_config)
    return optimizer

class DummyWriter():
    
    def __init__(self):
        pass
    
    def write(self, batch, prediction: List[Dict], dataset_module: Optional[PCDatasetModule]=None):
        pass

class TrainingLogger():

    def __init__(
        self,
        log_dir: str="logs",
        experiment_name: str="default",
        enabled: bool=True,
        metrics_csv: str="metrics.csv",
        config_snapshot=None,
        log_config: bool=True,
    ):
        self.enabled = enabled
        self.log_root = log_dir
        self.experiment_name = experiment_name
        self.log_dir = os.path.join(log_dir, experiment_name)
        self.metrics_csv = metrics_csv
        self.rows = []
        self.fieldnames = []
        if not self.enabled:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, "train.log")
        self.csv_path = os.path.join(self.log_dir, self.metrics_csv)
        self.latest_path = os.path.join(self.log_dir, "latest.json")
        if log_config and config_snapshot is not None:
            with open(os.path.join(self.log_dir, "config_snapshot.yaml"), "w") as f:
                json.dump(_jsonable(config_snapshot), f, indent=2, ensure_ascii=False)
                f.write("\n")
        self.log_text(f"Logger initialized: {self.log_dir}")

    def log_text(self, message: str):
        if not self.enabled:
            return
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def write_metrics(self, row: Dict):
        if not self.enabled:
            return
        clean_row = {}
        for k, v in row.items():
            if isinstance(v, jt.Var):
                v = _get_item(v)
            clean_row[k] = v
        self.rows.append(clean_row)
        keys = sorted({k for r in self.rows for k in r.keys()})
        preferred = [
            "epoch", "total_epochs", "lr", "batch_size", "train_batches",
            "val_batches", "epoch_seconds", "elapsed_seconds",
            "best_val_loss", "best_checkpoint", "last_checkpoint",
        ]
        self.fieldnames = [k for k in preferred if k in keys] + [k for k in keys if k not in preferred]
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)

    def write_latest(self, data: Dict):
        if not self.enabled:
            return
        with open(self.latest_path, "w") as f:
            json.dump(_jsonable(data), f, indent=2, ensure_ascii=False)
            f.write("\n")

class DummySystem():
    
    def __init__(
        self,
        dataset_module: PCDatasetModule,
        model: ModelSpec,
        loss_config=None,
        optimizer_config=None,
        trainer_config=None,
        writer: Optional[DummyWriter]=None,
        
        ckpt_save_dir: str="experiments",
        ckpt_save_name: str="checkpoint",
        config_snapshot=None,
    ):
        self.dataset_module = dataset_module
        self.model = model
        self.loss_config = loss_config
        self.ckpt_save_dir = ckpt_save_dir
        self.ckpt_save_name = ckpt_save_name
        self.writer = writer
        if trainer_config is None:
            trainer_config = {}
        self.epochs = trainer_config.get('epochs', 1)
        self.save_every = trainer_config.get('save_every', 1)
        self.save_best = trainer_config.get('save_best', True)
        self.save_last = trainer_config.get('save_last', True)
        self.validate_every = trainer_config.get('validate_every', 1)
        self.save_last_every = trainer_config.get('save_last_every', 1)
        self.log_every_n_steps = trainer_config.get('log_every_n_steps', 0)
        self.lr_scheduler_config = trainer_config.get('lr_scheduler', None)
        self.grad_clip_norm = trainer_config.get('grad_clip_norm', None)
        self.best_validation_loss = math.inf
        self.base_lr = None if optimizer_config is None else optimizer_config.get('lr', None)
        self.current_lr = self.base_lr
        experiment_name = trainer_config.get("experiment_name", os.path.basename(os.path.normpath(self.ckpt_save_dir)) or "default")
        self.logger = TrainingLogger(
            log_dir=trainer_config.get("log_dir", "logs"),
            experiment_name=experiment_name,
            enabled=trainer_config.get("log_to_file", True),
            metrics_csv=trainer_config.get("metrics_csv", "metrics.csv"),
            config_snapshot=config_snapshot,
            log_config=trainer_config.get("log_config", True),
        )
        self.training_started_at = None
        self.last_checkpoint_path = None
        self.best_checkpoint_path = None
        
        if optimizer_config is not None and model is not None:
            self.optimizer = get_optimizer(optimizer_config, model)
        else:
            self.optimizer = None
        
        self._validation_loss = defaultdict(list)
        self._train_loss = defaultdict(list)
        self._warned_grad_clip = False
        self.logger.log_text(
            "Training setup: "
            f"epochs={self.epochs}, save_every={self.save_every}, save_last={self.save_last}, "
            f"save_last_every={self.save_last_every}, validate_every={self.validate_every}, "
            f"base_lr={self.base_lr}, ckpt_dir={self.ckpt_save_dir}"
        )

    def _checkpoint_path(self, name: str) -> str:
        return os.path.join(self.ckpt_save_dir, f'{name}.pkl')

    def _save_checkpoint(self, path: str, reason: str):
        os.makedirs(self.ckpt_save_dir, exist_ok=True)
        self.model.save(path)
        print(f"\033[92mSaved checkpoint ({reason}): {path}\033[0m")
        self.logger.log_text(f"Saved checkpoint ({reason}): {path}")
    
    def forward(self, batch, validate: bool=False): # return loss sum
        loss_dict = self.model.training_step(batch)
        assert isinstance(loss_dict, dict), "loss_dict must be a dict containing loss/metrics"
        assert self.loss_config is not None, "do not have loss_confing"
        loss_sum = 0.
        num_weighted_losses = 0
        if validate:
            assets: List[Asset] = [a for a in batch.get('asset', [])]
            cls = assets[0].cls if assets else "all"
            for name in loss_dict:
                self._validation_loss[f"val/{cls}_{name}"].append(_get_item(loss_dict[name]))
                if name in self.loss_config and self.loss_config[name] > 0:
                    loss_sum += self.loss_config[name] * loss_dict[name]
                    num_weighted_losses += 1
            assert num_weighted_losses > 0, "no configured validation loss was produced by the model"
            self._validation_loss[f"val/{cls}_loss_sum"].append(_get_item(loss_sum))
            # TODO: log
            # self.log('val/loss_sum', loss_sum, prog_bar=True, logger=True, sync_dist=True, batch_size=len(assets))
        else:
            for name in loss_dict:
                if name in self.loss_config and self.loss_config[name] > 0:
                    loss_sum += self.loss_config[name] * loss_dict[name]
                    num_weighted_losses += 1
                self._train_loss[f"train/{name}"].append(_get_item(loss_dict[name]))
            assert num_weighted_losses > 0, "no configured training loss was produced by the model"
            loss_dict['loss_sum'] = loss_sum
            self._train_loss["train/loss_sum"].append(_get_item(loss_sum))
            # TODO: log
            # # add train prefix to loss_dict
            # prefixed_loss_dict = {f"train/{k}": v for k, v in loss_dict.items()}
            # d = dict(sorted(prefixed_loss_dict.items()))
        if not isinstance(loss_sum, jt.Var):
            return jt.array(loss_sum)
        return loss_sum
    
    def on_train_epoch_start(self):
        self._train_loss = defaultdict(list)
    
    def on_train_batch_start(self):
        pass
    
    def training_step(self, batch):
        return self.forward(batch, validate=False)
    
    def on_train_batch_end(self):
        pass

    def _summarize_losses(self, losses: Dict[str, List[float]]) -> Dict[str, float]:
        summary = {}
        for name, values in losses.items():
            avg = _mean(values)
            if avg is not None:
                summary[name] = avg
        return summary
    
    def on_train_epoch_end(self):
        if not self._train_loss:
            return {}
        summary = self._summarize_losses(self._train_loss)
        msg = []
        for name in sorted(summary.keys()):
            msg.append(f"{name}={summary[name]:.8f}")
        if msg:
            print("\033[95mTrain mean " + ", ".join(msg) + "\033[0m")
        return summary
    
    def on_validation_epoch_start(self):
        self._validation_loss = defaultdict(list)
    
    def on_validation_batch_start(self):
        pass
    
    def validation_step(self, batch):
        assert self.loss_config is not None, "do not have loss_confing"
        return self.forward(batch, validate=True)
    
    def on_validation_batch_end(self):
        pass
    
    def on_validation_epoch_end(self):
        summary = self._summarize_losses(self._validation_loss)
        loss_values = []
        for name, values in self._validation_loss.items():
            if name.endswith("loss_sum"):
                loss_values.extend(values)
        if not loss_values:
            return None, summary
        mean_loss = sum(loss_values) / len(loss_values)
        summary["val/loss_sum"] = mean_loss
        print(f"\033[96mValidation mean loss: {mean_loss:.8f}\033[0m")
        return mean_loss, summary
    
    def on_before_optimizer_step(self, optimizer):
        if self.grad_clip_norm is None:
            return
        try:
            total_norm_sq = 0.0
            grads = []
            for p in self.model.parameters():
                # Jittor 的 Var 没有 .grad 接口，必须通过 optimizer 取梯度
                grad = p.opt_grad(optimizer)
                if grad is None:
                    continue
                grads.append(grad)
                total_norm_sq += float(((grad ** 2.0).sum()).item())
            if not grads:
                return
            total_norm = math.sqrt(total_norm_sq)
            clip_coef = self.grad_clip_norm / (total_norm + 1e-6)
            if clip_coef < 1.0:
                for grad in grads:
                    # Jittor 原地更新梯度，optimizer.step 会读取更新后的值
                    grad.update(grad * clip_coef)
        except Exception as e:
            if not self._warned_grad_clip:
                print(f"\033[93mGrad clipping skipped: {e}\033[0m")
                self._warned_grad_clip = True

    def _compute_lr(self, epoch: int):
        if self.base_lr is None or self.lr_scheduler_config is None:
            return None
        cfg = self.lr_scheduler_config
        target = cfg.get('__target__', cfg.get('type', 'cosine'))
        min_lr = cfg.get('min_lr', 0.0)
        warmup_epochs = cfg.get('warmup_epochs', 0)
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return self.base_lr * float(epoch + 1) / float(warmup_epochs)
        if target == 'cosine':
            total = max(1, self.epochs - warmup_epochs)
            progress = min(1.0, max(0.0, float(epoch - warmup_epochs) / float(total)))
            return min_lr + 0.5 * (self.base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
        if target == 'multistep':
            lr = self.base_lr
            gamma = cfg.get('gamma', 0.5)
            milestones = cfg.get('milestones', [])
            for milestone in milestones:
                if epoch >= milestone:
                    lr *= gamma
            return max(min_lr, lr)
        raise ValueError(f"unsupported lr scheduler: {target}")

    def _set_optimizer_lr(self, lr):
        if lr is None or self.optimizer is None:
            return
        self.current_lr = lr
        updated = False
        for attr in ("lr", "learning_rate"):
            if hasattr(self.optimizer, attr):
                try:
                    setattr(self.optimizer, attr, lr)
                    updated = True
                except Exception:
                    pass
        if hasattr(self.optimizer, "param_groups"):
            try:
                for group in self.optimizer.param_groups:
                    group["lr"] = lr
                updated = True
            except Exception:
                pass
        if updated:
            print(f"\033[94mLearning rate: {lr:.8g}\033[0m")
    
    def on_predict_epoch_start(self):
        pass
    
    def on_predict_batch_start(self):
        pass
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        return self.model.predict_step(batch)
    
    def on_predict_batch_end(self):
        pass
    
    def on_predict_epoch_end(self):
        pass
    
    def train(self):
        assert self.optimizer is not None, "optimizer is None, cannot train"
        self.model.set_predict(False)
        self.training_started_at = time.time()
        last_row = None
        self.logger.log_text("Training started")
        for epoch in range(self.epochs):
            epoch_started_at = time.time()
            self.model.train()
            lr = self._compute_lr(epoch)
            if lr is not None:
                self._set_optimizer_lr(lr)
            elif self.current_lr is None:
                self.current_lr = self.base_lr
            self.on_train_epoch_start()
            train_dataloader = self.dataset_module.train_dataloader()
            assert train_dataloader is not None, "train_dataloader is None"
            train_batch_size = getattr(train_dataloader, "batch_size", None)
            train_total = len(train_dataloader)//train_dataloader.batch_size
            pbar = tqdm(train_dataloader, total=len(train_dataloader)//train_dataloader.batch_size) # type: ignore
            for batch_idx, batch in enumerate(pbar):
                self.on_train_batch_start()
                loss = self.training_step(batch)
                self.optimizer.zero_grad()
                self.optimizer.backward(loss)
                pbar.set_description(f"Epoch {epoch}, Loss: {_get_item(loss)}")
                if self.log_every_n_steps and self.log_every_n_steps > 0 and ((batch_idx + 1) % self.log_every_n_steps == 0):
                    self.logger.log_text(
                        f"epoch={epoch + 1}/{self.epochs} batch={batch_idx + 1}/{train_total} "
                        f"loss={_get_item(loss):.8f} lr={self.current_lr}"
                    )
                self.on_before_optimizer_step(self.optimizer)
                self.optimizer.step()
                self.on_train_batch_end()
            train_summary = self.on_train_epoch_end()
            
            self.model.eval()
            should_validate = self.validate_every and self.validate_every > 0 and ((epoch + 1) % self.validate_every == 0 or epoch == self.epochs - 1)
            val_summary = {}
            val_batches = 0
            if should_validate:
                validate_dataloader = self.dataset_module.validate_dataloader()
            else:
                validate_dataloader = None
            if validate_dataloader is not None:
                self.on_validation_epoch_start()
                if isinstance(validate_dataloader, dict):
                    for name, dataloader in validate_dataloader.items():
                        pbar = tqdm(dataloader, total=len(dataloader)//dataloader.batch_size)
                        val_batches += len(dataloader)//dataloader.batch_size
                        for batch in pbar:
                            self.on_validation_batch_start()
                            loss = self.validation_step(batch)
                            pbar.set_description(f"Epoch {epoch}, Validate {name}, Loss: {_get_item(loss)}")
                            self.on_validation_batch_end()
                else:
                    pbar = tqdm(validate_dataloader, total=len(validate_dataloader)//validate_dataloader.batch_size)
                    val_batches += len(validate_dataloader)//validate_dataloader.batch_size
                    for batch in pbar:
                        self.on_validation_batch_start()
                        loss = self.validation_step(batch)
                        pbar.set_description(f"Epoch {epoch}, Validate, Loss: {_get_item(loss)}")
                        self.on_validation_batch_end()
                mean_val_loss, val_summary = self.on_validation_epoch_end()
            else:
                mean_val_loss = None
            
            if self.save_last and self.save_last_every and self.save_last_every > 0 and ((epoch + 1) % self.save_last_every == 0 or epoch == self.epochs - 1):
                self.last_checkpoint_path = self._checkpoint_path(f'{self.ckpt_save_name}_last')
                self._save_checkpoint(self.last_checkpoint_path, 'last')
            if self.save_best and mean_val_loss is not None and mean_val_loss < self.best_validation_loss:
                self.best_validation_loss = mean_val_loss
                self.best_checkpoint_path = self._checkpoint_path(f'{self.ckpt_save_name}_best')
                self._save_checkpoint(self.best_checkpoint_path, f'best val {mean_val_loss:.8f}')
            if self.save_every and self.save_every > 0 and ((epoch + 1) % self.save_every == 0 or epoch == self.epochs - 1):
                checkpoint_path = self._checkpoint_path(f'{self.ckpt_save_name}_{epoch}')
                self._save_checkpoint(checkpoint_path, f'epoch {epoch}')
            epoch_seconds = time.time() - epoch_started_at
            elapsed_seconds = time.time() - self.training_started_at
            row = {
                "epoch": epoch + 1,
                "total_epochs": self.epochs,
                "lr": self.current_lr,
                "batch_size": train_batch_size,
                "train_batches": train_total,
                "val_batches": val_batches,
                "epoch_seconds": epoch_seconds,
                "elapsed_seconds": elapsed_seconds,
                "best_val_loss": self.best_validation_loss if self.best_validation_loss != math.inf else "",
                "best_checkpoint": self.best_checkpoint_path or "",
                "last_checkpoint": self.last_checkpoint_path or "",
            }
            row.update(train_summary)
            row.update(val_summary)
            last_row = dict(row)
            self.logger.write_metrics(row)
            self.logger.write_latest(row)
            self.logger.log_text(
                f"Epoch {epoch + 1}/{self.epochs}: "
                f"train_loss={train_summary.get('train/loss_sum', '')} "
                f"val_loss={mean_val_loss if mean_val_loss is not None else ''} "
                f"lr={self.current_lr} epoch_seconds={epoch_seconds:.2f} "
                f"best_val_loss={row['best_val_loss']}"
            )
        total_seconds = time.time() - self.training_started_at
        self.logger.log_text(f"Training finished: total_seconds={total_seconds:.2f}")
        if last_row is not None:
            last_row["status"] = "finished"
            last_row["total_seconds"] = total_seconds
            self.logger.write_latest(last_row)
    
    def predict(self):
        # only iterate once
        self.model.set_predict(True)
        self.model.eval()
        self.on_predict_epoch_start()
        predict_dataloader = self.dataset_module.predict_dataloader()
        assert predict_dataloader is not None, "predict_dataloader is None"
        if not isinstance(predict_dataloader, dict):
            predict_dataloader = {"predict": predict_dataloader}
        for dataloader_name, dataloader in predict_dataloader.items():
            pbar = tqdm(dataloader, total=len(dataloader)//dataloader.batch_size) # type: ignore
            for batch_idx, batch in enumerate(pbar):
                self.on_predict_batch_start()
                output = self.predict_step(batch, batch_idx)
                if self.writer is not None:
                    self.writer.write(batch, output, dataset_module=self.dataset_module)
                pbar.set_description(f"Predicting {dataloader_name}, Batch {batch_idx}")
