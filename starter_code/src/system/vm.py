from typing import List, Dict, Optional

import numpy as np
import os

from .spec import DummySystem, DummyWriter
from ..data.asset import Asset, Exporter

class VMWriter(DummyWriter):
    
    def __init__(self, save_dir: str="tmp_predict", save_name: str="denoised", output_format: str="npy"):
        super().__init__()
        self.save_dir = save_dir
        self.save_name = save_name
        self.output_format = output_format

    def _relative_output_dir(self, path: str, asset: Asset, dataset_module=None) -> str:
        dirname = os.path.dirname(path)
        input_root = None
        if dataset_module is not None and dataset_module.predict_dataset_config is not None:
            config = dataset_module.predict_dataset_config.get(asset.cls)
            if config is None and len(dataset_module.predict_dataset_config) == 1:
                config = next(iter(dataset_module.predict_dataset_config.values()))
            if config is not None:
                input_root = config.datapath.input_dataset_dir
        if input_root:
            rel_dir = os.path.relpath(dirname, input_root)
        else:
            rel_dir = dirname
        return os.path.join(self.save_dir, rel_dir)
    
    def write(self, batch, prediction: List[Dict], dataset_module=None):
        pc_noisy_batch = batch['pc_noisy']
        for i, asset in enumerate(batch['asset']):
            path = asset.path
            assert path is not None, "asset path is None"
            dirname = self._relative_output_dir(path, asset, dataset_module=dataset_module)
            os.makedirs(dirname, exist_ok=True)
            denoised = prediction[i]['pc_denoised']
            if isinstance(denoised, np.ndarray):
                denoised_np = denoised
            else:
                denoised_np = denoised.numpy()
            denoised_np = np.asarray(denoised_np)
            noisy_shape = tuple(pc_noisy_batch[i].shape)
            assert denoised_np.shape == noisy_shape, (
                f"denoised shape {denoised_np.shape} must equal noisy shape {noisy_shape}"
            )
            assert denoised_np.ndim == 2 and denoised_np.shape[1] == 3, (
                f"denoised point cloud must have shape (N, 3), found {denoised_np.shape}"
            )
            assert np.isfinite(denoised_np).all(), "denoised point cloud contains NaN or Inf"
            denoised_np = denoised_np.astype(np.float32)
            if self.output_format == 'npy':
                np.save(os.path.join(dirname, f"{self.save_name}.npy"), denoised_np)
            else:
                Exporter.export_obj(denoised_np, os.path.join(dirname, f"{self.save_name}.obj"))

class VMSystem(DummySystem):
    
    def __init__(
        self,
        dataset_module,
        model,
        loss_config=None,
        optimizer_config=None,
        trainer_config=None,
        writer: Optional[DummyWriter]=None,
        
        ckpt_save_dir: str="experiments",
        ckpt_save_name: str="checkpoint",
        config_snapshot=None,
    ):
        super().__init__(
            dataset_module=dataset_module,
            model=model,
            loss_config=loss_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
            writer=writer,
            ckpt_save_dir=ckpt_save_dir,
            ckpt_save_name=ckpt_save_name,
            config_snapshot=config_snapshot,
        )
    
    # override functions in dummy system if you want to implement training/validation/prediction logic
