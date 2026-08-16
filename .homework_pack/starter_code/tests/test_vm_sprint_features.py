from math import ceil
import copy
import os
import sys
import unittest

import jittor as jt
from jittor import nn
import numpy as np
import yaml


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.model.vm import (
    VelocityModule,
    farthest_point_sampling,
    knn_points,
    patch_based_denoise,
)


def legacy_patch_based_denoise(model, pcl_noisy, patch_size, seed_k, seed_k_alpha):
    n_points, dim = pcl_noisy.shape
    patch_size = min(patch_size, n_points)
    num_patches = max(1, int(ceil(seed_k * n_points / patch_size)))
    pcl_noisy = pcl_noisy.unsqueeze(0)

    seed_pnts, _ = farthest_point_sampling(pcl_noisy, num_patches)
    patch_dists, point_idxs, patches = knn_points(seed_pnts, pcl_noisy, patch_size)
    patches = patches[0]
    patch_dists = patch_dists[0]
    point_idxs = point_idxs[0]

    seed_expand = seed_pnts[0].unsqueeze(1).broadcast(patches.shape)
    patches = patches - seed_expand
    patch_dists = patch_dists / (
        patch_dists[:, -1:].broadcast(patch_dists.shape) + 1e-8
    )
    patch_weights = jt.exp(-patch_dists).unsqueeze(-1)

    patch_step = int(ceil(n_points / (seed_k_alpha * patch_size)))
    denoised = []
    for start in range(0, num_patches, patch_step):
        output, _ = model.denoise_langevin_dynamics(
            patches[start:start + patch_step]
        )
        denoised.append(output)
    denoised = jt.concat(denoised, dim=0) + seed_expand

    pcl_sum = jt.zeros((n_points, dim))
    weight_sum = jt.zeros((n_points, 1))
    for patch_id in range(num_patches):
        idx = point_idxs[patch_id]
        weight = patch_weights[patch_id]
        weighted_points = denoised[patch_id] * weight
        pcl_sum = pcl_sum.scatter_(
            0,
            idx.unsqueeze(1).broadcast(weighted_points.shape),
            weighted_points,
            reduce="add",
        )
        weight_sum = weight_sum.scatter_(
            0,
            idx.unsqueeze(1),
            weight,
            reduce="add",
        )
    return pcl_sum / (weight_sum.broadcast((n_points, dim)) + 1e-8)


class DenoiseStub:
    predict_patch_weight_gamma = 1.0

    def denoise_langevin_dynamics(self, patches):
        return patches + 0.05 * jt.tanh(patches), None


class EncoderStub(nn.Module):
    output_dim = 3

    def execute(self, points):
        return points


class DecoderStub(nn.Module):
    def execute(self, c):
        return c * 0.0


class VMSprintFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        jt.flags.use_cuda = 0
        jt.set_global_seed(123)
        np.random.seed(123)
        with open(os.path.join(ROOT, "configs/model/vm_strong.yaml"), "r") as f:
            cls.config = yaml.safe_load(f)
        with open(os.path.join(ROOT, "configs/transform/vm_strong.yaml"), "r") as f:
            cls.transform = yaml.safe_load(f)

    def test_new_options_do_not_change_parameter_keys(self):
        old_config = copy.deepcopy(self.config)
        for key in (
            "predict_patch_weight_gamma",
            "info_cd_enabled",
            "info_cd_sample_points",
            "info_cd_tau",
            "info_cd_reg_weight",
            "info_cd_normalize",
        ):
            old_config.pop(key, None)
        old_model = VelocityModule(old_config, self.transform)
        new_model = VelocityModule(self.config, self.transform)
        old_keys = [name for name, _ in old_model.named_parameters()]
        new_keys = [name for name, _ in new_model.named_parameters()]
        self.assertEqual(old_keys, new_keys)

    def test_gamma_one_matches_legacy_patch_fusion(self):
        points = jt.array(
            np.random.RandomState(7).normal(size=(32, 3)).astype(np.float32)
        )
        legacy = legacy_patch_based_denoise(
            DenoiseStub(), points, patch_size=12, seed_k=3, seed_k_alpha=1
        ).numpy()
        current = patch_based_denoise(
            DenoiseStub(), points, patch_size=12, seed_k=3, seed_k_alpha=1
        ).numpy()
        np.testing.assert_array_equal(current, legacy)

    def test_info_cd_is_finite_and_scale_invariant(self):
        model = VelocityModule(self.config, self.transform)
        pred_np = np.random.RandomState(11).normal(size=(2, 24, 3)).astype(np.float32)
        clean_np = pred_np + np.random.RandomState(12).normal(
            scale=0.03, size=(2, 24, 3)
        ).astype(np.float32)
        pred = jt.array(pred_np)
        clean = jt.array(clean_np)

        loss = model._info_cd_loss(pred, clean)
        grad = jt.grad(loss, pred).numpy()
        self.assertTrue(np.isfinite(float(loss.item())))
        self.assertTrue(np.isfinite(grad).all())

        scaled_losses = [
            float(model._info_cd_loss(pred * scale, clean * scale).item())
            for scale in (0.8, 1.0, 1.2)
        ]
        self.assertLess(max(scaled_losses) - min(scaled_losses), 1e-5)

    def test_info_cd_key_respects_enabled_flag(self):
        info_config = copy.deepcopy(self.config)
        info_config.update({
            "num_train_points": 16,
            "cd_loss_enabled": False,
            "info_cd_enabled": True,
            "point_plane_loss_enabled": False,
            "offset_reg_enabled": False,
        })
        pred = np.random.RandomState(21).normal(size=(2, 16, 3)).astype(np.float32)
        pc_noisy = jt.array(pred)
        pc_mix = pc_noisy + 0.01
        pc_clean = pc_noisy + 0.02

        enabled_model = VelocityModule(info_config, self.transform)
        enabled_model.encoder = EncoderStub()
        enabled_model.decoder = DecoderStub()
        enabled = enabled_model.get_supervised_loss(pc_noisy, pc_mix, pc_clean)
        self.assertIn("info_cd_loss", enabled)

        info_config["info_cd_enabled"] = False
        disabled_model = VelocityModule(info_config, self.transform)
        disabled_model.encoder = EncoderStub()
        disabled_model.decoder = DecoderStub()
        disabled = disabled_model.get_supervised_loss(pc_noisy, pc_mix, pc_clean)
        self.assertNotIn("info_cd_loss", disabled)


if __name__ == "__main__":
    unittest.main()
