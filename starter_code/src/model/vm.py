from math import ceil
from typing import Dict, List

import jittor as jt
from jittor import nn
import numpy as np

from .feature import FeatureExtraction, MultiScaleFeatureExtraction, Decoder
from .spec import ModelSpec

from ..data.asset import Asset

def get_random_indices(n, m):
    m = min(m, n)
    idx = np.random.permutation(n)[:m]
    return jt.array(idx).int32()

def _cfg_enabled(cfg, enabled_key, legacy_weight_key=None, default=False):
    """Read boolean loss switch; fall back to legacy *_weight > 0 for compatibility."""
    if enabled_key in cfg:
        return bool(cfg[enabled_key])
    if legacy_weight_key is not None and legacy_weight_key in cfg:
        return float(cfg[legacy_weight_key]) > 0
    return default

class PointScalarHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(in_dim // 2, 32)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def execute(self, feat):
        """
        feat: (B, N, F)
        """
        B, N, F = feat.shape
        return self.net(feat.reshape(B * N, F)).reshape(B, N, 1)

class VelocityModule(ModelSpec):
    
    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        
        cfg = self.model_config
        # geometry
        self.frame_knn = cfg['frame_knn']
        self.num_train_points = cfg['num_train_points']
        self.predict_patch_size = cfg.get('predict_patch_size', 1000)
        self.predict_seed_k = cfg.get('predict_seed_k', 6)
        self.predict_seed_k_alpha = cfg.get('predict_seed_k_alpha', 1)
        self.predict_num_steps = cfg.get('predict_num_steps', 1)
        self.denoise_inner_steps = cfg.get('denoise_inner_steps', 4)
        self.predict_step_size = cfg.get('predict_step_size', 1.0)
        self.predict_momentum = cfg.get('predict_momentum', 0.0)
        self.predict_step_decay = cfg.get('predict_step_decay', 'none')
        assert self.predict_step_decay in ['none', 'linear'], f"unsupported predict_step_decay: {self.predict_step_decay}"
        self.alpha_blend = cfg.get('alpha_blend', 1.0)
        self.target_mode = cfg.get('target_mode', 'mix')
        assert self.target_mode in ['noisy', 'mix'], f"unsupported target_mode: {self.target_mode}"
        
        # score-matching
        self.dsm_sigma = float(cfg['dsm_sigma'])
        assert self.dsm_sigma > 0.0, f"dsm_sigma must be positive, got {self.dsm_sigma}"
        self.cd_loss_enabled = cfg.get('cd_loss_enabled', False)
        self.cd_loss_sample_points = cfg.get('cd_loss_sample_points', self.num_train_points)
        self.cd_loss_independent_sampling = cfg.get('cd_loss_independent_sampling', False)
        self.point_plane_loss_enabled = cfg.get('point_plane_loss_enabled', False)
        self.offset_reg_enabled = cfg.get('offset_reg_enabled', False)
        self.infocd_loss_enabled = cfg.get('infocd_loss_enabled', False)
        self.infocd_mode = cfg.get('infocd_mode', 'soft')
        assert self.infocd_mode in ['soft', 'contrastive'], f"unsupported infocd_mode: {self.infocd_mode}"
        self.infocd_tau = cfg.get('infocd_tau', 0.02)
        self.infocd_tau_prime = cfg.get('infocd_tau_prime', self.infocd_tau)
        self.infocd_lambda = cfg.get('infocd_lambda', 1.0)
        self.infocd_use_l1 = cfg.get('infocd_use_l1', True)
        self.infocd_sample_points = cfg.get('infocd_sample_points', self.cd_loss_sample_points)
        self.infocd_normalize_by_dsm_sigma = cfg.get(
            'infocd_normalize_by_dsm_sigma',
            self.infocd_mode == 'soft',
        )
        self.repulsion_loss_enabled = cfg.get('repulsion_loss_enabled', False)
        self.repulsion_k = cfg.get('repulsion_k', 8)
        self.repulsion_h = cfg.get('repulsion_h', 0.03)
        self.scale_loss_enabled = cfg.get('scale_loss_enabled', False)
        self.scale_min_ratio = cfg.get('scale_min_ratio', 0.96)
        self.scale_max_ratio = cfg.get('scale_max_ratio', 1.04)
        
        # CVM2 / gated velocity / confidence fusion
        self.encoder_type = cfg.get('encoder_type', 'standard')
        assert self.encoder_type in ['standard', 'multiscale'], f"unsupported encoder_type: {self.encoder_type}"
        self.cvm_stages = int(cfg.get('cvm_stages', 1))
        assert self.cvm_stages in [1, 2], f"unsupported cvm_stages: {self.cvm_stages}"
        self.stage_step = cfg.get('stage_step', 1.0 if self.cvm_stages == 1 else 0.5)
        self.distance_gate_enabled = cfg.get('distance_gate_enabled', False)
        self.gate_min = cfg.get('gate_min', 0.25)
        self.gate_max = cfg.get('gate_max', 1.25)
        self.stage0_loss_enabled = _cfg_enabled(cfg, 'stage0_loss_enabled', 'stage0_loss_weight')
        self.gate_reg_enabled = _cfg_enabled(cfg, 'gate_reg_enabled', 'gate_reg_weight')
        self.gate_target_enabled = _cfg_enabled(cfg, 'gate_target_enabled', 'gate_target_weight')
        self.confidence_enabled = cfg.get('confidence_enabled', False)
        self.confidence_min = cfg.get('confidence_min', 0.1)
        self.confidence_max = cfg.get('confidence_max', 2.0)
        self.confidence_target_enabled = _cfg_enabled(cfg, 'confidence_target_enabled', 'confidence_target_weight')
        self.confidence_reg_enabled = _cfg_enabled(cfg, 'confidence_reg_enabled', 'confidence_reg_weight')
        if self.confidence_target_enabled:
            self.confidence_reg_enabled = False
        self.confidence_target_alpha = cfg.get('confidence_target_alpha', 25.0)
        
        # networks
        if self.encoder_type == 'multiscale':
            self.encoder = MultiScaleFeatureExtraction(
                knns=cfg.get('multiscale_knns', [16, 24, 40]),
                input_dim=3,
                embedding_dim=cfg['feat_embedding_dim'],
                branch_dim=cfg.get('multiscale_branch_dim', None),
                global_feat=cfg.get('global_feat', False),
                channel_attention=cfg.get('channel_attention', True),
            )
        else:
            self.encoder = FeatureExtraction(
                k=self.frame_knn,
                input_dim=3,
                embedding_dim=cfg['feat_embedding_dim'],
                global_feat=cfg.get('global_feat', False),
            )
        
        self.decoder = Decoder(
            z_dim=self.encoder.output_dim,
            dim=3,
            out_dim=3,
            hidden_size=cfg['decoder_hidden_dim'],
        )
        if self.cvm_stages == 2:
            self.decoder1 = Decoder(
                z_dim=self.encoder.output_dim,
                dim=3,
                out_dim=3,
                hidden_size=cfg['decoder_hidden_dim'],
            )
        if self.distance_gate_enabled:
            self.gate_head0 = PointScalarHead(self.encoder.output_dim)
            if self.cvm_stages == 2:
                self.gate_head1 = PointScalarHead(self.encoder.output_dim)
        if self.confidence_enabled:
            self.confidence_head = PointScalarHead(self.encoder.output_dim)
    
    def _logsumexp(self, x, dim, keepdims=False):
        xmax = jt.max(x, dim=dim, keepdims=True)
        out = xmax + jt.log(jt.sum(jt.exp(x - xmax), dim=dim, keepdims=True) + 1e-8)
        if not keepdims:
            out = out.squeeze(dim)
        return out

    def _patch_chamfer_components(self, pc_pred, pc_clean):
        """
        pc_pred:  (B, M, 3)
        pc_clean: (B, M, 3)
        """
        dist = ((pc_pred.unsqueeze(2) - pc_clean.unsqueeze(1)) ** 2.0).sum(dim=-1)
        pred_to_clean = jt.min(dist, dim=2)
        clean_to_pred = jt.min(dist, dim=1)
        pred_to_clean = pred_to_clean.mean() / self.dsm_sigma
        clean_to_pred = clean_to_pred.mean() / self.dsm_sigma
        return pred_to_clean + clean_to_pred, pred_to_clean, clean_to_pred

    def _patch_chamfer_loss(self, pc_pred, pc_clean):
        total, _, _ = self._patch_chamfer_components(pc_pred, pc_clean)
        return total

    def _infocd_contrastive_one_way(self, metric):
        """
        metric: (B, M, N) — each row contrasts N candidates.
        logsumexp(-d/tau) + lambda * d_pos/tau_prime
        """
        tau = max(float(self.infocd_tau), 1e-6)
        tau_prime = max(float(self.infocd_tau_prime), 1e-6)
        lam = float(self.infocd_lambda)
        logits = -metric / tau
        pos_dist = jt.min(metric, dim=2)
        log_sum = self._logsumexp(logits, dim=2)
        return (log_sum + lam * pos_dist / tau_prime).mean()

    def _infocd_loss_soft(self, pc_pred, pc_clean):
        """
        Soft-nearest bidirectional alignment. Exponential weights spread
        gradients across near neighbors instead of only the hard nearest point.
        """
        dist = ((pc_pred.unsqueeze(2) - pc_clean.unsqueeze(1)) ** 2.0).sum(dim=-1)
        metric = jt.sqrt(dist + 1e-12) if self.infocd_use_l1 else dist
        tau = max(float(self.infocd_tau), 1e-6)

        weight_pred = jt.exp(-metric / tau)
        pred_to_clean = (weight_pred * metric).sum(dim=2) / (weight_pred.sum(dim=2) + 1e-8)

        weight_clean = jt.exp(-metric / tau)
        clean_to_pred = (weight_clean * metric).sum(dim=1) / (weight_clean.sum(dim=1) + 1e-8)
        return (pred_to_clean.mean() + clean_to_pred.mean()) / self.dsm_sigma

    def _infocd_loss_contrastive(self, pc_pred, pc_clean):
        """
        Contrastive Chamfer (InfoCD): bidirectional hard-positive InfoNCE on
        point-to-set distances. The terms are already normalized by tau and
        tau_prime, so DSM normalization is optional and disabled by default.
        """
        dist = ((pc_pred.unsqueeze(2) - pc_clean.unsqueeze(1)) ** 2.0).sum(dim=-1)
        metric = jt.sqrt(dist + 1e-12) if self.infocd_use_l1 else dist
        pred_to_clean = self._infocd_contrastive_one_way(metric)
        clean_to_pred = self._infocd_contrastive_one_way(metric.transpose(0, 2, 1))
        loss = pred_to_clean + clean_to_pred
        if self.infocd_normalize_by_dsm_sigma:
            loss = loss / self.dsm_sigma
        return loss

    def _infocd_loss(self, pc_pred, pc_clean):
        if self.infocd_mode == 'contrastive':
            return self._infocd_loss_contrastive(pc_pred, pc_clean)
        return self._infocd_loss_soft(pc_pred, pc_clean)

    def _gate_target_loss(self, gate, residual_dist, reference_dist):
        """
        Supervise gate toward relative remaining distance (DM-style scalar).
        """
        gate_target = residual_dist / (reference_dist + 1e-8)
        gate_target = jt.clamp(gate_target, self.gate_min, self.gate_max).stop_grad()
        return ((gate - gate_target) ** 2.0).mean()

    def _confidence_target_loss(self, confidence, pc_pred, pc_clean):
        """
        Supervise confidence toward exp(-alpha * point error).
        """
        point_error = jt.sqrt(((pc_pred - pc_clean) ** 2.0).sum(dim=-1) + 1e-12)
        conf_target_raw = jt.exp(-self.confidence_target_alpha * point_error)
        conf_target = self.confidence_min + (self.confidence_max - self.confidence_min) * conf_target_raw
        conf_target = jt.clamp(conf_target, self.confidence_min, self.confidence_max).stop_grad()
        return ((confidence.squeeze(-1) - conf_target) ** 2.0).mean()
    
    def _point_plane_loss(self, pc_pred, pc_clean, pc_clean_normal):
        normal = pc_clean_normal / (jt.sqrt((pc_clean_normal ** 2.0).sum(dim=-1, keepdims=True)) + 1e-8)
        signed_dist = ((pc_pred - pc_clean) * normal).sum(dim=-1)
        return ((signed_dist ** 2.0) / self.dsm_sigma).mean()

    def _repulsion_loss(self, pc_pred):
        """
        pc_pred: (B, M, 3)
        """
        _, M, _ = pc_pred.shape
        if M <= 1 or self.repulsion_k <= 0:
            return jt.array(0.0)
        k = min(self.repulsion_k + 1, M)
        dist = ((pc_pred.unsqueeze(2) - pc_pred.unsqueeze(1)) ** 2.0).sum(dim=-1)
        dist_k, _ = jt.topk(dist, k=k, dim=-1, largest=False)
        dist_k = dist_k[:, :, 1:]
        dist_k = jt.sqrt(dist_k + 1e-12)
        gap = self.repulsion_h - dist_k
        penalty = jt.maximum(gap, gap * 0.0)
        return ((penalty ** 2.0) / self.dsm_sigma).mean()

    def _scale_consistency_loss(self, pc_pred, pc_noisy):
        pred_center = pc_pred.mean(dim=1, keepdims=True)
        noisy_center = pc_noisy.mean(dim=1, keepdims=True)
        pred_radius = jt.sqrt(((pc_pred - pred_center) ** 2.0).sum(dim=-1).mean(dim=1) + 1e-12)
        noisy_radius = jt.sqrt(((pc_noisy - noisy_center) ** 2.0).sum(dim=-1).mean(dim=1) + 1e-12)
        min_radius = noisy_radius * self.scale_min_ratio
        max_radius = noisy_radius * self.scale_max_ratio
        shrink_gap = min_radius - pred_radius
        expand_gap = pred_radius - max_radius
        shrink_loss = jt.maximum(shrink_gap, shrink_gap * 0.0) ** 2.0
        expand_loss = jt.maximum(expand_gap, expand_gap * 0.0) ** 2.0
        return ((shrink_loss + expand_loss) / self.dsm_sigma).mean()

    def _decay_factor(self, step: int, num_steps: int):
        if self.predict_step_decay == 'none' or num_steps <= 1:
            return 1.0
        if self.predict_step_decay == 'linear':
            tail = 1.0 / float(num_steps)
            return max(0.25, 1.0 - (float(step) / float(num_steps - 1)) * (1.0 - tail))
        raise ValueError(f"unsupported predict_step_decay: {self.predict_step_decay}")
    
    def _map_gate(self, gate_raw):
        return self.gate_min + (self.gate_max - self.gate_min) * jt.sigmoid(gate_raw)
    
    def _map_confidence(self, confidence_raw):
        return self.confidence_min + (self.confidence_max - self.confidence_min) * jt.sigmoid(confidence_raw)
    
    def _unit_scalar(self, feat):
        return feat[:, :, :1] * 0.0 + 1.0
    
    def _predict_direction(self, pc, return_aux=False):
        """
        pc: (B, N, 3)
        """
        B, N, d = pc.shape
        aux = {}
        
        feat0 = self.encoder(pc)
        F_dim = feat0.shape[2]
        v0 = self.decoder(c=feat0.reshape(-1, F_dim)).reshape(B, N, d)
        if self.distance_gate_enabled:
            gate0 = self._map_gate(self.gate_head0(feat0))
        else:
            gate0 = self._unit_scalar(feat0)
        
        if self.cvm_stages == 1:
            pred_dir = self.stage_step * gate0 * v0
            feat_final = feat0
            aux["gate0"] = gate0
        else:
            mid = pc + self.stage_step * gate0 * v0
            feat1 = self.encoder(mid)
            F_dim1 = feat1.shape[2]
            v1 = self.decoder1(c=feat1.reshape(-1, F_dim1)).reshape(B, N, d)
            if self.distance_gate_enabled:
                gate1 = self._map_gate(self.gate_head1(feat1))
            else:
                gate1 = self._unit_scalar(feat1)
            pred_dir = self.stage_step * gate0 * v0 + self.stage_step * gate1 * v1
            feat_final = feat1
            aux["mid"] = mid
            aux["gate0"] = gate0
            aux["gate1"] = gate1
        
        if self.confidence_enabled:
            aux["confidence"] = self._map_confidence(self.confidence_head(feat_final))
        
        if return_aux:
            return pred_dir, aux
        return pred_dir
    
    def get_supervised_loss(self, pc_noisy, pc_mix, pc_clean, pc_clean_normal=None, pc_clean_cd=None):
        """
        pcl_noisy: (B, N, 3)
        pcl_clean: (B, N, 3)
        """
        B, N_noisy, d = pc_mix.shape
        
        pnt_idx = get_random_indices(N_noisy, self.num_train_points)
        
        pred_dir_full, aux = self._predict_direction(pc_mix, return_aux=True)
        
        # gather
        pc_noisy = pc_noisy[:, pnt_idx, :]
        pc_mix = pc_mix[:, pnt_idx, :]
        pc_clean = pc_clean[:, pnt_idx, :]
        pred_dir = pred_dir_full[:, pnt_idx, :]
        if pc_clean_normal is not None:
            pc_clean_normal = pc_clean_normal[:, pnt_idx, :]
        
        # target
        target_base = pc_noisy if self.target_mode == 'noisy' else pc_mix
        grad_dir_t_target = pc_clean - target_base
        
        disp_loss = (((pred_dir - grad_dir_t_target) ** 2.0) / self.dsm_sigma).sum(dim=-1).mean()
        
        if (
            not self.cd_loss_enabled
            and not self.point_plane_loss_enabled
            and not self.offset_reg_enabled
            and not self.infocd_loss_enabled
            and not self.repulsion_loss_enabled
            and not self.scale_loss_enabled
            and not (self.cvm_stages == 2 and self.stage0_loss_enabled)
            and not (self.distance_gate_enabled and self.gate_reg_enabled)
            and not (self.distance_gate_enabled and self.gate_target_enabled)
            and not (self.confidence_enabled and self.confidence_reg_enabled)
            and not (self.confidence_enabled and self.confidence_target_enabled)
        ):
            return {"loss": disp_loss}
        
        pc_denoised = pc_mix + pred_dir
        losses = {
            "disp_loss": disp_loss,
        }
        
        if self.cvm_stages == 2 and self.stage0_loss_enabled and "mid" in aux:
            mid = aux["mid"][:, pnt_idx, :]
            losses["stage0_loss"] = (((mid - pc_clean) ** 2.0) / self.dsm_sigma).sum(dim=-1).mean()
        if self.distance_gate_enabled and self.gate_reg_enabled:
            gate_reg = ((aux["gate0"][:, pnt_idx, :] - 1.0) ** 2.0).mean()
            if "gate1" in aux:
                gate_reg = gate_reg + ((aux["gate1"][:, pnt_idx, :] - 1.0) ** 2.0).mean()
                gate_reg = gate_reg * 0.5
            losses["gate_reg"] = gate_reg
        if self.distance_gate_enabled and self.gate_target_enabled:
            ref_dist = jt.sqrt(((pc_noisy - pc_clean) ** 2.0).sum(dim=-1, keepdims=True) + 1e-12)
            mix_residual = jt.sqrt(((pc_mix - pc_clean) ** 2.0).sum(dim=-1, keepdims=True) + 1e-12)
            gate_target_loss = self._gate_target_loss(
                aux["gate0"][:, pnt_idx, :],
                mix_residual,
                ref_dist,
            )
            if "gate1" in aux and "mid" in aux:
                mid = aux["mid"][:, pnt_idx, :]
                mid_residual = jt.sqrt(((mid - pc_clean) ** 2.0).sum(dim=-1, keepdims=True) + 1e-12)
                gate_target_loss = gate_target_loss + self._gate_target_loss(
                    aux["gate1"][:, pnt_idx, :],
                    mid_residual,
                    ref_dist,
                )
                gate_target_loss = gate_target_loss * 0.5
            losses["gate_target_loss"] = gate_target_loss
        if self.confidence_enabled and self.confidence_reg_enabled and "confidence" in aux:
            losses["confidence_reg"] = (aux["confidence"][:, pnt_idx, :].mean() - 1.0) ** 2.0
        if self.confidence_enabled and self.confidence_target_enabled and "confidence" in aux:
            losses["confidence_target_loss"] = self._confidence_target_loss(
                aux["confidence"][:, pnt_idx, :],
                pc_denoised,
                pc_clean,
            )

        if self.cd_loss_enabled:
            cd_points_pred = min(self.cd_loss_sample_points, pnt_idx.shape[0])
            if self.cd_loss_independent_sampling:
                clean_for_cd = pc_clean_cd if pc_clean_cd is not None else pc_clean
                clean_points = clean_for_cd.shape[1]
                cd_points_clean = min(self.cd_loss_sample_points, clean_points)
                pred_idx = get_random_indices(pnt_idx.shape[0], cd_points_pred)
                clean_idx = get_random_indices(clean_points, cd_points_clean)
                pc_cd_pred = pc_denoised[:, pred_idx, :]
                pc_cd_clean = clean_for_cd[:, clean_idx, :]
                cd_loss, cd_pred_to_clean, cd_clean_to_pred = self._patch_chamfer_components(
                    pc_pred=pc_cd_pred,
                    pc_clean=pc_cd_clean,
                )
            else:
                pc_cd_pred = pc_denoised[:, :cd_points_pred, :]
                pc_cd_clean = pc_clean[:, :cd_points_pred, :]
                cd_loss, cd_pred_to_clean, cd_clean_to_pred = self._patch_chamfer_components(
                    pc_pred=pc_cd_pred,
                    pc_clean=pc_cd_clean,
                )
            losses["cd_loss"] = cd_loss
            losses["cd_pred_to_clean"] = cd_pred_to_clean
            losses["cd_clean_to_pred"] = cd_clean_to_pred
        if self.infocd_loss_enabled:
            info_points = min(self.infocd_sample_points, pnt_idx.shape[0])
            losses["infocd_loss"] = self._infocd_loss(
                pc_pred=pc_denoised[:, :info_points, :],
                pc_clean=pc_clean[:, :info_points, :],
            )
        if self.point_plane_loss_enabled and pc_clean_normal is not None:
            losses["point_plane_loss"] = self._point_plane_loss(
                pc_pred=pc_denoised,
                pc_clean=pc_clean,
                pc_clean_normal=pc_clean_normal,
            )
        if self.offset_reg_enabled:
            losses["offset_reg"] = ((pred_dir ** 2.0).sum(dim=-1) / self.dsm_sigma).mean()
        if self.repulsion_loss_enabled:
            losses["repulsion_loss"] = self._repulsion_loss(pc_denoised)
        if self.scale_loss_enabled:
            losses["scale_loss"] = self._scale_consistency_loss(pc_denoised, pc_noisy)
        return losses

    def denoise_langevin_dynamics(self, pcl_noisy, num_steps: int=None, step_size: float=None):
        """
        pcl_noisy: (B, N, 3)
        """
        if num_steps is None:
            num_steps = self.denoise_inner_steps
        if step_size is None:
            step_size = self.predict_step_size
        B, N, d = pcl_noisy.shape
        with jt.no_grad():
            pcl_next = pcl_noisy.clone()
            velocity = pcl_next * 0.0
            confidence = None
            for it in range(num_steps):
                pred_dir, aux = self._predict_direction(pcl_next, return_aux=True)
                confidence = aux.get("confidence", None)
                
                velocity = self.predict_momentum * velocity + pred_dir
                step_scale = (step_size / num_steps) * self._decay_factor(it, num_steps)
                pcl_next = pcl_next + step_scale * velocity
        return pcl_next, confidence
    
    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch['pc_noisy'].shape[-2]
        pc_noisy = batch['pc_noisy'].reshape(-1, patch_size, 3)
        pc_mix = batch['pc_mix'].reshape(-1, patch_size, 3)
        pc_clean = batch['pc_clean'].reshape(-1, patch_size, 3)
        pc_clean_cd = batch.get('pc_clean_cd', None)
        if pc_clean_cd is not None:
            clean_cd_size = batch['pc_clean_cd'].shape[-2]
            pc_clean_cd = pc_clean_cd.reshape(-1, clean_cd_size, 3)
        pc_clean_normal = batch.get('pc_clean_normal', None)
        if pc_clean_normal is not None:
            pc_clean_normal = pc_clean_normal.reshape(-1, patch_size, 3)
        loss = self.get_supervised_loss(
            pc_noisy=pc_noisy,
            pc_mix=pc_mix,
            pc_clean=pc_clean,
            pc_clean_normal=pc_clean_normal,
            pc_clean_cd=pc_clean_cd,
        )
        return loss
    
    def execute(self, **kwargs) -> Dict: # type: ignore
        return self.training_step(**kwargs)
    
    @jt.no_grad()
    def predict_step(self, batch: Dict) -> List[Dict]:
        pc_noisy_batch = batch['pc_noisy']
        assert pc_noisy_batch.ndim == 3
        
        res = []
        for i, pc_noisy in enumerate(pc_noisy_batch):
            pc_next = pc_noisy
            for it in range(self.predict_num_steps):
                pc_next = patch_based_denoise(
                    model=self,
                    pcl_noisy=pc_next,
                    patch_size=self.predict_patch_size,
                    seed_k=self.predict_seed_k,
                    seed_k_alpha=self.predict_seed_k_alpha,
                )
                assert pc_next is not None, "patch_based_denoise returned None"
            if self.alpha_blend != 1.0:
                pc_next = pc_noisy + self.alpha_blend * (pc_next - pc_noisy)
            pc_denoised = pc_next.detach().numpy()
            res.append({"pc_denoised": pc_denoised})
        return res
    
    def process_fn(self, batch: List[Asset]) -> List[Dict]:
        res = []
        for b in batch:
            if not self.is_predict():
                assert b.meta is not None
                res.append({
                    "pc_noisy": b.meta['pc_noisy'], # (num_patches, patch_size, 3)
                    "pc_clean": b.meta['pc_clean'],
                    "pc_mix": b.meta['pc_mix'],
                    **({"pc_clean_cd": b.meta["pc_clean_cd"]} if "pc_clean_cd" in b.meta else {}),
                    **({"pc_clean_normal": b.meta["pc_clean_normal"]} if "pc_clean_normal" in b.meta else {}),
                })
            else:
                d = {
                    "pc_noisy": b.sampled_vertices_noisy, # (N, 3)
                }
                if b.sampled_vertices is not None:
                    d["pc_clean"] = b.sampled_vertices
                res.append(d)
        return res

def farthest_point_sampling(pcls, num_pnts):
    """
    pcls: (B, N, 3)
    return:
        sampled: (B, num_pnts, 3)
        indices: (B, num_pnts)
    """
    B, N, _ = pcls.shape
    sampled = []
    indices = []
    for b in range(B):
        pts = pcls[b]  # (N, 3)
        selected = []
        dist = jt.ones((N,)) * 1e10
        farthest = 0
        for i in range(num_pnts):
            selected.append(farthest)
            centroid = pts[farthest]  # (3,)
            d = ((pts - centroid) ** 2).sum(dim=1)
            dist = jt.minimum(dist, d)
            farthest, _ = jt.argmax(dist, dim=-1)
            farthest = farthest.item()
        idx = jt.array(selected).int32()
        sampled.append(pts[idx][None, ...])
        indices.append(idx[None, ...])
    sampled = jt.concat(sampled, dim=0)
    indices = jt.concat(indices, dim=0)
    return sampled, indices

def knn_points(x, y, k):
    """
    x: (B, P, 3)
    y: (B, N, 3)
    return:
        dist: (B, P, k)
        idx:  (B, P, k)
        nn:   (B, P, k, 3)
    """
    dist = ((x.unsqueeze(2) - y.unsqueeze(1)) ** 2).sum(-1)
    dist_k, idx = jt.topk(dist, k=k, dim=-1, largest=False)
    B = x.shape[0]
    nn = []
    for b in range(B):
        nn.append(y[b][idx[b]])
    nn = jt.stack(nn, dim=0)
    return dist_k, idx, nn

def patch_based_denoise(model: VelocityModule, pcl_noisy, patch_size=1000, seed_k=6, seed_k_alpha=1) -> jt.Var:
    """
    pcl_noisy: (N, 3)
    """
    assert len(pcl_noisy.shape) == 2
    
    N, d = pcl_noisy.shape
    patch_size = min(patch_size, N)
    num_patches = max(1, int(ceil(seed_k * N / patch_size)))
    pcl_noisy = pcl_noisy.unsqueeze(0)  # (1, N, 3)
    
    seed_pnts, seed_idx = farthest_point_sampling(pcl_noisy, num_patches)
    patch_dists, point_idxs, patches = knn_points(seed_pnts, pcl_noisy, patch_size)
    
    patches = patches[0]              # (P, M, 3)
    patch_dists = patch_dists[0]      # (P, M)
    point_idxs = point_idxs[0]        # (P, M)
    
    seed_expand = seed_pnts[0].unsqueeze(1).broadcast(patches.shape)
    patches = patches - seed_expand
    
    patch_dists = patch_dists / (patch_dists[:, -1:].broadcast(patch_dists.shape) + 1e-8)
    patch_weights = jt.exp(-patch_dists).unsqueeze(-1)
    patches_denoised = []
    patch_confidences = []
    
    i = 0
    patch_step = int(ceil(N / (seed_k_alpha * patch_size)))
    assert patch_step > 0
    while i < num_patches:
        curr = patches[i:i+patch_step]
        try:
            out, confidence = model.denoise_langevin_dynamics(curr)
        except Exception as e:
            print("Denoise error:", e)
            return None
        patches_denoised.append(out)
        if model.confidence_enabled:
            if confidence is None:
                confidence = jt.ones_like(out[:, :, :1])
            patch_confidences.append(confidence)
        i += patch_step
    
    patches_denoised = jt.concat(patches_denoised, dim=0)
    if model.confidence_enabled and patch_confidences:
        patch_confidence = jt.concat(patch_confidences, dim=0)
        patch_confidence = jt.minimum(
            jt.maximum(patch_confidence, patch_confidence * 0.0 + 0.05),
            patch_confidence * 0.0 + 2.0,
        )
        patch_weights = patch_weights * patch_confidence
    patches_denoised = patches_denoised + seed_expand
    pcl_sum = jt.zeros((N, d))
    weight_sum = jt.zeros((N, 1))
    for patch_id in range(num_patches):
        idx = point_idxs[patch_id]
        weight = patch_weights[patch_id]
        weighted_points = patches_denoised[patch_id] * weight
        pcl_sum = pcl_sum.scatter_(0, idx.unsqueeze(1).broadcast(weighted_points.shape), weighted_points, reduce='add')
        weight_sum = weight_sum.scatter_(0, idx.unsqueeze(1), weight, reduce='add')
    pcl_out = pcl_sum / (weight_sum.broadcast((N, d)) + 1e-8)
    return pcl_out
