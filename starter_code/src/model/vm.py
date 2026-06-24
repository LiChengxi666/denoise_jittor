from math import ceil
from typing import Dict, List

import jittor as jt
import numpy as np

from .feature import FeatureExtraction, Decoder
from .spec import ModelSpec

from ..data.asset import Asset

def get_random_indices(n, m):
    m = min(m, n)
    idx = np.random.permutation(n)[:m]
    return jt.array(idx).int32()

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
        self.dsm_sigma = cfg['dsm_sigma']
        self.cd_loss_enabled = cfg.get('cd_loss_enabled', False)
        self.cd_loss_sample_points = cfg.get('cd_loss_sample_points', self.num_train_points)
        self.cd_loss_independent_sampling = cfg.get('cd_loss_independent_sampling', False)
        self.point_plane_loss_enabled = cfg.get('point_plane_loss_enabled', False)
        self.offset_reg_enabled = cfg.get('offset_reg_enabled', False)
        self.repulsion_loss_enabled = cfg.get('repulsion_loss_enabled', False)
        self.repulsion_k = cfg.get('repulsion_k', 8)
        self.repulsion_h = cfg.get('repulsion_h', 0.03)
        self.scale_loss_enabled = cfg.get('scale_loss_enabled', False)
        self.scale_min_ratio = cfg.get('scale_min_ratio', 0.96)
        self.scale_max_ratio = cfg.get('scale_max_ratio', 1.04)
        
        # networks
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
    
    def _patch_chamfer_loss(self, pc_pred, pc_clean):
        """
        pc_pred:  (B, M, 3)
        pc_clean: (B, M, 3)
        """
        dist = ((pc_pred.unsqueeze(2) - pc_clean.unsqueeze(1)) ** 2.0).sum(dim=-1)
        pred_to_clean = jt.min(dist, dim=2)
        clean_to_pred = jt.min(dist, dim=1)
        return (pred_to_clean.mean() + clean_to_pred.mean()) / self.dsm_sigma
    
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
    
    def get_supervised_loss(self, pc_noisy, pc_mix, pc_clean, pc_clean_normal=None, pc_clean_cd=None):
        """
        pcl_noisy: (B, N, 3)
        pcl_clean: (B, N, 3)
        """
        B, N_noisy, d = pc_mix.shape
        
        pnt_idx = get_random_indices(N_noisy, self.num_train_points)
        
        # Feature extraction
        feat = self.encoder(pc_mix)  # (B, N, F)
        F_dim = feat.shape[2]
        
        # gather
        feat = feat[:, pnt_idx, :]
        pc_noisy = pc_noisy[:, pnt_idx, :]
        pc_mix = pc_mix[:, pnt_idx, :]
        pc_clean = pc_clean[:, pnt_idx, :]
        if pc_clean_normal is not None:
            pc_clean_normal = pc_clean_normal[:, pnt_idx, :]
        
        # target
        target_base = pc_noisy if self.target_mode == 'noisy' else pc_mix
        grad_dir_t_target = pc_clean - target_base
        
        # decoder
        pred_dir = self.decoder(
            c=feat.reshape(-1, F_dim)
        ).reshape(B, pnt_idx.shape[0], d) # type: ignore
        
        disp_loss = (((pred_dir - grad_dir_t_target) ** 2.0) / self.dsm_sigma).sum(dim=-1).mean()
        
        if (
            not self.cd_loss_enabled
            and not self.point_plane_loss_enabled
            and not self.offset_reg_enabled
            and not self.repulsion_loss_enabled
            and not self.scale_loss_enabled
        ):
            return {"loss": disp_loss}
        
        pc_denoised = pc_mix + pred_dir
        losses = {
            "disp_loss": disp_loss,
        }
        
        if self.cd_loss_enabled:
            cd_points_pred = min(self.cd_loss_sample_points, pnt_idx.shape[0])
            if self.cd_loss_independent_sampling:
                clean_for_cd = pc_clean_cd if pc_clean_cd is not None else pc_clean
                clean_points = clean_for_cd.shape[1]
                cd_points_clean = min(self.cd_loss_sample_points, clean_points)
                pred_idx = get_random_indices(pnt_idx.shape[0], cd_points_pred)
                clean_idx = get_random_indices(clean_points, cd_points_clean)
                losses["cd_loss"] = self._patch_chamfer_loss(
                    pc_pred=pc_denoised[:, pred_idx, :],
                    pc_clean=clean_for_cd[:, clean_idx, :],
                )
            else:
                losses["cd_loss"] = self._patch_chamfer_loss(
                    pc_pred=pc_denoised[:, :cd_points_pred, :],
                    pc_clean=pc_clean[:, :cd_points_pred, :],
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
            for it in range(num_steps):
                feat = self.encoder(pcl_next)  # (B, N, F)
                F_dim = feat.shape[2]
                
                pred_dir = self.decoder(
                    c=feat.reshape(-1, F_dim)
                ).reshape(B, N, d)
                
                velocity = self.predict_momentum * velocity + pred_dir
                step_scale = (step_size / num_steps) * self._decay_factor(it, num_steps)
                pcl_next = pcl_next + step_scale * velocity
        return pcl_next, None
    
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
    
    i = 0
    patch_step = int(ceil(N / (seed_k_alpha * patch_size)))
    assert patch_step > 0
    while i < num_patches:
        curr = patches[i:i+patch_step]
        try:
            out, _ = model.denoise_langevin_dynamics(curr)
        except Exception as e:
            print("Denoise error:", e)
            return None
        patches_denoised.append(out)
        i += patch_step
    
    patches_denoised = jt.concat(patches_denoised, dim=0)
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
