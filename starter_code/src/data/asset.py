from dataclasses import dataclass

from numpy import ndarray
from typing import Dict, Optional

import numpy as np
import os

@dataclass
class Asset():
    path: Optional[str]=None # where is the asset loaded from
    
    cls: Optional[str]=None # cls
    
    vertices: Optional[ndarray]=None # shape (N, 3)

    vertex_normals: Optional[ndarray]=None # shape (N, 3)
    
    faces: Optional[ndarray]=None # shape (F, 3)

    face_normals: Optional[ndarray]=None # shape (F, 3)
    
    sampled_vertices: Optional[ndarray]=None

    sampled_normals: Optional[ndarray]=None
    
    sampled_vertices_noisy: Optional[ndarray]=None
    
    meta: Optional[Dict]=None
    
    def transform(self, trans: ndarray):
        """trans: 4x4 affine matrix"""
        def _apply(v: ndarray, trans: ndarray) -> ndarray:
            return np.matmul(v, trans[:3, :3].transpose()) + trans[:3, 3]

        def _apply_normals(n: ndarray, trans: ndarray) -> ndarray:
            normal_mat = np.linalg.pinv(trans[:3, :3]).transpose()
            out = np.matmul(n, normal_mat.transpose())
            norm = np.linalg.norm(out, axis=1, keepdims=True)
            return out / (norm + 1e-12)
        
        if self.vertices is not None:
            self.vertices = _apply(self.vertices, trans)
        if self.sampled_vertices is not None:
            self.sampled_vertices = _apply(self.sampled_vertices, trans)
        if self.sampled_vertices_noisy is not None:
            self.sampled_vertices_noisy = _apply(self.sampled_vertices_noisy, trans)
        if self.vertex_normals is not None:
            self.vertex_normals = _apply_normals(self.vertex_normals, trans)
        if self.face_normals is not None:
            self.face_normals = _apply_normals(self.face_normals, trans)
        if self.sampled_normals is not None:
            self.sampled_normals = _apply_normals(self.sampled_normals, trans)

class Exporter(): # a simple parser
    
    @classmethod
    def _safe_make_dir(cls, path: str):
        if os.path.dirname(path) == '':
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
    
    @classmethod
    def export_obj(cls, vertices, path: str, precision: int=6):
        lines = []
        for v in vertices:
            lines.append(f'v {v[0]:.{precision}f} {v[2]:.{precision}f} {-v[1]:.{precision}f}\n')
        cls._safe_make_dir(path)
        f = open(path, "w")
        f.writelines(lines)
        f.close()
