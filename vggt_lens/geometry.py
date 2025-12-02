import colorsys
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R


def get_c2w_from_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
    """Invert extrinsic (W2C) to get C2W."""
    return np.linalg.inv(extrinsic)


def get_extrinsic_from_c2w(c2w: np.ndarray) -> np.ndarray:
    return np.linalg.inv(c2w)


def canonicalize_layer_by_view0(config_data):
    """
    Transform layer coordinates so view 0 is the origin (Extrinsic = I).
    """
    extrinsics = np.asarray(config_data["extrinsic"])
    points = np.asarray(config_data["points"])
    num_views = len(extrinsics)

    T_canon = extrinsics[0]

    new_extrinsics = []
    new_points = []

    R_c = T_canon[:3, :3]
    t_c = T_canon[:3, 3]

    for i in range(num_views):
        C2Wi = get_c2w_from_extrinsic(extrinsics[i])
        C2Wi_new = T_canon @ C2Wi
        new_extrinsics.append(get_extrinsic_from_c2w(C2Wi_new))

        pts = points[i]
        flat_pts = pts.reshape(-1, 3)
        flat_pts_new = (flat_pts @ R_c.T) + t_c
        new_points.append(flat_pts_new.reshape(pts.shape))

    config_data["extrinsic"] = np.array(new_extrinsics, dtype=np.float32)
    config_data["points"] = np.array(new_points, dtype=np.float32)
    return config_data


def create_frustum_mesh_c2w(
    c2w: np.ndarray,
    intrinsic: np.ndarray,
    color=[1, 0, 0, 1],
    scale: float = 0.1,
) -> trimesh.Trimesh:
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    if abs(fx) < 1e-4:
        fx = 1.0
    if abs(fy) < 1e-4:
        fy = 1.0

    W = (scale / fx) * cx * 2.0
    H = (scale / fy) * cy * 2.0

    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [-W / 2.0, -H / 2.0, scale],
            [W / 2.0, -H / 2.0, scale],
            [W / 2.0, H / 2.0, scale],
            [-W / 2.0, H / 2.0, scale],
        ]
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 1],
            [1, 4, 3],
            [1, 3, 2],
        ]
    )

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.apply_transform(c2w)
    if len(color) == 3:
        color = [*color, 1.0]
    mesh.visual.vertex_colors = np.tile(color, (len(mesh.vertices), 1))
    return mesh


def interpolate_c2w(C2W1: np.ndarray, C2W2: np.ndarray, alpha: float) -> np.ndarray:
    t1, R1 = C2W1[:3, 3], C2W1[:3, :3]
    t2, R2 = C2W2[:3, 3], C2W2[:3, :3]

    t_interp = (1.0 - alpha) * t1 + alpha * t2

    q1 = R.from_matrix(R1).as_quat()
    q2 = R.from_matrix(R2).as_quat()
    if np.dot(q1, q2) < 0.0:
        q2 = -q2
    q_interp = (1.0 - alpha) * q1 + alpha * q2
    q_interp /= np.linalg.norm(q_interp)

    C2W_interp = np.eye(4, dtype=np.float64)
    C2W_interp[:3, :3] = R.from_quat(q_interp).as_matrix()
    C2W_interp[:3, 3] = t_interp
    return C2W_interp


def get_view_colors(n: int) -> np.ndarray:
    colors = []
    for i in range(n):
        hue = i / max(n, 1)
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append(np.array(rgb))
    return np.array(colors)


def normalize_depth_turbo(
    depth_map: np.ndarray,
    global_min: float,
    global_max: float,
) -> np.ndarray:
    valid_mask = depth_map > 1e-4
    if not valid_mask.any():
        return np.zeros((*depth_map.shape, 3), dtype=np.uint8)
    d_clamped = np.clip(depth_map, global_min, global_max)
    d_norm = (d_clamped - global_min) / (global_max - global_min + 1e-8)
    cmap = plt.get_cmap("turbo")
    d_color = cmap(d_norm)[:, :, :3]
    d_color[~valid_mask] = 0.0
    return (d_color * 255).astype(np.uint8)


def normalize_diff_inferno(
    diff_map: np.ndarray,
    global_diff_max: float,
) -> np.ndarray:
    d_norm = np.clip(diff_map / (global_diff_max + 1e-8), 0.0, 1.0)
    cmap = plt.get_cmap("inferno")
    d_color = cmap(d_norm)[:, :, :3]
    return (d_color * 255).astype(np.uint8)


def pad_extrinsic_to_4x4(extrinsic: np.ndarray) -> np.ndarray:
    if isinstance(extrinsic, np.ndarray):
        if extrinsic.shape[-2:] == (4, 4):
            return extrinsic
        if extrinsic.shape[-2:] == (3, 4):
            pad = np.array([0.0, 0.0, 0.0, 1.0], dtype=extrinsic.dtype)
            pad = np.broadcast_to(
                pad,
                extrinsic.shape[:-2] + (1, 4),
            )
            return np.concatenate([extrinsic, pad], axis=-2)
    return extrinsic


def create_instanced_points(
    points: np.ndarray,
    colors: np.ndarray,
    size: float = 0.02,
) -> trimesh.Trimesh | trimesh.PointCloud:
    if size <= 0.02:
        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
        if colors.shape[1] == 3:
            alpha = np.full((len(colors), 1), 255, dtype=np.uint8)
            colors = np.hstack([colors, alpha])
        return trimesh.PointCloud(vertices=points, colors=colors)

    s = size * 0.5
    base_verts = np.array(
        [
            [0.0, 0.0, s],
            [s, 0.0, -s * 0.3],
            [-s * 0.5, s * 0.8, -s * 0.3],
            [-s * 0.5, -s * 0.8, -s * 0.3],
        ]
    )
    base_faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 1],
            [1, 3, 2],
        ]
    )

    n_pts = len(points)
    n_v = len(base_verts)
    final_verts = (base_verts[None, :, :] + points[:, None, :]).reshape(-1, 3)
    offsets = np.arange(n_pts) * n_v
    final_faces = (base_faces[None, :, :] + offsets[:, None, None]).reshape(-1, 3)

    if colors.max() <= 1.0:
        colors = (colors * 255).astype(np.uint8)
    if colors.shape[1] == 3:
        alpha = np.full((len(colors), 1), 255, dtype=np.uint8)
        colors = np.hstack([colors, alpha])
    final_colors = np.repeat(colors, n_v, axis=0)
    return trimesh.Trimesh(
        vertices=final_verts,
        faces=final_faces,
        vertex_colors=final_colors,
    )
