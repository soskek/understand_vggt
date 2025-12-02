import os
import sys
import logging
import tempfile
import traceback
import colorsys
import io
import time
import threading
import pandas as pd
from typing import List, Dict, Any, Optional

import torch
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import gradio as gr
from PIL import Image
from scipy.spatial.transform import Rotation as R

# Viser for interactive 3D
import viser

# VGGT Modules (Dummy if not present)
try:
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map
except ImportError:
    class VGGT:
        pass

    def load_and_preprocess_images(*args):
        pass

    def pose_encoding_to_extri_intri(*args):
        pass

    def unproject_depth_map_to_point_map(*args):
        pass


# ==========================================
# 0. Logging & Config
# ==========================================
logger = logging.getLogger("VGGT-Lens")
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
)
logger.addHandler(stdout_handler)

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap');
body, .gradio-container { font-family: 'Inter', sans-serif !important; }
.model-3d-container { border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden; margin-bottom: 20px; }
.viser-link { font-weight: bold; color: #2563eb; text-decoration: none; font-size: 1.1em; }
.viser-link:hover { text-decoration: underline; }
"""


# ==========================================
# 1. Math & Geometry Helpers
# ==========================================

def get_c2w_from_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
    """Invert extrinsic (W2C) to get C2W."""
    return np.linalg.inv(extrinsic)


def get_extrinsic_from_c2w(c2w: np.ndarray) -> np.ndarray:
    return np.linalg.inv(c2w)


def canonicalize_layer_by_view0(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transforms the entire layer's coordinate system such that
    View 0 is at the origin with identity rotation (Extrinsic = I).

    T_canon = (C2W_view0)^-1 = Extrinsic_view0
    """
    extrinsics = np.asarray(config_data["extrinsic"])
    points = np.asarray(config_data["points"])
    num_views = len(extrinsics)

    T_canon = extrinsics[0]  # W2C of view 0

    new_extrinsics = []
    new_points = []

    R_c = T_canon[:3, :3]
    t_c = T_canon[:3, 3]

    for i in range(num_views):
        # 1. Transform Camera
        C2Wi = get_c2w_from_extrinsic(extrinsics[i])
        C2Wi_new = T_canon @ C2Wi
        new_extrinsics.append(get_extrinsic_from_c2w(C2Wi_new))

        # 2. Transform Points: p_new = R_c * p_old + t_c
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


def plot_camera_drift(cache_data: Dict[str, Any]) -> Optional[Image.Image]:
    if not cache_data:
        return None
    configs = cache_data["configs"]
    layer_labels = [c["label"] for c in configs]
    x = range(len(layer_labels))

    trans_drifts: List[float] = []
    num_views = cache_data["num_views"]
    if num_views < 2:
        return None

    final_c2ws = [
        get_c2w_from_extrinsic(E) for E in configs[-1]["extrinsic"]
    ]

    for config in configs:
        curr_c2ws = [
            get_c2w_from_extrinsic(E) for E in config["extrinsic"]
        ]
        t_errs = []
        for i in range(1, num_views):
            t_errs.append(
                float(
                    np.linalg.norm(
                        curr_c2ws[i][:3, 3] - final_c2ws[i][:3, 3]
                    )
                )
            )
        trans_drifts.append(float(np.mean(t_errs)) if t_errs else 0.0)

    fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))
    ax1.plot(x, trans_drifts, "o-", color="tab:blue", linewidth=2.0)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(layer_labels)
    ax1.set_title("Camera Drift (vs Final)")
    ax1.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def format_pose_dataframe(cache_data: Dict[str, Any]) -> pd.DataFrame:
    """Create a pandas DataFrame of all poses for Gradio."""
    if not cache_data:
        return pd.DataFrame()
    rows = []
    for cfg in cache_data["configs"]:
        layer = cfg["label"]
        for i, ext in enumerate(cfg["extrinsic"]):
            row = {
                "Layer": layer,
                "View": i,
                "Tx": ext[0, 3],
                "Ty": ext[1, 3],
                "Tz": ext[2, 3],
                "R00": ext[0, 0],
                "R01": ext[0, 1],
                "R02": ext[0, 2],
                "Is_Identity": np.allclose(ext, np.eye(4), atol=1e-5)
                if i == 0
                else "",
            }
            rows.append(row)
    return pd.DataFrame(rows)


# ==========================================
# 2. Inference Class
# ==========================================

class VGGTInference:
    def __init__(self, model_name: str = "facebook/VGGT-1B", device: str = "cuda"):
        self.device = device
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float16
        logger.info("Loading VGGT on %s (%s)...", device, self.dtype)
        self.model = VGGT()
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        try:
            state_dict = torch.hub.load_state_dict_from_url(
                _URL,
                map_location=device,
            )
            self.model.load_state_dict(state_dict)
        except Exception:
            logger.warning("Download failed, trying local.")
            self.model = VGGT.from_pretrained(model_name)

        self.model.to(device).eval()

    def match_activation_stats(
        self,
        target_tokens: torch.Tensor,
        source_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simple scale matching:
        target = target * (std_source / std_target)
        """
        src_std = source_tokens.std()
        tgt_std = target_tokens.std()
        scale = src_std / (tgt_std + 1e-8)
        return target_tokens * scale

    @torch.no_grad()
    def infer_all_layers(
        self,
        image_paths: List[str],
        match_std: bool = False,
    ) -> Optional[Dict[str, Any]]:
        if not image_paths:
            return None

        raw_images = load_and_preprocess_images(image_paths).to(self.device)
        raw_images = raw_images.to(self.dtype)
        if raw_images.ndim == 4:
            images = raw_images.unsqueeze(0)
        else:
            images = raw_images
        batch_size, num_views, _, height, width = images.shape
        logger.info("Inference: %d images. Match STD: %s", num_views, match_std)
        assert batch_size == 1, "This demo assumes batch size 1."

        # 1) Get all tokens once
        with torch.amp.autocast("cuda", dtype=self.dtype):
            full_tokens_list, ps_idx = self.model.aggregator(images)
            num_layers = len(full_tokens_list)

        results_cache: Dict[str, Any] = {}
        if num_layers == 24:
            layer_specs = [
                ("Layer 4", 3),
                ("Layer 11", 10),
                ("Layer 17", 16),
                ("Layer 24", 23),
            ]
        else:
            layer_specs = [("Output", num_layers - 1)] * 4

        results_cache["images_np"] = (
            images[0].permute(0, 2, 3, 1).float().cpu().numpy()
        )
        results_cache["num_views"] = num_views

        temp_data: List[Dict[str, Any]] = []

        # 2) レンズごとの推論結果を集める
        for label, source_idx in layer_specs:
            lens_tokens = list(full_tokens_list)
            source_token = lens_tokens[source_idx]

            std_val = source_token.std().item()
            logger.info("[%s] Source Token STD: %.4f", label, std_val)

            for i_layer in range(source_idx + 1, num_layers):
                if match_std:
                    adjusted_token = self.match_activation_stats(
                        source_token,
                        full_tokens_list[i_layer],
                    )
                    lens_tokens[i_layer] = adjusted_token
                else:
                    lens_tokens[i_layer] = source_token

            with torch.amp.autocast("cuda", dtype=self.dtype):
                pose_enc = self.model.camera_head(lens_tokens)[-1]
                extrinsic, intrinsic = pose_encoding_to_extri_intri(
                    pose_enc,
                    images.shape[-2:],
                )

                out = self.model.depth_head(lens_tokens, images, ps_idx)
                if isinstance(out, tuple):
                    depth_map, conf_map = out
                else:
                    depth_map = out
                    conf_map = torch.ones_like(depth_map)

            d_np = depth_map[0].float().cpu().numpy()
            c_np = conf_map[0].float().cpu().numpy()
            e_np = pad_extrinsic_to_4x4(
                extrinsic[0].float().cpu().numpy(),
            )
            i_np = intrinsic[0].float().cpu().numpy()

            if d_np.ndim == 3:
                d_np = np.expand_dims(d_np, axis=-1)
            elif d_np.ndim == 4 and d_np.shape[1] == 1:
                d_np = np.moveaxis(d_np, 1, -1)

            if c_np.ndim == 3:
                c_np = np.expand_dims(c_np, axis=-1)
            elif c_np.ndim == 4 and c_np.shape[1] == 1:
                c_np = np.moveaxis(c_np, 1, -1)

            world_points = unproject_depth_map_to_point_map(d_np, e_np, i_np)

            temp_data.append(
                {
                    "label": label,
                    "depth": d_np.astype(np.float32),
                    "conf": c_np.astype(np.float32),
                    "points": world_points.astype(np.float32),
                    "extrinsic": e_np.astype(np.float32),
                    "intrinsic": i_np.astype(np.float32),
                }
            )

        # 3) Canonicalization: 全レイヤーを「View 0 = I」の座標系に揃える
        logger.info("--- Canonicalizing Layers (View 0 -> Identity) ---")
        canonical_data: List[Dict[str, Any]] = []
        for cfg in temp_data:
            cfg = canonicalize_layer_by_view0(cfg)
            canonical_data.append(cfg)
        temp_data = canonical_data

        # 4) Depth ベースのスケール合わせ（各レイヤーを final depth に L2 フィット）
        logger.info("--- Depth-based scale alignment to final layer ---")
        final_cfg = temp_data[-1]
        final_depth = final_cfg["depth"][:, :, :, 0]  # (N,H,W)

        layer_scales: List[float] = []
        for cfg_idx, cfg in enumerate(temp_data):
            if cfg is final_cfg:
                s_layer = 1.0
            else:
                s_vals: List[float] = []
                for v in range(num_views):
                    d_l = cfg["depth"][v, :, :, 0]
                    d_f = final_depth[v]

                    mask = (d_l > 1e-4) & (d_f > 1e-4)
                    if mask.sum() < 100:
                        continue

                    dl = d_l[mask].astype(np.float64)
                    df = d_f[mask].astype(np.float64)
                    denom = float(np.dot(dl, dl))
                    if denom < 1e-6:
                        continue
                    s_i = float(np.dot(dl, df) / denom)
                    if np.isfinite(s_i):
                        s_vals.append(s_i)

                if s_vals:
                    # 外れ値に強くするため median を採用
                    s_layer = float(np.median(s_vals))
                else:
                    s_layer = 1.0

            layer_scales.append(s_layer)
            logger.info(
                "Layer %s: depth-based scale factor = %.4f",
                cfg["label"],
                s_layer,
            )

        # 5) スケール適用（depth / points / camera translation を同じ割合で拡大縮小）
        for cfg, s_layer in zip(temp_data, layer_scales):
            if abs(s_layer - 1.0) < 1e-4:
                continue

            cfg["depth"] *= s_layer
            cfg["points"] *= s_layer

            new_extrinsics = []
            for v in range(num_views):
                C2W = get_c2w_from_extrinsic(cfg["extrinsic"][v])
                C2W[:3, 3] *= s_layer
                new_extrinsics.append(get_extrinsic_from_c2w(C2W))
            cfg["extrinsic"] = np.array(new_extrinsics, dtype=np.float32)

        # 6) baseline（frustum のデフォルトスケール用）を最終層カメラ距離から算出
        final_cams_c2w = np.array(
            [get_c2w_from_extrinsic(E)[:3, 3] for E in final_cfg["extrinsic"]]
        )
        dists = np.linalg.norm(final_cams_c2w, axis=1)
        if len(dists) > 1:
            ref_scale = float(np.mean(dists[1:]))
        elif len(dists) == 1:
            ref_scale = float(dists[0])
        else:
            ref_scale = 1.0
        logger.info("Reference baseline distance (final layer): %.4f", ref_scale)

        # 7) depth_conf を使った master mask + subsampling
        final_d = temp_data[-1]["depth"]
        final_c = temp_data[-1]["conf"]

        master_mask = (final_d > 1e-4) & (final_c > 0.6)
        mask_3d = master_mask.squeeze(-1)

        subsample_indices: List[np.ndarray] = []
        all_depth_values: List[np.ndarray] = []

        for v in range(num_views):
            view_mask = master_mask[v].flatten()
            valid_idx = np.where(view_mask)[0]
            if len(valid_idx) > 20000:
                chosen = np.random.choice(valid_idx, 20000, replace=False)
            else:
                chosen = valid_idx
            chosen.sort()
            subsample_indices.append(chosen)

        for cfg in temp_data:
            cfg["depth"][~master_mask] = 0.0
            cfg["points"][~mask_3d] = 0.0
            valid_d = cfg["depth"][master_mask]
            if len(valid_d) > 0:
                all_depth_values.append(valid_d)

        results_cache["subsample_indices"] = subsample_indices
        results_cache["configs"] = temp_data
        results_cache["baseline"] = ref_scale

        if all_depth_values:
            flat_d = np.concatenate(all_depth_values)
            results_cache["g_min"] = float(np.percentile(flat_d, 1.0))
            results_cache["g_max"] = float(np.percentile(flat_d, 99.0))
        else:
            results_cache["g_min"], results_cache["g_max"] = 0.0, 1.0

        logger.info("Inference Complete.")
        return results_cache


# ==========================================
# 3. Visualization Generators
# ==========================================

def generate_static_visualizations(
    cache_data: Dict[str, Any],
    mix_ratio: float,
    pt_size: float,
    cam_size_coeff: float,
    visible_views: List[int],
):
    if not cache_data:
        return [None] * 16
    num_views = cache_data["num_views"]
    images_np = cache_data["images_np"]
    view_colors = get_view_colors(num_views)
    g_min, g_max = cache_data["g_min"], cache_data["g_max"]
    subsample_indices = cache_data["subsample_indices"]

    base_scale = cache_data.get("baseline", 1.0)
    actual_cam_size = base_scale * cam_size_coeff

    final_depth = cache_data["configs"][-1]["depth"]
    max_diff_val = 1.0

    out_glbs, out_depths, out_df, out_dp = [], [], [], []
    transform_to_yup = trimesh.transformations.rotation_matrix(
        np.pi,
        [1.0, 0.0, 0.0],
    )

    for cfg_idx, config in enumerate(cache_data["configs"]):
        scene = trimesh.Scene()
        points_list: List[np.ndarray] = []
        colors_list: List[np.ndarray] = []
        vis_d_list: List[np.ndarray] = []
        vis_df_list: List[np.ndarray] = []
        vis_dp_list: List[np.ndarray] = []

        prev_depth = (
            cache_data["configs"][cfg_idx - 1]["depth"]
            if cfg_idx > 0
            else None
        )

        for i in range(num_views):
            d_raw = config["depth"][i, :, :, 0]
            vis_d_list.append(normalize_depth_turbo(d_raw, g_min, g_max))
            vis_df_list.append(
                normalize_diff_inferno(
                    np.abs(d_raw - final_depth[i, :, :, 0]),
                    max_diff_val,
                )
            )
            if prev_depth is not None:
                vis_dp_list.append(
                    normalize_diff_inferno(
                        np.abs(d_raw - prev_depth[i, :, :, 0]),
                        max_diff_val,
                    )
                )
            else:
                vis_dp_list.append(
                    np.zeros((*d_raw.shape, 3), dtype=np.uint8),
                )

            if i not in visible_views:
                continue

            chosen = subsample_indices[i]
            if len(chosen) > 0:
                pts = config["points"][i].reshape(-1, 3)[chosen]
                c_orig = images_np[i].reshape(-1, 3)[chosen]
                if c_orig.max() > 1.1:
                    c_orig = c_orig / 255.0
                c_orig = np.clip(c_orig, 0.0, 1.0)
                c_mixed = c_orig * (1.0 - mix_ratio) + view_colors[i] * mix_ratio
                points_list.append(pts)
                colors_list.append(c_mixed)

            c2w = get_c2w_from_extrinsic(config["extrinsic"][i])
            frustum = create_frustum_mesh_c2w(
                c2w,
                config["intrinsic"][i],
                np.concatenate([view_colors[i], [0.8]]),
                scale=actual_cam_size,
            )
            scene.add_geometry(frustum)

        if points_list:
            all_pts = np.concatenate(points_list, axis=0)
            all_cols = np.concatenate(colors_list, axis=0)
            mesh_or_pts = create_instanced_points(all_pts, all_cols, size=pt_size)
            scene.add_geometry(mesh_or_pts)
        else:
            scene.add_geometry(trimesh.creation.axis(origin_size=0.1))

        scene.apply_transform(transform_to_yup)
        tmp = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
        scene.export(tmp.name)
        out_glbs.append(tmp.name)
        out_depths.append(
            np.concatenate(vis_d_list, axis=1) if vis_d_list else None
        )
        out_df.append(
            np.concatenate(vis_df_list, axis=1) if vis_df_list else None
        )
        out_dp.append(
            np.concatenate(vis_dp_list, axis=1) if vis_dp_list else None
        )

    return out_glbs, out_depths, out_df, out_dp


def generate_motion_frame(
    cache_data: Dict[str, Any],
    time_val: float,
    mix_ratio: float,
    pt_size: float,
    cam_size_coeff: float,
    visible_views: List[int],
) -> Optional[str]:
    if not cache_data:
        return None
    configs = cache_data["configs"]
    num_views = cache_data["num_views"]
    subsample_indices = cache_data["subsample_indices"]
    images_np = cache_data["images_np"]
    view_colors = get_view_colors(num_views)

    base_scale = cache_data.get("baseline", 1.0)
    actual_cam_size = base_scale * cam_size_coeff

    idx = int(np.floor(time_val))
    alpha = time_val - float(idx)
    if idx >= 3:
        idx, alpha = 2, 1.0
    cfg_start = configs[idx]
    cfg_end = configs[idx + 1]

    scene = trimesh.Scene()
    points_list: List[np.ndarray] = []
    colors_list: List[np.ndarray] = []

    for i in range(num_views):
        if i not in visible_views:
            continue

        C2W1 = get_c2w_from_extrinsic(cfg_start["extrinsic"][i])
        C2W2 = get_c2w_from_extrinsic(cfg_end["extrinsic"][i])
        C2W_interp = interpolate_c2w(C2W1, C2W2, alpha)

        chosen = subsample_indices[i]
        if len(chosen) > 0:
            P1 = cfg_start["points"][i].reshape(-1, 3)[chosen]
            P2 = cfg_end["points"][i].reshape(-1, 3)[chosen]
            P_interp = (1.0 - alpha) * P1 + alpha * P2

            c_orig = images_np[i].reshape(-1, 3)[chosen]
            if c_orig.max() > 1.1:
                c_orig = c_orig / 255.0
            c_orig = np.clip(c_orig, 0.0, 1.0)
            c_mixed = c_orig * (1.0 - mix_ratio) + view_colors[i] * mix_ratio
            points_list.append(P_interp)
            colors_list.append(c_mixed)

        frustum = create_frustum_mesh_c2w(
            C2W_interp,
            cfg_start["intrinsic"][i],
            np.concatenate([view_colors[i], [0.8]]),
            scale=actual_cam_size,
        )
        scene.add_geometry(frustum)

    if points_list:
        all_pts = np.concatenate(points_list, axis=0)
        all_cols = np.concatenate(colors_list, axis=0)
        mesh_or_pts = create_instanced_points(all_pts, all_cols, size=pt_size)
        scene.add_geometry(mesh_or_pts)

    scene.apply_transform(
        trimesh.transformations.rotation_matrix(
            np.pi,
            [1.0, 0.0, 0.0],
        )
    )
    tmp = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
    scene.export(tmp.name)
    return tmp.name


# ==========================================
# 4. Viser Manager
# ==========================================

class ViserManager:
    def __init__(self, port: int = 8080) -> None:
        self.server = viser.ViserServer(port=port)
        self.current_cache: Optional[Dict[str, Any]] = None
        self.slider_time = None
        self.slider_pt_size = None
        self.slider_cam_size = None
        self.slider_mix = None
        self.checkbox_play = None
        self.playing = False

        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            client.camera.position = (0.0, 0.0, -2.0)
            client.camera.look_at = (0.0, 0.0, 1.0)
            self._setup_ui(client)
            if self.current_cache:
                self._update_scene(client)

    def _setup_ui(self, client: viser.ClientHandle) -> None:
        with client.gui.add_folder("Timeline"):
            self.slider_time = client.gui.add_slider(
                "Layer Time",
                min=0.0,
                max=3.0,
                step=0.05,
                initial_value=3.0,
            )
            self.checkbox_play = client.gui.add_checkbox(
                "Auto Play",
                initial_value=False,
            )
        with client.gui.add_folder("Appearance"):
            self.slider_pt_size = client.gui.add_slider(
                "Point Size",
                min=0.001,
                max=0.05,
                step=0.001,
                initial_value=0.005,
            )
            self.slider_cam_size = client.gui.add_slider(
                "Cam Scale Coeff",
                min=0.01,
                max=1.0,
                step=0.01,
                initial_value=0.2,
            )
            self.slider_mix = client.gui.add_slider(
                "View Color Mix",
                min=0.0,
                max=1.0,
                step=0.1,
                initial_value=0.0,
            )

        @self.slider_time.on_update
        def _(_event: viser.GuiEvent) -> None:
            if self.current_cache and not self.playing:
                self._update_scene(client)

        @self.slider_pt_size.on_update
        def _(_event: viser.GuiEvent) -> None:
            if self.current_cache:
                self._update_scene(client)

        @self.slider_cam_size.on_update
        def _(_event: viser.GuiEvent) -> None:
            if self.current_cache:
                self._update_scene(client)

        @self.slider_mix.on_update
        def _(_event: viser.GuiEvent) -> None:
            if self.current_cache:
                self._update_scene(client)

        @self.checkbox_play.on_update
        def _(_event: viser.GuiEvent) -> None:
            if self.checkbox_play.value:
                self.playing = True
                self._start_play_loop(client)
            else:
                self.playing = False

    def _start_play_loop(self, client: viser.ClientHandle) -> None:
        def loop() -> None:
            while self.playing:
                if self.slider_time is None:
                    break
                current = self.slider_time.value
                nxt = current + 0.05
                if nxt > 3.0:
                    nxt = 0.0
                self.slider_time.value = nxt
                if self.current_cache:
                    self._update_scene(client)
                time.sleep(0.05)

        threading.Thread(target=loop, daemon=True).start()

    def update_data(self, cache_data: Dict[str, Any]) -> None:
        self.current_cache = cache_data
        for client in self.server.get_clients().values():
            self._update_scene(client)

    def _update_scene(self, client: viser.ClientHandle) -> None:
        if not self.current_cache:
            return
        time_val = self.slider_time.value
        pt_size = self.slider_pt_size.value
        cam_coeff = self.slider_cam_size.value
        mix = self.slider_mix.value

        configs = self.current_cache["configs"]
        num_views = self.current_cache["num_views"]
        subsample_indices = self.current_cache["subsample_indices"]
        images_np = self.current_cache["images_np"]
        view_colors = get_view_colors(num_views)

        base_scale = self.current_cache.get("baseline", 1.0)
        actual_cam_size = base_scale * cam_coeff

        idx = int(np.floor(time_val))
        alpha = time_val - float(idx)
        if idx >= 3:
            idx, alpha = 2, 1.0
        cfg_start = configs[idx]
        cfg_end = configs[idx + 1]

        for i in range(num_views):
            C2W1 = get_c2w_from_extrinsic(cfg_start["extrinsic"][i])
            C2W2 = get_c2w_from_extrinsic(cfg_end["extrinsic"][i])
            C2W_interp = interpolate_c2w(C2W1, C2W2, alpha)

            chosen = subsample_indices[i]
            if len(chosen) > 0:
                P1 = cfg_start["points"][i].reshape(-1, 3)[chosen]
                P2 = cfg_end["points"][i].reshape(-1, 3)[chosen]
                P_interp = (1.0 - alpha) * P1 + alpha * P2

                c_orig = images_np[i].reshape(-1, 3)[chosen]
                if c_orig.max() > 1.1:
                    c_orig = c_orig / 255.0
                c_orig = np.clip(c_orig, 0.0, 1.0)
                c_mixed = c_orig * (1.0 - mix) + view_colors[i] * mix
                client.scene.add_point_cloud(
                    f"/view_{i}/points",
                    points=P_interp,
                    colors=c_mixed,
                    point_size=pt_size,
                )

            frustum_mesh = create_frustum_mesh_c2w(
                C2W_interp,
                cfg_start["intrinsic"][i],
                np.concatenate([view_colors[i], [1.0]]),
                scale=actual_cam_size,
            )
            client.scene.add_mesh_trimesh(
                f"/view_{i}/frustum",
                mesh=frustum_mesh,
            )

    def get_url(self) -> str:
        return f"http://localhost:{self.server.get_port()}"


# ==========================================
# 5. Gradio UI
# ==========================================

def launch_demo() -> None:
    try:
        model_engine = VGGTInference()
        viser_manager = ViserManager(port=8080)
    except Exception as exc:
        print(f"Initialization Error: {exc}")
        return

    theme = gr.themes.Soft(font=[gr.themes.GoogleFont("Inter")])

    with gr.Blocks(
        title="VGGT Logit Lens",
        theme=theme,
        css=CUSTOM_CSS,
    ) as demo:
        state_cache = gr.State({})

        gr.Markdown("## 🔭 VGGT Logit Lens: Geometry Emergence Explorer")

        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### 1. Input")
                file_input = gr.File(
                    file_count="multiple",
                    label="Images (Auto-runs on upload)",
                )

                gr.Markdown("### 2. Ablation Settings")
                match_std_chk = gr.Checkbox(
                    label="Match Activation STD for DPT",
                    value=False,
                )

                gr.Markdown("### 3. Viser Visualization")
                viser_url = viser_manager.get_url()
                gr.HTML(
                    f"""
                <div style="background-color: #f0f9ff; padding: 15px; border-radius: 8px; border: 1px solid #bae6fd;">
                    <a href="{viser_url}" target="_blank" class="viser-link">🔗 Open Viser</a>
                </div>"""
                )

                gr.Markdown("### 4. Visual Settings")
                mix_slider = gr.Slider(
                    0.0,
                    1.0,
                    value=0.5,
                    label="Color Mix",
                )
                pt_size_slider = gr.Slider(
                    0.01,
                    0.1,
                    value=0.01,
                    step=0.01,
                    label="Point Size",
                )
                cam_size_slider = gr.Slider(
                    0.01,
                    2.0,
                    value=0.2,
                    step=0.01,
                    label="Cam Scale (Relative)",
                )
                view_toggles = gr.CheckboxGroup(
                    label="Views",
                    choices=[],
                    value=[],
                )

                drift_plot = gr.Image(
                    label="Drift",
                    show_label=False,
                )

                gr.Markdown("### 5. Camera Poses (Extrinsic)")
                pose_df = gr.Dataframe(
                    label="Canonical Poses",
                    wrap=True,
                    interactive=False,
                )

            with gr.Column(scale=4):
                with gr.Tabs():
                    with gr.Tab("Vertical Stack"):
                        model_outputs: List[tuple] = []

                        def create_cell(label: str):
                            with gr.Column(elem_classes="model-3d-container"):
                                gr.Markdown(f"### {label}")
                                m = gr.Model3D(
                                    clear_color=[0.97, 0.97, 0.97, 1.0],
                                    height=400,
                                    show_label=False,
                                )
                                with gr.Row():
                                    d = gr.Image(
                                        show_label=False,
                                        label="Depth",
                                        height=150,
                                    )
                                    df = gr.Image(
                                        show_label=False,
                                        label="Diff Final",
                                        height=150,
                                    )
                                    dp = gr.Image(
                                        show_label=False,
                                        label="Diff Prev",
                                        height=150,
                                    )
                                return m, d, df, dp

                        model_outputs.append(create_cell("Layer 4"))
                        model_outputs.append(create_cell("Layer 11"))
                        model_outputs.append(create_cell("Layer 17"))
                        model_outputs.append(create_cell("Layer 24 (Ref)"))

                    with gr.Tab("Motion Interpolation"):
                        time_slider = gr.Slider(
                            0.0,
                            3.0,
                            value=3.0,
                            step=0.1,
                            label="Evolution Time (L4 -> L24)",
                        )
                        motion_model = gr.Model3D(
                            clear_color=[0.95, 0.95, 0.95, 1.0],
                            height=800,
                            label="Interpolated Geometry",
                        )

        def run_inference(files, match_std: bool):
            if not files:
                return {}, None, None, gr.update()
            try:
                cache = model_engine.infer_all_layers(
                    sorted(f.name for f in files),
                    match_std=match_std,
                )
            except Exception as exc:
                traceback.print_exc()
                logger.error("Inference failed: %s", exc)
                return {}, None, None, gr.update()

            viser_manager.update_data(cache)
            drift_img = plot_camera_drift(cache)
            df = format_pose_dataframe(cache)
            choices = [f"View {i}" for i in range(cache["num_views"])]
            return cache, drift_img, df, gr.update(
                choices=choices,
                value=choices,
            )

        def update_static(
            cache,
            mix,
            pt_size,
            cam_coeff,
            views,
        ):
            if not cache:
                return [None] * 16
            vis_idx = (
                [int(s.split(" ")[1]) for s in views]
                if views
                else []
            )
            glbs, depths, dfs, dps = generate_static_visualizations(
                cache,
                mix,
                pt_size,
                cam_coeff,
                vis_idx,
            )
            flat: List[Any] = []
            for i in range(4):
                flat.extend([glbs[i], depths[i], dfs[i], dps[i]])
            return flat

        def update_motion(
            cache,
            time_val,
            mix,
            pt_size,
            cam_coeff,
            views,
        ):
            if not cache:
                return None
            vis_idx = (
                [int(s.split(" ")[1]) for s in views]
                if views
                else []
            )
            return generate_motion_frame(
                cache,
                time_val,
                mix,
                pt_size,
                cam_coeff,
                vis_idx,
            )

        triggers = [file_input, match_std_chk]
        settings = [
            state_cache,
            mix_slider,
            pt_size_slider,
            cam_size_slider,
            view_toggles,
        ]

        for t in triggers:
            t.change(
                run_inference,
                [file_input, match_std_chk],
                [state_cache, drift_plot, pose_df, view_toggles],
            ).success(
                update_static,
                settings,
                [x for cell in model_outputs for x in cell],
            ).success(
                update_motion,
                [
                    state_cache,
                    time_slider,
                    mix_slider,
                    pt_size_slider,
                    cam_size_slider,
                    view_toggles,
                ],
                motion_model,
            )

        for sl in [mix_slider, pt_size_slider, cam_size_slider]:
            sl.release(
                update_static,
                settings,
                [x for cell in model_outputs for x in cell],
            )
            sl.release(
                update_motion,
                [
                    state_cache,
                    time_slider,
                    mix_slider,
                    pt_size_slider,
                    cam_size_slider,
                    view_toggles,
                ],
                motion_model,
            )

        view_toggles.change(
            update_static,
            settings,
            [x for cell in model_outputs for x in cell],
        )
        time_slider.change(
            update_motion,
            [
                state_cache,
                time_slider,
                mix_slider,
                pt_size_slider,
                cam_size_slider,
                view_toggles,
            ],
            motion_model,
        )

    demo.launch(server_name="0.0.0.0", share=True)


if __name__ == "__main__":
    launch_demo()
