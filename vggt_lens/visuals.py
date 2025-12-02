import io
import tempfile
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trimesh
from PIL import Image

from .geometry import (
    create_frustum_mesh_c2w,
    create_instanced_points,
    get_c2w_from_extrinsic,
    get_view_colors,
    interpolate_c2w,
    normalize_depth_turbo,
    normalize_diff_inferno,
)
from .logging_config import setup_logger

logger = setup_logger()


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
