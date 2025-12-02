from typing import Any, Dict, List, Optional

import numpy as np
import torch

from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from .geometry import (
    canonicalize_layer_by_view0,
    get_c2w_from_extrinsic,
    get_extrinsic_from_c2w,
    pad_extrinsic_to_4x4,
)
from .logging_config import setup_logger

logger = setup_logger()


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

        logger.info("--- Canonicalizing Layers (View 0 -> Identity) ---")
        canonical_data: List[Dict[str, Any]] = []
        for cfg in temp_data:
            cfg = canonicalize_layer_by_view0(cfg)
            canonical_data.append(cfg)
        temp_data = canonical_data

        logger.info("--- Depth-based scale alignment to final layer ---")
        final_cfg = temp_data[-1]
        final_depth = final_cfg["depth"][:, :, :, 0]

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
                    s_layer = float(np.median(s_vals))
                else:
                    s_layer = 1.0

            layer_scales.append(s_layer)
            logger.info(
                "Layer %s: depth-based scale factor = %.4f",
                cfg["label"],
                s_layer,
            )

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
