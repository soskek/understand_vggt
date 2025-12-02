import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np
import trimesh
import viser

from .geometry import (
    create_frustum_mesh_c2w,
    create_instanced_points,
    get_c2w_from_extrinsic,
    get_view_colors,
    interpolate_c2w,
)


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
