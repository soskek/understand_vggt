import traceback
from typing import Any, List

import gradio as gr

from vggt_lens.inference import VGGTInference
from vggt_lens.logging_config import setup_logger
from vggt_lens.viser_manager import ViserManager
from vggt_lens.visuals import (
    format_pose_dataframe,
    generate_motion_frame,
    generate_static_visualizations,
    plot_camera_drift,
)


logger = setup_logger()

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap');
body, .gradio-container { font-family: 'Inter', sans-serif !important; }
.model-3d-container { border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden; margin-bottom: 20px; }
.viser-link { font-weight: bold; color: #2563eb; text-decoration: none; font-size: 1.1em; }
.viser-link:hover { text-decoration: underline; }
"""


# ==========================================
# Gradio UI
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
