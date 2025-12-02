# Visualize VGGT Inference Process via DPT Logit Lens

This web demo explores how VGGT's geometry emerges across layers using a "logit lens"-like style ablation. Inspired by the paper [Understanding Multi-View Transformers (Stary and Gaubil et al., 2025)](https://arxiv.org/abs/2510.24907).  
It
- loads multiple views, swaps DPT inputs with earlier layers, and runs DPT depth head (and camera head)
- canonicalizes poses, aligns scales, and subsamples colored point clouds
- visualizes static 3D layers and interpolated motion in Gradio, and streams interactive 3D via Viser


Setup
```bash
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/vggt.git
```


Run
```bash
python run_g2.py
```

