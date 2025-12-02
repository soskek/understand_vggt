# Visualize VGGT Inference Process via DPT Logit Lens

This web demo explores how [VGGT](https://github.com/facebookresearch/vggt)'s geometry emerges across layers using a "logit lens"-like style ablation. Inspired by the paper [Understanding Multi-View Transformers (Stary and Gaubil et al., 2025)](https://arxiv.org/abs/2510.24907).  
This
- visualize point clouds and camera poses encoded in the intermediate hidden layers (by applying DPT head with swapping its inputs with earlier layers and applyng camera head)
- canonicalizes poses, aligns scales, and subsamples colored point clouds
- visualizes each result in Gradio, and combined (interpolated) motions via Viser.

The point clouds and camera poses are refined step by step.

<img width="884" height="134" src="https://github.com/user-attachments/assets/89d21da1-b101-44cf-84cd-2362809d221a" />

![Dec-02-2025 09-35-22](https://github.com/user-attachments/assets/1dd38abb-c1bb-414d-9ce1-68aa8af23b9a)
![Dec-02-2025 09-37-15](https://github.com/user-attachments/assets/85430098-60fd-4e79-8a65-276818eaa75f)

Setup
```bash
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/vggt.git
```


Run and open the page by web browsers.
```bash
python run_g2.py
```


----

<img width="629" height="875" src="https://github.com/user-attachments/assets/f0116c01-9acc-4c47-ac83-4c3d0f5f825c" />
<img width="545" height="382" alt="スクリーンショット 2025-12-02 10 38 32" src="https://github.com/user-attachments/assets/381ce718-5406-485f-b526-6cf21e44f4b0" />

----

Other Examples

https://github.com/user-attachments/assets/c4676d2e-fcb3-4532-ad24-e5d51d7c86a6

https://github.com/user-attachments/assets/224b1581-abea-4938-a158-c0200cb85eb7

