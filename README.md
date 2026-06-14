<div align="center">
GAMDFormer: A Gated Attention Modulation Transformer with Dual-Domain Interaction for Image Restoration
<p>
<a href="#installation">Installation</a> •
<a href="#training">Training</a> •
<a href="#evaluation">Evaluation</a> •
<a href="#results">Results</a> •
<a href="#citation">Citation</a>
</p>
A single unified transformer for image denoising, low-light enhancement, and single-image motion deblurring.
</div>
---
Overview
Restoring a clean image from a degraded observation is fundamentally ill-posed: the same network is expected to suppress noise, amplify under-exposed content, and invert motion blur, each of which stresses a different part of the signal. Convolutional priors capture local structure well but miss the long-range channel statistics that distinguish signal from degradation, while purely spatial self-attention overlooks the fact that many degradations are most naturally described in the frequency domain.
GAMDFormer addresses this with a hierarchical encoder–decoder transformer that couples a gated channel-attention mechanism with a dual-domain feed-forward network operating jointly in the spatial and spectral domains. A lightweight task-adaptive embedding lets one set of weights specialise its behaviour across three restoration problems without architectural changes.
<div align="center">
<img src="assets/architecture.png" width="100%" alt="GAMDFormer architecture"/>
<br/>
<em>Overall GAMDFormer architecture. [A] Transformer block, [B] RCG-MHA, [C] FGDD-FFN.</em>
</div>
---
Key Innovations
Refined Channel-Gated Multi-Head Attention (RCG-MHA).
Attention is computed across channels (transposed self-attention), keeping the cost linear in the number of pixels. Two complementary channel-affinity descriptors are formed — a softmax content-affinity map and a squared-SiLU, convolution-refined map — and fused by a Channel-Adaptive Modulation Gating (CAMG) unit that learns a per-channel convex gate from pooled first- and second-order descriptors. This lets the layer interpolate between globally correlated and locally structured channel relations on a per-input basis.
Frequency-Gated Dual-Domain Feed-Forward Network (FGDD-FFN).
The feed-forward layer processes features in two domains in parallel. A spatial path derives a global channel-wise modulation gate from pooled dilated and local descriptors through a SwiGLU non-linearity. A frequency path applies a differentiable 2-D DCT, splits the spectrum, re-weights the high-frequency half with an Adaptive Gated Frequency Pooling (AGFP) sigmoid gate, and inverts the transform. The spatial gate then modulates the frequency-refined content, explicitly steering the recovery of fine detail.
Task-adaptive conditioning.
A learned per-task embedding produces FiLM-style scale/shift parameters applied to the stem features, enabling a single model to serve denoising, low-light enhancement, and motion deblurring while sharing the bulk of its capacity.
Hierarchical multi-scale design.
A four-level encoder–decoder with pixel-(un)shuffle resampling and skip fusion aggregates context across scales, and a global residual connection lets the network predict the degradation rather than the full image.
---
Repository Structure
```
GAMDFormer/
├── gamdformer/
│   ├── models/
│   │   ├── gamdformer.py          # top-level model
│   │   ├── encoder.py             # multi-scale encoder
│   │   ├── decoder.py             # multi-scale decoder + refinement
│   │   ├── transformer_block.py   # [A] proposed transformer block
│   │   ├── attention.py           # [B] RCG-MHA + CAMG
│   │   ├── ffn.py                 # [C] FGDD-FFN (spatial + DCT frequency)
│   │   ├── task_embedding.py      # task-adaptive FiLM conditioning
│   │   └── common.py              # LayerNorm, resampling, DCT/IDCT
│   ├── datasets/                  # paired-image + multi-task datasets
│   ├── losses/                    # Charbonnier + frequency loss
│   └── utils/                     # config, metrics, image I/O, logging
├── configs/                       # YAML configs per task
├── scripts/                       # train / test / inference entry points
├── requirements.txt
└── setup.py
```
---
Installation
```bash
git clone https://github.com/<your-account>/GAMDFormer.git
cd GAMDFormer

conda create -n gamdformer python=3.9 -y
conda activate gamdformer

pip install -r requirements.txt
pip install -e .
```
Tested with Python 3.9, PyTorch ≥ 1.12, and CUDA 11.x.
---
Data Preparation
Each task expects paired degraded/clean images in parallel directories (matched by sorted filename):
```
data/
├── denoising/{train,val,test}/{input,target}/
├── lowlight/{train,val,test}/{input,target}/
└── deblur/{train,val,test}/{input,target}/
```
Update the paths in the corresponding YAML file under `configs/`.
---
Training
Single-task training:
```bash
python scripts/train.py --config configs/denoising_gaussian.yaml
python scripts/train.py --config configs/lowlight_enhancement.yaml
python scripts/train.py --config configs/motion_deblurring.yaml
```
Joint multi-task training (datasets concatenated, each tagged with its task id):
```bash
python scripts/train.py --config configs/multitask_joint.yaml
```
Resume from a checkpoint:
```bash
python scripts/train.py --config configs/denoising_gaussian.yaml \
    --resume experiments/gamdformer_denoising/latest.pth
```
---
Evaluation
```bash
python scripts/test.py \
    --config configs/denoising_gaussian.yaml \
    --weights experiments/gamdformer_denoising/best.pth \
    --save_dir results/denoising
```
Reports PSNR and SSIM over the test set and optionally writes restored images.
---
Inference
Restore a single image or a folder, selecting the task with `--task` (0 = denoising, 1 = low-light, 2 = deblurring):
```bash
python scripts/inference.py \
    --weights experiments/gamdformer_denoising/best.pth \
    --input demo/noisy.png \
    --output results/demo \
    --task 0
```
---
Results
Placeholder tables — populate with your trained checkpoints.
Gaussian Color Denoising (PSNR ↑, σ = 15 / 25 / 50)
Method	CBSD68	Kodak24	Urban100
Restormer	– / – / –	– / – / –	– / – / –
GAMDFormer	– / – / –	– / – / –	– / – / –
Low-Light Enhancement (LOL)
Method	PSNR ↑	SSIM ↑
Baseline	–	–
GAMDFormer	–	–
Single-Image Motion Deblurring (GoPro / HIDE)
Method	GoPro PSNR ↑	GoPro SSIM ↑	HIDE PSNR ↑
Baseline	–	–	–
GAMDFormer	–	–	–
---
Citation
If you find this work useful, please cite:
```bibtex
@article{gamdformer2026,
  title   = {GAMDFormer: A Gated Attention Modulation Transformer with
             Dual-Domain Interaction for Image Restoration},
  author  = {Das, Debashis and Maji, Suman Kumar},
  journal = {arXiv preprint},
  year    = {2026}
}
```
---
Acknowledgements
The hierarchical encoder–decoder backbone and channel-transposed attention
formulation are inspired by Restormer
(Zamir et al., CVPR 2022). The novel contributions of this repository — the
CAMG-gated attention and the dual-domain frequency feed-forward network — are
original to GAMDFormer.
License
Released under the MIT License.
