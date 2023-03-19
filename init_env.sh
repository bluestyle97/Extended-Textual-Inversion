pip install pytorch-lightning==1.5.9 kornia==0.6 albumentations diffusers pudb invisible-watermark imageio-ffmpeg omegaconf test-tube streamlit einops torch-fidelity transformers torchmetrics
export PYTHONPATH="${PYTHONPATH}:/group/30042/jialexu/projects/SDF-StyleGAN/text2image/taming-transformers"
pip install -e .
mkdir -p /usr/local/app/.cache/torch/hub/checkpoints/
cp checkpoint_liberty_with_aug.pth /usr/local/app/.cache/torch/hub/checkpoints/
