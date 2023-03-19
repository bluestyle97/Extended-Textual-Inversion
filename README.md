# Extended Textual Inversion

This repo provides an unofficial implementation of the paper [$P+$: Extended Textual Conditioning in Text-to-Image Generation](https://arxiv.org/abs/2303.09522) (XTI).

## Setup

This repo shares the same requirements with [Stable Diffusion](https://github.com/CompVis/stable-diffusion). To set up their environment, please run:

```
conda env create -f environment.yaml
conda activate ldm
```

You also need to download the official Stable Diffusion checkpoint, available at [Huggingface](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original). Other versions of checkpoints (e.g., sd v1.5) should also work, but I haven't try.

## Usage

### Extended Textural Inversion

```
CUDA_VISIBLE_DEVICES=0,1 python main.py --base configs/stable-diffusion/v1-finetune_xti.yaml \
                          -t \
                          --no-test True \
                          --actual_resume /path/to/sd-v1-4-full-ema.ckpt \
                          -n "cat" \
                          --gpus 0,1 \
                          --logdir logs \
                          --data_root datasets/textual_inversion/cat_statue \
                          --init_word "cat"
```

### Generation

```
CUDA_VISIBLE_DEVICES=0 python scripts/stable_txt2img.py --ddim_eta 0.0 \
                          --n_samples 8 \
                          --n_iter 1 \
                          --scale 10.0 \
                          --ddim_steps 50 \
                          --outdir outputs/txt2img-samples \
                          --config configs/stable-diffusion/v1-finetune_xti.yaml \
                          --embedding_path /path/to/embeddings.pt \
                          --ckpt /path/to/sd-v1-4-full-ema.ckpt \
                          --prompt "* standing in Times Square"
```

### Style Mixing

```
CUDA_VISIBLE_DEVICES=0 python scripts/stable_txt2img.py --ddim_eta 0.0 \
                          --n_samples 8 \
                          --n_iter 1 \
                          --scale 10.0 \
                          --ddim_steps 50 \
                          --outdir outputs/txt2img-samples \
                          --config configs/stable-diffusion/v1-finetune_xti.yaml \
                          --style_mixing \
                          --mixing_layers_range "5,8" \
                          --shape_embedding_path /path/to/shape/embeddings.pt \
                          --appearance_embedding_path /path/to/appearance/embeddings.pt \
                          --ckpt /path/to/sd-v1-4-full-ema.ckpt \
                          --prompt "*"
```
The `mixing_layers_range` argument defines the range of cross-attention layers that use shape embeddings as described in the paper. `"5,8"` means that the 5th, 6th and 7th layers will use shape embeddings as conditions, while the other layers use appearance embeddings as conditions. Since there are 16 cross-attention layers in Stable Diffusion's UNet in total, this range should be within `[0,16)`.

## Acknowledgements & Tips
- This repo is based on the official repos of [Stable Diffusion](https://github.com/CompVis/stable-diffusion) and [Textual Inversion](https://github.com/rinongal/textual_inversion).


## Citation

Please cite the original paper:

```
@article{xu2022dream3d,
  author    = {Voynov, Andrey and Chu, Qinghao and Cohen-Or, Daniel and Cao, Yan-Pei and Aberman, Kfir},
  title     = {$P+$: Extended Textual Conditioning in Text-to-Image Generation},
  journal   = {arXiv preprint arXiv:2303.09522},
  year      = {2023},
}
```