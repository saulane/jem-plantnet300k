# JEM ViT-B/16 on Pl@ntNet-300K-v2

A reproduction of **JEM** (*Scalable Patch-Level Self-Supervised Learning*, arXiv 2610.10013), training a ViT-B/16 from scratch on Pl@ntNet-300K-v2 (243,866 training images, 1,000 plant species, long-tailed). Two versions are released:
- **JEM as in the paper**, with LayerScale (init 1e-5);
- **JEM without LayerScale**, an ablation that changes nothing else.

## Weights

Both 100-epoch backbones (the EMA teacher) are on the Hugging Face Hub:
**[saulane/jem-vitb16-plantnet300kv2](https://huggingface.co/saulane/jem-vitb16-plantnet300kv2)** (CC-BY-4.0).

| File | Version | Linear probe, test top-1 | Rare species (< 20 train images), macro top-1 |
|---|---|---|---|
| `jem_nols_vitb16_plantnet300kv2_ep100.safetensors` | JEM without LayerScale | **83.7** | **17.9** |
| `jem_vitb16_plantnet300kv2_ep100.safetensors` | JEM as in the paper (LayerScale) | 82.4 | 14.5 |

These numbers come from a linear probe on the frozen mean patch token, with one run per version and one seed.

```python
from huggingface_hub import hf_hub_download
from load import load_jem

model = load_jem(hf_hub_download("saulane/jem-vitb16-plantnet300kv2", "jem_nols_vitb16_plantnet300kv2_ep100.safetensors"))
out = model(images, layers=(4, 8))  # images: (B, 3, H, W), ImageNet mean/std, H and W divisible by 16
feats = out["final"].mean(1)        # (B, 768) mean patch token
```

`load_jem` detects from the weights whether the checkpoint has LayerScale. The evaluation transform is `common.eval_transform`: a square bicubic resize to 224 × 224.

## Training

```bash
uv sync
export PLANTNET_DIR=/path/to/plantnet300kv2   # images/{train,val,test}/<species_id>/*.jpg
uv run python jem.py --epochs 100 --out runs/jem                           # as in the paper
uv run python jem.py --epochs 100 --no_layerscale --out runs/jem_nols      # without LayerScale
```

- **Requirements:** one 80 GB GPU. A run takes about 22 h on an H100.
- **Outputs:** `teacher_backbone_epXXX.pt` every 10 epochs, `last.pt` to resume from, and `log.jsonl`. An online linear probe reports val accuracy every 5 epochs.
- **Data:** Pl@ntNet-300K-v2 is on Zenodo (record 10419064).

| File | What |
|---|---|
| `jem.py` | JEM: view construction, Eq. (4) alignment, Eqs. (5–6) `R_global`, Eqs. (7–10) `R_struct`, Eq. (11), training loop |
| `vit.py` | the ViT-B/16 backbone: no CLS token, 4 registers, axial RoPE, optional LayerScale, mask token, multi-depth outputs |
| `common.py` | dataset, evaluation transform, logger |
| `load.py` | loads a released backbone (`.safetensors` or `.pt`) |

**Changes from the paper:**
- the paper uses ViT-L (24 blocks), so the supervised depths and `R_struct` targets are scaled to 12 blocks;
- training uses batch 128 on one GPU instead of 2048.

## License

Code: [MIT](LICENSE). Weights: [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/), the license of Pl@ntNet-300K.
