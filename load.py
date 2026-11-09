"""Load a released JEM ViT-B/16 backbone (Pl@ntNet-300K-v2, 100 epochs) and extract features.

    from load import load_jem
    model = load_jem("weights/jem_nols_vitb16_plantnet300kv2_ep100.safetensors")
    feats = model(images)["final"].mean(1)  # (B, 768) mean patch token

The checkpoint is the EMA teacher backbone. Both versions are detected from the weights:
with LayerScale (as in the paper) or without it (the ablation).
Inputs: RGB, ImageNet mean/std, any size divisible by 16 (trained at 224; see common.eval_transform).
"""

import sys

import torch
from safetensors.torch import load_file

from vit import ViT


def load_jem(path, device="cpu"):
    sd = load_file(path) if str(path).endswith(".safetensors") else torch.load(path, map_location="cpu")
    model = ViT(drop_path=0.0, ls_init=1e-5 if "blocks.0.ls1" in sd else None)
    model.load_state_dict(sd)
    return model.eval().to(device)


if __name__ == "__main__":  # python load.py weights/<file>.safetensors
    model = load_jem(sys.argv[1])
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224), layers=(4, 8))
    print({k: tuple(v.shape) for k, v in out.items()})
