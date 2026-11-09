"""JEM: Scalable Patch-Level Self-Supervised Learning (Seitzer et al., arXiv 2610.10013).

Map from the paper to this file:
    Sec. 3.3 / App. A.3.2   view construction          -> ViewSets, block_mask, random_mask
    Sec. 3.2.1, Eq. (4)     patch-wise KL alignment    -> align_loss
    Sec. 3.2.2, Eqs. (5-6)  global collapse reg.       -> global_collapse_reg
    Sec. 3.2.3, Eqs. (7-10) structural collapse reg.   -> struct_reg
    App. A.3.3, Eq. (11)    full objective             -> jem_loss
    Tabs. 5-6               hyper-parameters           -> CFG
Adapted to ViT-B (supervised depths 4 / 8 / 12, R_struct targets scaled to 12 blocks) and one GPU (batch 128).
"""

import argparse
import copy
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from common import IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES, Logger, PlantNet, cosine, eval_transform
from vit import ViT

CFG = dict(
    patch=16, global_size=16, local_size=7,       # crops in patches: 256 px teacher/global, 112 px local
    n_local=4, n_view_sets=2,                     # 2 view sets x (1 teacher, 1 global, 4 local student views)
    teacher_scale=(0.32, 1.0), min_overlap=0.1,
    mask_prob=0.8, global_mask_ratio=0.65, local_mask_ratio=0.5,
    K=4096, tau=0.1,                              # categories, student = teacher temperature
    head_layers={4: 0.25, 8: 0.25, 12: 1.0},      # supervised depths -> KL weight (ViT-L: {8,16,24})
    struct_layers=(0, 4, 8),                      # R_struct teacher target blocks (ViT-L: {0,8,16})
    w_vis=0.2, w_msk=0.8,
    w_global=(1.0, 0.1), w_struct=(0.0, 0.05, 1.0), reg_warmup=12500,  # steps
    w_crop={"global": 0.5, "local": 0.5},         # ViT-L IN-1k setting
    base_lr=1e-3, lr_final_ratio=5.697e-6 / 1.697e-3, warmup=0.1, llrd=0.9,
    wd=(0.04, 0.4), betas=(0.9, 0.999), ema=(0.992, 1.0), clip_grad=3.0, drop_path=0.1,
    ls_init=1e-5,                                 # LayerScale init; None = no LayerScale (ablation)
    head_mode="inblock",                          # "inblock" = paper; "final" = shallow targets predicted from the final layer
)


# ---------------------------------------------------------------- view construction (Sec. 3.3, A.3.2)

def block_mask(n, ratio):
    """Multi-block masking (iBOT / DINOv2) followed by a random cyclic shift (CAPI)."""
    target, m = int(ratio * n * n), np.zeros((n, n), bool)
    for _ in range(100):
        remaining = target - m.sum()
        if remaining <= 0:
            break
        area = random.uniform(4, max(4, remaining))
        aspect = math.exp(random.uniform(math.log(0.3), math.log(1 / 0.3)))
        h, w = round(math.sqrt(area * aspect)), round(math.sqrt(area / aspect))
        if 0 < h < n and 0 < w < n:
            y, x = random.randint(0, n - h), random.randint(0, n - w)
            m[y:y + h, x:x + w] = True
    m = m.flatten()
    surplus = int(m.sum()) - target  # trim / top up to the exact ratio
    m[np.random.choice(np.flatnonzero(m if surplus > 0 else ~m), abs(surplus), replace=False)] = surplus < 0
    m = m.reshape(n, n)
    return np.roll(m, (random.randrange(n), random.randrange(n)), (0, 1)).flatten()


def random_mask(n, ratio):
    m = np.zeros(n * n, bool)
    m[np.random.choice(n * n, int(ratio * n * n), replace=False)] = True
    return m


class ViewSets:
    """Per image: 2 view sets, each a teacher crop, a shifted global student crop and 4 local student
    crops, all cut at patch-aligned positions from one resized super-crop (shared patch grid)."""

    def __init__(self, cfg=CFG):
        self.c = cfg
        self.color = v2.Compose([v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                                 v2.RandomGrayscale(p=0.2)])

    def photometric(self, crop, blur_p, solarize_p):  # BYOL / DINO augmentations, independent per crop
        crop = self.color(crop)
        if random.random() < blur_p:
            crop = crop.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 2.0)))
        if random.random() < solarize_p:
            crop = ImageOps.solarize(crop, 128)
        return torch.from_numpy(np.asarray(crop).copy()).permute(2, 0, 1)

    def view_set(self, img, swap):
        c, P, G, L = self.c, self.c["patch"], self.c["global_size"], self.c["local_size"]
        # teacher scale in [0.32, 1] of the image area, then a student-teacher shift (in patches) with
        # >= min_overlap overlap whose super-crop (side S patches, area scale * (S/G)^2) still fits in the image
        scale = random.uniform(*c["teacher_scale"])
        S_max = int(G / math.sqrt(scale))
        while True:
            dy, dx = random.randint(1 - G, G - 1), random.randint(1 - G, G - 1)
            if (G - abs(dy)) * (G - abs(dx)) >= c["min_overlap"] * G * G and G + max(abs(dy), abs(dx)) <= S_max:
                break
        S = G + max(abs(dy), abs(dx))  # square super-crop side, in patches
        ty = random.randint(max(0, -dy), S - G - max(0, dy))
        tx = random.randint(max(0, -dx), S - G - max(0, dx))
        # super-crop with random aspect ratio (re-drawn until it fits, as in RandomResizedCrop)
        W, H = img.size
        area = min(1.0, scale * (S / G) ** 2) * W * H
        for _ in range(10):
            ar = math.exp(random.uniform(math.log(3 / 4), math.log(4 / 3)))
            w, h = round(math.sqrt(area * ar)), round(math.sqrt(area / ar))
            if w <= W and h <= H:
                break
        w, h = min(W, w), min(H, h)
        x0, y0 = random.randint(0, W - w), random.randint(0, H - h)
        sup = img.resize((S * P, S * P), Image.BICUBIC, box=(x0, y0, x0 + w, y0 + h))
        if random.random() < 0.5:  # flip shared by all views of the set
            sup = sup.transpose(Image.FLIP_LEFT_RIGHT)
        crop = lambda y, x, n: sup.crop((x * P, y * P, (x + n) * P, (y + n) * P))
        (bt, st), (bs, ss) = ((1.0, 0.0), (0.1, 0.2))[::-1 if swap else 1]
        teacher = self.photometric(crop(ty, tx, G), bt, st)
        glob = self.photometric(crop(ty + dy, tx + dx, G), bs, ss)
        locs, loc_off = [], []
        for _ in range(c["n_local"]):
            while True:
                ly, lx = random.randint(0, S - L), random.randint(0, S - L)
                ov = max(0, min(ly + L, ty + G) - max(ly, ty)) * max(0, min(lx + L, tx + G) - max(lx, tx))
                if ov >= c["min_overlap"] * L * L:
                    break
            locs.append(self.photometric(crop(ly, lx, L), 0.5, 0.0))
            loc_off.append((ly - ty, lx - tx))
        g_mask = block_mask(G, c["global_mask_ratio"]) if random.random() < c["mask_prob"] else np.zeros(G * G, bool)
        l_mask = [random_mask(L, c["local_mask_ratio"]) if random.random() < c["mask_prob"] else np.zeros(L * L, bool)
                  for _ in locs]
        return teacher, glob, (dy, dx), g_mask, locs, loc_off, l_mask

    def __call__(self, img):
        sets = [self.view_set(img, swap=i % 2 == 1) for i in range(self.c["n_view_sets"])]
        return dict(
            teacher=torch.stack([s[0] for s in sets]), glob=torch.stack([s[1] for s in sets]),
            g_off=torch.tensor([s[2] for s in sets]), g_mask=torch.from_numpy(np.stack([s[3] for s in sets])),
            local=torch.stack([x for s in sets for x in s[4]]), l_off=torch.tensor([o for s in sets for o in s[5]]),
            l_mask=torch.from_numpy(np.stack([m for s in sets for m in s[6]])))


def overlap_index(off, n, n_t):
    """Student crop of n x n patches at offset `off` (V,2) inside the teacher grid (n_t x n_t):
    m(l) = flat teacher patch index for each student patch, -1 where the patch is outside the teacher view."""
    r = torch.arange(n, device=off.device)
    yy, xx = off[:, 0, None, None] + r[None, :, None], off[:, 1, None, None] + r[None, None, :]
    idx = yy * n_t + xx
    idx[(yy < 0) | (yy >= n_t) | (xx < 0) | (xx >= n_t)] = -1
    return idx.flatten(1)


# ---------------------------------------------------------------- model

def mlp(dim, hidden, out):
    return nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, out))


def trunc_normal_init(module):  # DINOv3 head init: trunc-normal(0.02) weights, zero biases
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class ProjHead(nn.Module):
    """Patch features -> K-way categorical logits, DINOv3 DINOHead: 3-layer MLP, L2-normalised 256-d
    bottleneck, plain bias-free linear layer to the K prototypes; divided by the temperature."""

    def __init__(self, dim, K, tau, norm):
        super().__init__()
        self.tau, self.norm = tau, nn.LayerNorm(dim) if norm else nn.Identity()
        self.mlp, self.prototypes = mlp(dim, 2048, 256), nn.Linear(256, K, bias=False)
        trunc_normal_init(self)

    def forward(self, x):
        return self.prototypes(F.normalize(self.mlp(self.norm(x)), dim=-1)) / self.tau


class JEMNet(nn.Module):
    def __init__(self, cfg=CFG):
        super().__init__()
        self.backbone = ViT(drop_path=cfg["drop_path"], ls_init=cfg["ls_init"])
        last = len(self.backbone.blocks)
        # one projection head per supervised depth (intermediate depths get their own LayerNorm)
        self.heads = nn.ModuleDict({str(k): ProjHead(768, cfg["K"], cfg["tau"], norm=k != last)
                                    for k in cfg["head_layers"]})
        # g_k of R_struct: same MLP as the projection head but 1024-d output, no prototypes
        self.struct = nn.ModuleDict({str(k): mlp(768, 2048, 1024) for k in cfg["struct_layers"]})
        trunc_normal_init(self.struct)
        self.last = last
        # head_mode "final": the final student layer predicts the teacher's shallow-depth distributions
        # through its own heads (final features are already LayerNormed, like the final head's input)
        if cfg["head_mode"] == "final":
            self.pred = nn.ModuleDict({str(k): ProjHead(768, cfg["K"], cfg["tau"], norm=False)
                                       for k in cfg["head_layers"] if k != last})

    def logits(self, feats):
        return {k: self.heads[str(k)](feats[k]).float() for k in CFG["head_layers"]}


# ---------------------------------------------------------------- losses

def align_loss(logp_t, logq_s, valid, mask):
    """Eq. (4): KL(p(T^m(l) | x_T) || q(T^l | x_S)) averaged over overlapping patches, split into the
    visible and masked student patches (App. A.3.3). logp_t is already gathered at m(l)."""
    kl = (logp_t.exp() * (logp_t - logq_s)).sum(-1)  # (views, patches)

    def per_view_mean(sel):  # 1/|O_S| per student-teacher pair, then mean over the views that have such patches
        n = sel.sum(1)
        return ((kl * sel).sum(1) / n.clamp_min(1) * (n > 0)).sum() / (n > 0).sum().clamp_min(1)
    return per_view_mean(valid & ~mask), per_view_mean(valid & mask)


def patch_mutual_info(logq):
    """Eq. (5) for every patch position: H(mean_batch q) - mean_batch H(q). logq: (N samples, positions, K)."""
    q = logq.exp()
    q_bar = q.mean(0)
    h_marg = -(q_bar * q_bar.clamp_min(1e-12).log()).sum(-1)
    h_cond = -(q * logq).sum(-1).mean(0)
    return h_marg - h_cond, h_marg, h_cond


def global_collapse_reg(logq):
    """Eq. (6): R_global = - mean over elements (patch position) of the estimated mutual information."""
    mi, h_marg, h_cond = patch_mutual_info(logq)
    return -mi.mean(), h_marg.mean(), h_cond.mean()


def struct_reg(g_r, h_t, idx, t_view):
    """Eqs. (7)-(10) for one target layer k: squared difference between the cosine similarity of every
    pair of overlapping student patches after g_k and the cosine similarity of the corresponding teacher
    patches at layer k (parameter-free LayerNorm), averaged over pairs i != j per view."""
    s = F.normalize(g_r.float(), dim=-1)
    h = F.normalize(F.layer_norm(h_t.float(), h_t.shape[-1:]), dim=-1)[t_view[:, None], idx.clamp_min(0)]
    valid = idx >= 0
    pairs = valid[:, :, None] & valid[:, None, :] & ~torch.eye(idx.shape[1], dtype=torch.bool, device=idx.device)
    err = ((s @ s.transpose(1, 2) - h @ h.transpose(1, 2)).square() * pairs).sum((1, 2))
    n = pairs.sum((1, 2))
    return (err / n.clamp_min(1) * (n > 0)).sum() / (n > 0).sum().clamp_min(1)  # fixed-shape masked mean


def jem_loss(net, s_feats, mask, idx, t_view, t_logp, t_feats, w_global, w_struct):
    """Eq. (11) for one crop type: w_vis L_vis + w_msk L_msk + w_global R_global + w_struct R_struct.

    head_mode "final" (not in the paper): for every shallow depth k, the backbone is trained only through
    a predictor head on the *final* student layer that matches the teacher's depth-k distribution. The
    in-block head k still runs on the student's block-k features, but on detached ones: its alignment and
    R_global terms train the head alone (it is what the EMA teacher uses to make the depth-k targets), and
    no head pulls on intermediate blocks any more (L_head below has no gradient path into the backbone)."""
    final_mode = CFG["head_mode"] == "final"
    valid = idx >= 0
    l_vis = l_msk = r_global = l_head = 0.0
    stats = {}
    for k, w_layer in CFG["head_layers"].items():
        t_k = t_logp[k][t_view[:, None], idx.clamp_min(0)]
        shallow = final_mode and k != net.last
        feats = s_feats[k].detach() if shallow else s_feats[k]
        logq = F.log_softmax(net.heads[str(k)](feats).float(), -1)
        lv, lm = align_loss(t_k, logq, valid, mask)
        rg, hm, hc = global_collapse_reg(logq)
        r_global = r_global + rg / len(CFG["head_layers"])
        stats[f"H_marg_{k}"], stats[f"H_cond_{k}"] = hm, hc
        if shallow:  # train the in-block head only, then align the backbone through the final-layer predictor
            l_head = l_head + w_layer * (CFG["w_vis"] * lv + CFG["w_msk"] * lm)
            stats[f"kl_vis_{k}_head"], stats[f"kl_msk_{k}_head"] = lv, lm
            logq = F.log_softmax(net.pred[str(k)](s_feats[net.last]).float(), -1)
            lv, lm = align_loss(t_k, logq, valid, mask)
        l_vis, l_msk = l_vis + w_layer * lv, l_msk + w_layer * lm
        stats[f"kl_vis_{k}"], stats[f"kl_msk_{k}"] = lv, lm  # the KLs the backbone is trained on
    r_struct = sum(struct_reg(net.struct[str(k)](s_feats[net.last]), t_feats[k], idx, t_view)
                   for k in CFG["struct_layers"]) / len(CFG["struct_layers"])
    loss = CFG["w_vis"] * l_vis + CFG["w_msk"] * l_msk + w_global * r_global + w_struct * r_struct + l_head
    stats.update(loss=loss, L_vis=l_vis, L_msk=l_msk, R_global=r_global, R_struct=r_struct, L_head=l_head)
    return loss, stats


def reg_weights(step, total):
    """w_global: 1 -> 0.1 linearly over the first 12.5k steps; w_struct: 0 -> 0.05 over the same
    warm-up, then linearly 0.05 -> 1 until the end (Tab. 6: 12.5k steps in every run, 125k to 500k)."""
    warm = CFG["reg_warmup"]
    (g0, g1), (s0, s1, s2) = CFG["w_global"], CFG["w_struct"]
    if step < warm:
        return g0 + (g1 - g0) * step / warm, s0 + (s1 - s0) * step / warm
    return g1, s1 + (s2 - s1) * (step - warm) / max(1, total - warm)


# ---------------------------------------------------------------- training

def param_groups(net):
    """AdamW groups with layer-wise LR decay; no weight decay on biases, norms, LayerScale and tokens."""
    depth, groups = len(net.backbone.blocks), {}
    for name, p in net.named_parameters():
        if name.startswith("backbone.blocks."):
            layer = int(name.split(".")[2]) + 1
        elif name.startswith("backbone.") and not name.startswith("backbone.norm"):
            layer = 0
        else:
            layer = depth + 1
        decay = p.ndim > 1
        key = (layer, decay)
        groups.setdefault(key, {"params": [], "lr_scale": CFG["llrd"] ** (depth + 1 - layer), "decay": decay})
        groups[key]["params"].append(p)
    return list(groups.values())


@torch.no_grad()
def evaluate(backbone, probe, loader):
    backbone.eval()
    correct = n = 0
    for x, y in loader:
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            correct += (probe(backbone(x)["final"].mean(1)).argmax(1) == y).sum().item()
        n += len(y)
    return correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--out", default="runs/jem")
    ap.add_argument("--snapshot_every", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=5, help="online-probe val evaluation every N epochs")
    ap.add_argument("--max_steps", type=int, default=0, help="debug: stop after this many steps")
    ap.add_argument("--stop_epoch", type=int, default=0, help="stop after N epochs, keeping the --epochs schedules")
    ap.add_argument("--no_layerscale", action="store_true", help="ablation: plain residual branches")
    ap.add_argument("--head_mode", choices=("inblock", "final"), default="inblock",
                    help="inblock = paper (student block k predicts teacher block k); final = the final student "
                         "layer predicts the teacher's shallow-depth targets, in-block heads train on detached features")
    args = ap.parse_args()
    if args.no_layerscale:
        CFG["ls_init"] = None
    CFG["head_mode"] = args.head_mode
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True

    train = DataLoader(PlantNet("train", ViewSets()), batch_size=args.bs, shuffle=True, drop_last=True,
                       num_workers=args.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    val = DataLoader(PlantNet("val", eval_transform()), batch_size=256, num_workers=8)

    student = JEMNet().cuda()
    teacher = copy.deepcopy(student).requires_grad_(False)
    groups = param_groups(student)
    opt = torch.optim.AdamW(groups, betas=CFG["betas"], fused=True)
    probe = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, NUM_CLASSES)).cuda()  # online linear probe on the teacher, for monitoring
    probe_opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-7)
    s_backbone, t_backbone = torch.compile(student.backbone), torch.compile(teacher.backbone)
    loss_fn = torch.compile(jem_loss)  # fuses the 4096-way softmax / KL / entropy terms

    total = args.epochs * len(train)
    peak_lr = CFG["base_lr"] * math.sqrt(args.bs / 64)
    start_epoch, step = 0, 0
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location="cpu")  # on CPU: a GPU copy would stay resident and OOM an H100
        for m, k in [(student, "student"), (teacher, "teacher"), (opt, "opt"), (probe, "probe"), (probe_opt, "probe_opt")]:
            m.load_state_dict(ck[k])
        groups = opt.param_groups  # load_state_dict replaces the group dicts: schedule the live ones
        start_epoch, step = ck["epoch"] + 1, ck["step"]
        del ck
    log = Logger(out / "log.jsonl")
    s_params, t_params = list(student.parameters()), list(teacher.parameters())
    mean = torch.tensor(IMAGENET_MEAN, device="cuda").view(1, 3, 1, 1) * 255
    std = torch.tensor(IMAGENET_STD, device="cuda").view(1, 3, 1, 1) * 255
    norm = lambda x: (x.cuda(non_blocking=True).flatten(0, 1).float() - mean) / std
    G, L = CFG["global_size"], CFG["local_size"]
    nv, nl = CFG["n_view_sets"], CFG["n_view_sets"] * CFG["n_local"]
    t_view_g = torch.arange(args.bs * nv, device="cuda")                     # global view i <-> teacher view i
    t_view_l = (torch.arange(args.bs * nl, device="cuda") // CFG["n_local"])  # 4 locals per view set

    for epoch in range(start_epoch, args.epochs):
        student.train(), teacher.eval()
        t0 = time.time()
        for it, (b, y) in enumerate(train):
            # schedules (Tabs. 5-6): warm-up + cosine LR, cosine WD, cosine EMA momentum, loss weights
            warm = CFG["warmup"] * total
            lr = peak_lr * (step / warm if step < warm else cosine(step - warm, total - warm, 1.0, CFG["lr_final_ratio"]))
            wd = cosine(step, total, *CFG["wd"])
            mom = cosine(step, total, *CFG["ema"])
            w_global, w_struct = reg_weights(step, total)
            for g in groups:
                g["lr"], g["weight_decay"] = lr * g["lr_scale"], wd if g["decay"] else 0.0

            T, Xg, Xl = norm(b["teacher"]), norm(b["glob"]), norm(b["local"])
            g_mask, l_mask = b["g_mask"].cuda().flatten(0, 1), b["l_mask"].cuda().flatten(0, 1)
            idx_g = overlap_index(b["g_off"].cuda().flatten(0, 1), G, G)
            idx_l = overlap_index(b["l_off"].cuda().flatten(0, 1), L, G)
            y = y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():  # teacher: full (unmasked) teacher crops
                    t_out = t_backbone(T, None, (0, 4, 8))
                    t_out[12] = t_out["final"]  # final-layer head acts on the final-LayerNorm output
                    t_logp = {k: F.log_softmax(v, -1) for k, v in teacher.logits(t_out).items()}
                s_g = s_backbone(Xg, g_mask, (4, 8))
                s_g[12] = s_g["final"]
                s_l = s_backbone(Xl, l_mask, (4, 8))
                s_l[12] = s_l["final"]
                w_reg = [torch.tensor(w, device="cuda") for w in (w_global, w_struct)]  # tensors: no recompiles
                loss_g, st_g = loss_fn(student, s_g, g_mask, idx_g, t_view_g, t_logp, t_out, *w_reg)
                loss_l, st_l = loss_fn(student, s_l, l_mask, idx_l, t_view_l, t_logp, t_out, *w_reg)
                loss = CFG["w_crop"]["global"] * loss_g + CFG["w_crop"]["local"] * loss_l
                probe_loss = F.cross_entropy(probe(t_out["final"].mean(1).float()), y.repeat_interleave(nv))
            opt.zero_grad(set_to_none=True)
            probe_opt.zero_grad(set_to_none=True)
            (loss + probe_loss).backward()
            gnorm = torch.nn.utils.clip_grad_norm_(s_params, CFG["clip_grad"])
            opt.step()
            probe_opt.step()
            with torch.no_grad():  # teacher <- EMA(student)
                torch._foreach_lerp_(t_params, s_params, 1 - mom)
            step += 1

            if step % 50 == 0:
                with torch.no_grad():  # Fig. 5 diagnostics on the teacher: patch MI, l2 distance to mean patch
                    t_mi = patch_mutual_info(t_logp[12])[0].mean()
                    f = t_out["final"].float()
                    l2_mean = (f - f.mean(1, keepdim=True)).norm(dim=-1).mean()
                rec = {"step": step, "epoch": epoch, "loss": loss.item(), "probe_loss": probe_loss.item(),
                       "lr": lr, "opt_lr": opt.param_groups[-1]["lr"] / opt.param_groups[-1]["lr_scale"],  # live value
                       "wd": wd, "momentum": mom, "w_global": w_global, "w_struct": w_struct,
                       "grad_norm": gnorm.item(), "teacher_patch_MI": t_mi.item(), "teacher_l2_to_mean": l2_mean.item(),
                       "img_per_s": 50 * args.bs / (time.time() - t0), "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
                rec.update({f"g/{k}": float(v) for k, v in st_g.items()})
                rec.update({f"l/{k}": float(v) for k, v in st_l.items()})
                log.log(rec)
                print(f"ep {epoch} step {step}/{total} loss {loss.item():.4f} Rg {st_g['R_global'].item():.3f} "
                      f"Rs {st_g['R_struct'].item():.4f} MI {t_mi.item():.2f} {rec['img_per_s']:.0f} img/s", flush=True)
                t0 = time.time()
            if args.max_steps and step >= args.max_steps:  # debug: short run that still evaluates + saves
                break

        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs or args.max_steps:
            acc = evaluate(t_backbone, probe, val)
            log.log({"step": step, "epoch": epoch, "val_online_probe_acc": acc})
            print(f"epoch {epoch} online probe val acc {acc:.4f}", flush=True)
        ck = dict(student=student.state_dict(), teacher=teacher.state_dict(), opt=opt.state_dict(),
                  probe=probe.state_dict(), probe_opt=probe_opt.state_dict(), epoch=epoch, step=step)
        torch.save(ck, out / "last.pt.tmp")
        (out / "last.pt.tmp").rename(out / "last.pt")
        if (epoch + 1) % args.snapshot_every == 0 or epoch + 1 == args.epochs:
            torch.save(teacher.backbone.state_dict(), out / f"teacher_backbone_ep{epoch + 1:03d}.pt")
        if args.max_steps or (args.stop_epoch and epoch + 1 >= args.stop_epoch):
            return


if __name__ == "__main__":
    main()
