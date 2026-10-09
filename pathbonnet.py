#!/usr/bin/env python3
"""
PathBonnet - Bonnet (arXiv 2601.22576) adapted from CT bone to digital pathology.

Bonnet (CT)                                   ->  PathBonnet (H&E histology)
------------------------------------------------------------------------------------------
HU threshold [200,3000] -> sparse voxels      ->  stain/OD prior threshold -> sparse pixels
                                                  (threshold auto-calibrated on TRAIN data to
                                                  keep >= target_recall of nuclear pixels)
z-score features (mu, sigma)                  ->  z-scored RGB + prior score per kept pixel
128^3 sparse windows, rho=0.33 fg-biased crop ->  128x128 sparse windows, rho=0.33 nuclei-biased crop
spconv 3D U-Net (SubMConv / SparseConv /      ->  spconv 2D U-Net (same ops, SparseInstanceNorm,
  InverseConv), SparseInstanceNorm, LReLU(.01)    LeakyReLU(0.01), MLP head)
sliding window overlap 0.5 + Gaussian fusion  ->  same (sigma = 0.5, Eq. 1-3 of the paper)
CE (label smoothing) + Soft Dice              ->  same (Eq. 5-7)
train TotalSegmentator, test RibSeg/Pelvic1K  ->  train PanNuke folds, test PanNuke fold 3 (internal)
  /Spine1K without fine-tuning                    and CoNSeP / your own folder (external, no fine-tuning)
nnU-Net (RAW) = no HU masking                 ->  --prior raw   (no mask)   |  --model dense (dense U-Net)

Usage
  python pathbonnet.py train   --pannuke_root DATA/PanNuke --prior hema --model sparse --out runs/hema_sparse
  python pathbonnet.py eval    --ckpt runs/hema_sparse/best.pt --pannuke_root DATA/PanNuke \
                               --consep_root DATA/CoNSeP --out runs/hema_sparse
  python pathbonnet.py predict --ckpt runs/hema_sparse/best.pt --images my_samples/ --out preds/
  python pathbonnet.py table   runs/*/results.json
"""
import argparse, glob, json, math, os, time
import numpy as np

# =====================================================================================
# 1. PRIOR (analogue of the HU bone threshold) + sparsification  -- numpy only
# =====================================================================================
# Ruifrok & Johnston H&E stain vectors (rows: hematoxylin, eosin, residual)
_H = np.array([0.650, 0.704, 0.286]); _E = np.array([0.072, 0.990, 0.105])
_R = np.cross(_H, _E)
_M = np.stack([_H / np.linalg.norm(_H), _E / np.linalg.norm(_E), _R / np.linalg.norm(_R)])
_MINV = np.linalg.inv(_M)


def prior_score(img, kind):
    """Per-pixel scalar 'nuclear prior'. tissue = mean optical density; hema = hematoxylin concentration."""
    od = -np.log((img.astype(np.float32) + 1.0) / 256.0)
    if kind == "tissue":
        return od.mean(-1).astype(np.float32)
    if kind == "hema":
        c = od.reshape(-1, 3) @ _MINV
        return c[:, 0].reshape(img.shape[:2]).astype(np.float32)
    raise ValueError(kind)


def to_sparse(img, cfg, lab=None):
    """Threshold -> sparse (coords, feats, labels). Returns ys, xs, feats, labels, mask."""
    kind = "tissue" if cfg["prior"] == "raw" else cfg["prior"]
    score = prior_score(img, kind)
    mask = np.ones(img.shape[:2], bool) if cfg["prior"] == "raw" else score > cfg["tau"]
    ys, xs = np.nonzero(mask)
    rgb = (img[ys, xs].astype(np.float32) / 255.0 - np.array(cfg["mean"], np.float32)) / np.array(cfg["std"], np.float32)
    sc = ((score[ys, xs] - cfg["smean"]) / cfg["sstd"])[:, None].astype(np.float32)
    feats = np.concatenate([rgb, sc], 1).astype(np.float32)
    return ys, xs, feats, (lab[ys, xs] if lab is not None else None), mask


def dense_input(img, cfg):
    x = (img.astype(np.float32) / 255.0 - np.array(cfg["mean"], np.float32)) / np.array(cfg["std"], np.float32)
    return x.transpose(2, 0, 1).astype(np.float32)


def calibrate(src, prior, target_recall=0.98, n=400, seed=0):
    """Pick tau on TRAIN data so the mask keeps `target_recall` of nuclear pixels (like choosing HU>=200 for bone)."""
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(src), min(n, len(src)), replace=False)
    kind = "tissue" if prior == "raw" else prior
    imgs, scores, fg = [], [], []
    for i in ids:
        img, lab = src.get(int(i))
        s = prior_score(img, kind)
        imgs.append(img); scores.append(s); fg.append(s[lab > 0])
    tau = -np.inf if prior == "raw" else float(np.quantile(np.concatenate(fg), 1.0 - target_recall))
    s1 = np.zeros(3); s2 = np.zeros(3); c1 = c2 = 0.0; cnt = 0; kept = 0; tot = 0
    for img, s in zip(imgs, scores):
        m = s > tau
        x = img[m].astype(np.float64) / 255.0
        s1 += x.sum(0); s2 += (x ** 2).sum(0); c1 += s[m].sum(); c2 += (s[m].astype(np.float64) ** 2).sum()
        cnt += m.sum(); kept += m.sum(); tot += m.size
    cnt = max(cnt, 1)
    mean = s1 / cnt; std = np.sqrt(np.maximum(s2 / cnt - mean ** 2, 1e-8))
    smean = c1 / cnt; sstd = math.sqrt(max(c2 / cnt - smean ** 2, 1e-8))
    return dict(prior=prior, tau=tau, mean=mean.tolist(), std=std.tolist(), smean=float(smean), sstd=float(sstd),
                target_recall=target_recall, retained_train=kept / tot)


# =====================================================================================
# 2. Sliding-window geometry, Gaussian fusion weights, metrics -- numpy only
# =====================================================================================
def gauss_window(win, sigma=0.5):
    """a_t(x) = exp(-||x - m_t||^2 / (2 (sigma*w)^2))   (Eq. 3)"""
    c = (win - 1) / 2.0
    yy, xx = np.mgrid[:win, :win]
    s = sigma * win
    return np.exp(-((yy - c) ** 2 + (xx - c) ** 2) / (2 * s * s)).astype(np.float32)


def window_origins(n, win, stride):
    o = list(range(0, n - win + 1, stride))
    if o[-1] != n - win:
        o.append(n - win)
    return o


def conf_matrix(pred, gt, K):
    return np.bincount(gt.ravel().astype(np.int64) * K + pred.ravel().astype(np.int64), minlength=K * K).reshape(K, K)


def dice_from_conf(conf):
    tp = np.diag(conf).astype(np.float64); fp = conf.sum(0) - tp; fn = conf.sum(1) - tp
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)


# =====================================================================================
# 3. Data -- PanNuke, CoNSeP, generic folder
# =====================================================================================
PANNUKE_CLASSES = ["background", "neoplastic", "inflammatory", "connective", "dead", "epithelial"]


class PanNukeFold:
    """images.npy (N,256,256,3), masks.npy (N,256,256,6: 5 nuclei types + background)."""
    def __init__(self, img_path, mask_path):
        self.ip, self.mp = img_path, mask_path; self._i = self._m = None
        self.n = np.load(img_path, mmap_mode="r").shape[0]

    def __len__(self): return self.n

    def get(self, i):
        if self._i is None:
            self._i = np.load(self.ip, mmap_mode="r"); self._m = np.load(self.mp, mmap_mode="r")
        img = np.asarray(self._i[i]); m = np.asarray(self._m[i])
        if img.max() <= 1.0: img = img * 255.0
        lab = np.zeros(m.shape[:2], np.uint8)
        for c in range(5):
            lab[m[..., c] > 0] = c + 1
        return np.clip(img, 0, 255).astype(np.uint8), lab


def find_pannuke(root):
    ims = sorted(glob.glob(os.path.join(root, "**", "images.npy"), recursive=True))
    mks = sorted(glob.glob(os.path.join(root, "**", "masks.npy"), recursive=True))
    assert len(ims) == 3 and len(mks) == 3, f"Expected 3 folds under {root}, found {len(ims)} images.npy / {len(mks)} masks.npy"
    print("PanNuke folds (train/val/test):"); [print("  ", a, "|", b) for a, b in zip(ims, mks)]
    return [PanNukeFold(a, b) for a, b in zip(ims, mks)]


class ListSet:
    """Generic binary-nuclei set: list of (image_path, mask_path_or_None, kind)."""
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def get(self, i):
        from PIL import Image
        ip, mp, kind = self.items[i]
        img = np.array(Image.open(ip).convert("RGB"))
        if mp is None: return img, np.zeros(img.shape[:2], np.uint8)
        if kind == "mat":
            from scipy.io import loadmat
            lab = (loadmat(mp)["inst_map"] > 0).astype(np.uint8)
        else:
            m = np.array(Image.open(mp)); m = m.max(-1) if m.ndim == 3 else m
            lab = (m > 0).astype(np.uint8)
        return img, lab


def consep_set(root, split="Test"):
    ims = sorted(glob.glob(os.path.join(root, split, "Images", "*.png")))
    assert ims, f"No CoNSeP images in {root}/{split}/Images"
    return ListSet([(p, os.path.join(root, split, "Labels", os.path.splitext(os.path.basename(p))[0] + ".mat"), "mat") for p in ims])


def folder_set(root):
    exts = ("png", "jpg", "jpeg", "tif", "tiff"); items = []
    for p in sorted(sum([glob.glob(os.path.join(root, "images", f"*.{e}")) for e in exts], [])):
        stem = os.path.splitext(os.path.basename(p))[0]
        mp = next(iter(glob.glob(os.path.join(root, "masks", stem + ".*"))), None)
        items.append((p, mp, "img"))
    assert items, f"No images in {root}/images"
    return ListSet(items)


# =====================================================================================
# ==== TORCH ==== (everything below needs torch; sparse model also needs spconv, CUDA GPU)
# =====================================================================================
import torch, torch.nn as nn, torch.nn.functional as F
try:
    import spconv.pytorch as spconv
except ImportError:  # dense baseline still works
    spconv = None


class SparseInstanceNorm(nn.Module):
    """Per-sample, per-channel normalisation over the active sites only."""
    def __init__(self, c, eps=1e-5):
        super().__init__(); self.eps = eps
        self.w = nn.Parameter(torch.ones(c)); self.b = nn.Parameter(torch.zeros(c))

    def forward(self, x):
        f = x.features; b = x.indices[:, 0].long(); B = x.batch_size
        cnt = torch.bincount(b, minlength=B).clamp(min=1).to(f.dtype).unsqueeze(1)
        mean = torch.zeros(B, f.shape[1], device=f.device, dtype=f.dtype).index_add_(0, b, f) / cnt
        d = f - mean[b]
        var = torch.zeros_like(mean).index_add_(0, b, d * d) / cnt
        return x.replace_feature(d / torch.sqrt(var[b] + self.eps) * self.w + self.b)


class SpBlock(nn.Module):
    def __init__(self, conv, cout):
        super().__init__(); self.conv = conv; self.norm = SparseInstanceNorm(cout)

    def forward(self, x):
        x = self.norm(self.conv(x))
        return x.replace_feature(F.leaky_relu(x.features, 0.01))


class SparseUNet(nn.Module):
    def __init__(self, in_ch, n_cls, ch=(32, 64, 128, 256)):
        super().__init__()
        assert spconv is not None, "spconv not installed (pip install spconv-cu120 or matching your CUDA)"
        L = len(ch); self.L = L
        self.enc, self.down, self.up, self.dec = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i, c in enumerate(ch):
            cin = in_ch if i == 0 else c; k = f"subm{i}"
            self.enc.append(nn.ModuleList([
                SpBlock(spconv.SubMConv2d(cin, c, 3, padding=1, bias=False, indice_key=k), c),
                SpBlock(spconv.SubMConv2d(c, c, 3, padding=1, bias=False, indice_key=k), c)]))
            if i < L - 1:
                self.down.append(SpBlock(spconv.SparseConv2d(c, ch[i + 1], 2, 2, bias=False, indice_key=f"down{i}"), ch[i + 1]))
        for i in range(L - 2, -1, -1):
            self.up.append(SpBlock(spconv.SparseInverseConv2d(ch[i + 1], ch[i], 2, indice_key=f"down{i}", bias=False), ch[i]))
            self.dec.append(SpBlock(spconv.SubMConv2d(2 * ch[i], ch[i], 3, padding=1, bias=False, indice_key=f"subm{i}"), ch[i]))
        self.head = nn.Sequential(nn.Linear(ch[0], ch[0]), nn.LeakyReLU(0.01), nn.Linear(ch[0], n_cls))

    def forward(self, x):
        skips = []
        for i in range(self.L):
            for blk in self.enc[i]: x = blk(x)
            if i < self.L - 1:
                skips.append(x); x = self.down[i](x)
        for j, i in enumerate(range(self.L - 2, -1, -1)):
            x = self.up[j](x)
            x = x.replace_feature(torch.cat([x.features, skips[i].features], 1))
            x = self.dec[j](x)
        return self.head(x.features)  # (N_active, K)


def _dblock(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.InstanceNorm2d(cout, affine=True), nn.LeakyReLU(0.01))


class DenseUNet(nn.Module):
    """Dense 2D U-Net baseline (analogue of nnU-Net (RAW)). Same widths / norm / activation."""
    def __init__(self, in_ch=3, n_cls=6, ch=(32, 64, 128, 256)):
        super().__init__(); L = len(ch); self.L = L
        self.enc = nn.ModuleList([nn.Sequential(_dblock(in_ch if i == 0 else ch[i], c), _dblock(c, c)) for i, c in enumerate(ch)])
        self.down = nn.ModuleList([nn.Conv2d(ch[i], ch[i + 1], 2, 2) for i in range(L - 1)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, 2) for i in range(L - 2, -1, -1)])
        self.dec = nn.ModuleList([nn.Sequential(_dblock(2 * ch[i], ch[i]), _dblock(ch[i], ch[i])) for i in range(L - 2, -1, -1)])
        self.head = nn.Conv2d(ch[0], n_cls, 1)

    def forward(self, x):
        sk = []
        for i in range(self.L):
            x = self.enc[i](x)
            if i < self.L - 1: sk.append(x); x = self.down[i](x)
        for j, i in enumerate(range(self.L - 2, -1, -1)):
            x = self.dec[j](torch.cat([self.up[j](x), sk[i]], 1))
        return self.head(x)


def build_model(cfg):
    ch = tuple(cfg.get("ch", (32, 64, 128, 256)))
    return SparseUNet(4, cfg["n_cls"], ch) if cfg["model"] == "sparse" else DenseUNet(3, cfg["n_cls"], ch)


def seg_loss(logits, y, K, smooth=0.1):
    """CE with label smoothing + Soft Dice over foreground classes (Eq. 5-7). logits (P,K), y (P,)."""
    ce = F.cross_entropy(logits, y, label_smoothing=smooth)
    p = logits.softmax(1); oh = F.one_hot(y, K).to(p.dtype)
    inter = (p * oh).sum(0); den = p.sum(0) + oh.sum(0)
    dice = ((2 * inter + 1e-5) / (den + 1e-5))[1:].mean()
    return ce + (1 - dice)


# ----------------------------------- training data --------------------------------------
class TrainDS(torch.utils.data.Dataset):
    def __init__(self, src, cfg, crop=128, rho=0.33, length=1000):
        self.src, self.cfg, self.crop, self.rho, self.length = src, cfg, crop, rho, length

    def __len__(self): return self.length

    def __getitem__(self, i):
        rng = np.random.default_rng((torch.initial_seed() + i) % (2 ** 32))
        img, lab = self.src.get(int(rng.integers(len(self.src))))
        k = int(rng.integers(4)); img, lab = np.rot90(img, k), np.rot90(lab, k)
        if rng.random() < 0.5: img, lab = img[:, ::-1], lab[:, ::-1]
        # stain / intensity jitter (helps cross-dataset transfer)
        img = np.clip(img.astype(np.float32) * rng.uniform(0.9, 1.1, 3) + rng.uniform(-15, 15, 3), 0, 255).astype(np.uint8)
        H, W = lab.shape; c = self.crop
        if rng.random() < self.rho and (lab > 0).any():      # foreground-biased window (rho = 0.33)
            ys, xs = np.nonzero(lab > 0); j = int(rng.integers(len(ys)))
            y0 = int(np.clip(ys[j] - c // 2, 0, H - c)); x0 = int(np.clip(xs[j] - c // 2, 0, W - c))
        else:
            y0 = int(rng.integers(0, H - c + 1)); x0 = int(rng.integers(0, W - c + 1))
        img = np.ascontiguousarray(img[y0:y0 + c, x0:x0 + c]); lab = np.ascontiguousarray(lab[y0:y0 + c, x0:x0 + c])
        if self.cfg["model"] == "sparse":
            ys, xs, f, l, _ = to_sparse(img, self.cfg, lab)
            return np.stack([ys, xs], 1).astype(np.int32), f, l.astype(np.int64)
        return dense_input(img, self.cfg), lab.astype(np.int64)


def collate_sparse(batch):
    co = [np.concatenate([np.full((len(c), 1), b, np.int32), c], 1) for b, (c, _, _) in enumerate(batch)]
    return (torch.from_numpy(np.concatenate(co)), torch.from_numpy(np.concatenate([f for _, f, _ in batch])),
            torch.from_numpy(np.concatenate([l for _, _, l in batch])), len(batch))


def collate_dense(batch):
    return torch.from_numpy(np.stack([b[0] for b in batch])), torch.from_numpy(np.stack([b[1] for b in batch]))


# ----------------------------------- inference ------------------------------------------
def _sync(dev):
    if dev.type == "cuda": torch.cuda.synchronize()


@torch.no_grad()
def infer_image(model, img, cfg, dev, win=128, overlap=0.5, sigma=0.5, bs=8):
    """Threshold -> sparse -> overlapping windows -> per-window softmax -> Gaussian-weighted fusion -> argmax."""
    H0, W0 = img.shape[:2]
    if H0 < win or W0 < win:
        img = np.pad(img, ((0, max(0, win - H0)), (0, max(0, win - W0)), (0, 0)), constant_values=255)
    H, W = img.shape[:2]; K = cfg["n_cls"]
    stride = max(1, int(win * (1 - overlap)))
    wins = [(y, x) for y in window_origins(H, win, stride) for x in window_origins(W, win, stride)]
    g = torch.from_numpy(gauss_window(win, sigma)).to(dev)
    acc = torch.zeros(H, W, K, device=dev); wsum = torch.zeros(H, W, device=dev)
    _sync(dev); t0 = time.perf_counter()
    if cfg["model"] == "sparse":
        ys, xs, f, _, mask = to_sparse(img, cfg)
        feat = np.zeros((H, W, f.shape[1]), np.float32); feat[ys, xs] = f
        _sync(dev); t1 = time.perf_counter()
        for s in range(0, len(wins), bs):
            chunk = wins[s:s + bs]; co, fe, gy, gx, ly_, lx_ = [], [], [], [], [], []
            for b, (y0, x0) in enumerate(chunk):
                ly, lx = np.nonzero(mask[y0:y0 + win, x0:x0 + win])
                if len(ly) == 0: continue
                co.append(np.stack([np.full(len(ly), b), ly, lx], 1)); fe.append(feat[y0 + ly, x0 + lx])
                gy.append(y0 + ly); gx.append(x0 + lx); ly_.append(ly); lx_.append(lx)
            if not co: continue
            x = spconv.SparseConvTensor(torch.from_numpy(np.concatenate(fe)).to(dev),
                                        torch.from_numpy(np.concatenate(co).astype(np.int32)).to(dev), [win, win], len(chunk))
            p = model(x).softmax(1)
            gy = torch.from_numpy(np.concatenate(gy)).to(dev); gx = torch.from_numpy(np.concatenate(gx)).to(dev)
            a = g[torch.from_numpy(np.concatenate(ly_)).to(dev), torch.from_numpy(np.concatenate(lx_)).to(dev)]
            acc.index_put_((gy, gx), p * a[:, None], accumulate=True)
            wsum.index_put_((gy, gx), a, accumulate=True)
    else:
        mask = None; xin = torch.from_numpy(dense_input(img, cfg)).to(dev)
        _sync(dev); t1 = time.perf_counter()
        for s in range(0, len(wins), bs):
            chunk = wins[s:s + bs]
            xb = torch.stack([xin[:, y:y + win, x:x + win] for y, x in chunk])
            p = model(xb).softmax(1).permute(0, 2, 3, 1) * g[None, :, :, None]
            for (y, x), pp in zip(chunk, p):
                acc[y:y + win, x:x + win] += pp; wsum[y:y + win, x:x + win] += g
    pred = acc.argmax(-1); pred[wsum == 0] = 0        # pixels outside the prior mask -> background
    _sync(dev); t2 = time.perf_counter()
    pred = pred.cpu().numpy().astype(np.uint8)[:H0, :W0]
    m = None if mask is None else mask[:H0, :W0]
    return pred, dict(t_prep=t1 - t0, t_fwd=t2 - t1, mask=m, retained=1.0 if m is None else float(m.mean()))


def evaluate(model, cfg, src, dev, multiclass, limit=None, n_vis=0, args=None):
    model.eval(); n = len(src) if not limit else min(limit, len(src))
    K = cfg["n_cls"]; conf = np.zeros((K, K), np.int64); confb = np.zeros((2, 2), np.int64)
    dimg, rec, ret, tp_, tf_, vis = [], [], [], [], [], []
    for i in range(n):
        img, lab = src.get(i)
        pred, info = infer_image(model, img, cfg, dev, **(args or {}))
        if multiclass: conf += conf_matrix(pred, lab, K)
        pb, gb = pred > 0, lab > 0
        confb += conf_matrix(pb.astype(np.uint8), gb.astype(np.uint8), 2)
        if gb.any():
            dimg.append(2 * (pb & gb).sum() / max(pb.sum() + gb.sum(), 1))
            rec.append(1.0 if info["mask"] is None else float((info["mask"] & gb).sum() / gb.sum()))
        ret.append(info["retained"]); tp_.append(info["t_prep"]); tf_.append(info["t_fwd"])
        if len(vis) < n_vis: vis.append((img, lab, pred, info["mask"]))
    res = dict(n_images=n, dice_fg_pooled=float(dice_from_conf(confb)[1]), dice_fg_per_image=float(np.mean(dimg)),
               mask_recall=float(np.mean(rec)), retained_fraction=float(np.mean(ret)),
               t_prep_s=float(np.mean(tp_[1:] if n > 1 else tp_)), t_forward_s=float(np.mean(tf_[1:] if n > 1 else tf_)))  # skip warm-up
    if multiclass:
        d = dice_from_conf(conf)
        res["dice_per_class"] = {PANNUKE_CLASSES[c]: float(d[c]) for c in range(1, K)}
        res["dice_nuclei_macro"] = float(np.mean(d[1:]))
    return res, vis


def save_vis(vis, path, multiclass):
    import os as _os; _os.environ["MPLBACKEND"] = "Agg"
    import matplotlib; matplotlib.use("Agg", force=True); import matplotlib.pyplot as plt
    pal = np.array([[0, 0, 0], [255, 0, 0], [255, 200, 0], [0, 200, 0], [0, 0, 0], [0, 120, 255]], np.uint8)
    pal[4] = [255, 0, 255]
    fig, ax = plt.subplots(len(vis), 4, figsize=(12, 3 * len(vis)), squeeze=False)
    for r, (img, lab, pred, m) in enumerate(vis):
        col = (lambda a: pal[a]) if multiclass else (lambda a: np.stack([a * 255] * 3, -1).astype(np.uint8))
        maskimg = img.copy() if m is None else (img * m[..., None]).astype(np.uint8)
        for c, (im, t) in enumerate([(img, "input"), (col(lab), "ground truth"), (col(pred), "PathBonnet"), (maskimg, "sparse input (prior)")]):
            ax[r, c].imshow(im); ax[r, c].set_title(t if r == 0 else ""); ax[r, c].axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


# ----------------------------------- commands -------------------------------------------
def get_dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_ckpt(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False); cfg = ck["cfg"]
    model = build_model(cfg).to(dev); model.load_state_dict(ck["state"]); return model.eval(), cfg


def cmd_train(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed); os.makedirs(a.out, exist_ok=True)
    dev = get_dev(); f1, f2, f3 = find_pannuke(a.pannuke_root)
    if a.model == "sparse": assert dev.type == "cuda", "spconv needs a CUDA GPU (use Colab/Kaggle) - or run --model dense"
    cfg = calibrate(f1, a.prior, a.target_recall); cfg.update(model=a.model, n_cls=6, ch=list(a.ch))
    print(f"[prior={a.prior}] tau={cfg['tau']:.4f} keeps ~{100*a.target_recall:.0f}% nuclear px; "
          f"retained (sparsity) on train = {100*cfg['retained_train']:.1f}% of pixels")
    model = build_model(cfg).to(dev)
    print("params (M):", sum(p.numel() for p in model.parameters()) / 1e6)
    dl = torch.utils.data.DataLoader(TrainDS(f1, cfg, a.crop, 0.33, a.iters * a.batch), batch_size=a.batch, num_workers=a.workers,
                                     collate_fn=collate_sparse if a.model == "sparse" else collate_dense, drop_last=True,
                                     persistent_workers=a.workers > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * a.iters)
    best, log = -1, []
    for ep in range(a.epochs):
        model.train(); tl, nb, t0 = 0.0, 0, time.time()
        for batch in dl:
            if a.model == "sparse":
                co, fe, y, B = batch
                if len(y) == 0: continue
                x = spconv.SparseConvTensor(fe.to(dev), co.to(dev), [a.crop, a.crop], B)
                loss = seg_loss(model(x), y.to(dev), cfg["n_cls"])
            else:
                x, y = batch; out = model(x.to(dev))
                loss = seg_loss(out.permute(0, 2, 3, 1).reshape(-1, cfg["n_cls"]), y.to(dev).reshape(-1), cfg["n_cls"])
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            opt.step(); sched.step(); tl += loss.item(); nb += 1
        r, _ = evaluate(model, cfg, f2, dev, True, limit=a.val_n)
        score = r["dice_nuclei_macro"]; log.append(dict(epoch=ep + 1, loss=tl / max(nb, 1), **{k: r[k] for k in ("dice_fg_pooled", "dice_nuclei_macro")}))
        print(f"ep {ep+1:3d}/{a.epochs} loss {tl/max(nb,1):.4f} | val fgDice {r['dice_fg_pooled']:.4f} nucleiDice {score:.4f} | {time.time()-t0:.0f}s")
        if score > best:
            best = score; torch.save(dict(cfg=cfg, state=model.state_dict()), os.path.join(a.out, "best.pt"))
        json.dump(log, open(os.path.join(a.out, "train_log.json"), "w"), indent=1)
    print("best val nuclei-macro Dice:", best)


def cmd_eval(a):
    dev = get_dev(); model, cfg = load_ckpt(a.ckpt, dev); out = a.out or os.path.dirname(a.ckpt)
    kw = dict(win=a.win, overlap=a.overlap, sigma=a.sigma, bs=a.bs); results = dict(cfg={k: cfg[k] for k in ("prior", "tau", "model", "target_recall")})
    results["cfg"]["name"] = os.path.basename(os.path.normpath(out))
    if a.pannuke_root:
        r, vis = evaluate(model, cfg, find_pannuke(a.pannuke_root)[2], dev, True, a.limit, a.n_vis, kw)
        results["internal_pannuke_fold3"] = r; print("INTERNAL (PanNuke fold 3):", json.dumps(r, indent=1))
        if vis: save_vis(vis, os.path.join(out, "vis_internal.png"), True)
    ext = []
    if a.consep_root: ext.append(("external_consep", consep_set(a.consep_root)))
    if a.external_dir: ext.append(("external_folder", folder_set(a.external_dir)))
    for name, src in ext:
        r, vis = evaluate(model, cfg, src, dev, False, a.limit, a.n_vis, kw)   # binary nuclei vs background (label sets differ)
        results[name] = r; print(name.upper(), json.dumps(r, indent=1))
        if vis: save_vis(vis, os.path.join(out, f"vis_{name}.png"), False)
    json.dump(results, open(os.path.join(out, "results.json"), "w"), indent=1); print("saved", os.path.join(out, "results.json"))


def cmd_predict(a):
    from PIL import Image
    dev = get_dev(); model, cfg = load_ckpt(a.ckpt, dev); os.makedirs(a.out, exist_ok=True)
    pal = np.array([[0, 0, 0], [255, 0, 0], [255, 200, 0], [0, 200, 0], [255, 0, 255], [0, 120, 255]], np.uint8)
    files = sorted(sum([glob.glob(os.path.join(a.images, f"*.{e}")) for e in ("png", "jpg", "jpeg", "tif", "tiff")], []))
    for p in files:
        img = np.array(Image.open(p).convert("RGB")); pred, info = infer_image(model, img, cfg, dev, a.win, a.overlap, a.sigma, a.bs)
        stem = os.path.splitext(os.path.basename(p))[0]
        Image.fromarray(pred).save(os.path.join(a.out, stem + "_labels.png"))
        ov = (0.6 * img + 0.4 * pal[pred]).astype(np.uint8); ov[pred == 0] = img[pred == 0]
        Image.fromarray(ov).save(os.path.join(a.out, stem + "_overlay.png"))
        print(f"{stem}: {info['t_prep']+info['t_fwd']:.2f}s, kept {100*info['retained']:.1f}% px")


def cmd_table(a):
    print("| Run | Model | Prior | kept px % | GT recall % | PanNuke fg Dice | Nuclei macro Dice | Neo | Inf | Con | Dead | Epi | CoNSeP fg Dice | Fwd s/img |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in a.files:
        r = json.load(open(p)); c = r["cfg"]; i = r.get("internal_pannuke_fold3", {}); e = r.get("external_consep", {}); d = i.get("dice_per_class", {})
        f = lambda v: "-" if v is None else f"{100*v:.2f}"
        print(f"| {c['name']} | {c['model']} | {c['prior']} | {f(i.get('retained_fraction'))} | {f(i.get('mask_recall'))} | {f(i.get('dice_fg_pooled'))} | "
              f"{f(i.get('dice_nuclei_macro'))} | {f(d.get('neoplastic'))} | {f(d.get('inflammatory'))} | {f(d.get('connective'))} | {f(d.get('dead'))} | "
              f"{f(d.get('epithelial'))} | {f(e.get('dice_fg_pooled'))} | {i.get('t_forward_s', float('nan')):.3f} |")


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train"); t.set_defaults(fn=cmd_train)
    t.add_argument("--pannuke_root", required=True); t.add_argument("--out", required=True)
    t.add_argument("--model", choices=["sparse", "dense"], default="sparse")
    t.add_argument("--prior", choices=["hema", "tissue", "raw"], default="hema")
    t.add_argument("--target_recall", type=float, default=0.98)
    t.add_argument("--epochs", type=int, default=40); t.add_argument("--iters", type=int, default=200, help="batches per epoch")
    t.add_argument("--batch", type=int, default=8); t.add_argument("--crop", type=int, default=128)
    t.add_argument("--lr", type=float, default=2e-3); t.add_argument("--workers", type=int, default=4)
    t.add_argument("--ch", type=int, nargs="+", default=[32, 64, 128, 256]); t.add_argument("--val_n", type=int, default=150)
    t.add_argument("--seed", type=int, default=0)
    for name, fn in (("eval", cmd_eval), ("predict", cmd_predict)):
        e = sp.add_parser(name); e.set_defaults(fn=fn); e.add_argument("--ckpt", required=True)
        e.add_argument("--win", type=int, default=128); e.add_argument("--overlap", type=float, default=0.5)
        e.add_argument("--sigma", type=float, default=0.5); e.add_argument("--bs", type=int, default=8)
        if name == "eval":
            e.add_argument("--pannuke_root"); e.add_argument("--consep_root"); e.add_argument("--external_dir")
            e.add_argument("--out"); e.add_argument("--limit", type=int); e.add_argument("--n_vis", type=int, default=4)
        else:
            e.add_argument("--images", required=True); e.add_argument("--out", required=True)
    tb = sp.add_parser("table"); tb.set_defaults(fn=cmd_table); tb.add_argument("files", nargs="+")
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
