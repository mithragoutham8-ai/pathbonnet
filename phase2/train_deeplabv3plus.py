#!/usr/bin/env python3
"""
DeepLabV3+ dense-baseline experiment for PathBonnet (Phase 2).

Trains segmentation-models-pytorch DeepLabV3Plus (ResNet-101 encoder, ImageNet
weights) on either:
  - nuinsseg (default): Kaggle H&E nuclei dataset (665 patches, 30k+ nuclei).
    IMPORTANT - class mapping: NuInsSeg ships BINARY / instance masks only
    (background=0, nuclei=1); it has NO nuclear-type labels. The honest mapping
    is therefore binary: classes=2 (0=background, 1=nuclei). Ambiguous-area
    masks supplied by the authors are excluded from loss and metrics
    (ignore_index=255). Do NOT invent type labels for this dataset.
  - pannuke: Mithra's local PanNuke layout, fold 1 train / fold 2 val / fold 3
    test, with her 6-class scheme (0=background, 1..5 = neoplastic,
    inflammatory, connective, dead, epithelial) for the matched-protocol
    comparison against her dense U-Net.

Matched training protocol (same as her PathBonnet runs):
  40 epochs x 150 batches, batch size 8, AdamW lr 2e-3 / wd 1e-4,
  OneCycleLR peak lr 2e-3, loss = label-smoothed CE (0.1) + soft Dice over
  foreground classes. Best checkpoint by validation foreground Dice.
  Foreground Dice = mean Dice over the 5 foreground classes (PanNuke) or the
  Dice of the single foreground class (NuInsSeg); background excluded.

What this experiment answers: DeepLabV3+ is a DENSE model, so this tests
architecture strength (expect higher accuracy than her 2.3M-param U-Net, slower
inference) - NOT sparsity. It cannot overturn her occupancy finding (tissue
fills the patch at patch scale, so sparse bookkeeping never pays off).

Usage
  python train_deeplabv3plus.py --dataset nuinsseg --out runs/dl3p_nuinsseg
  python train_deeplabv3plus.py --dataset pannuke --pannuke_root DATA/PanNuke --out runs/dl3p_pannuke
"""
import argparse, glob, json, math, os, time
import numpy as np

PANNUKE_CLASSES = ["background", "neoplastic", "inflammatory", "connective", "dead", "epithelial"]
NUINSSEG_CLASSES = ["background", "nuclei"]
IGNORE = 255  # ambiguous-area pixels (NuInsSeg): excluded from loss and metrics


# =====================================================================================
# Data -- PanNuke (same layout as pathbonnet.py), NuInsSeg (Kaggle via kagglehub)
# =====================================================================================
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


def download_nuinsseg(kaggle_dir=None):
    """Download the NuInsSeg Kaggle dataset (needs Kaggle credentials configured)."""
    try:
        import kagglehub
    except ImportError:
        raise SystemExit("kagglehub not installed (pip install kagglehub) and Kaggle credentials required; "
                         "or place the dataset manually and pass --kaggle_dir")
    if kaggle_dir and os.path.isdir(kaggle_dir):
        print("using existing dataset dir:", kaggle_dir); return kaggle_dir
    d = kagglehub.dataset_download("ipateam/nuinsseg")
    print("downloaded to:", d); return d


class NuInsSegSet:
    """665 H&E patches (512x512). Binary nuclei masks + ambiguous-area masks.

    The Kaggle archive layout is discovered defensively: image files are found
    recursively, and each image's mask is located by filename-stem matching,
    preferring folders named like 'mask binary' (the authors' plain binary
    masks). Ambiguous masks (folders containing 'ambiguous') mark pixels that
    even experts could not annotate reliably -> set to IGNORE (255).

    The ipateam Kaggle upload contains extra copies of the tissue images in
    extra folders (~11x the true 665). Discovery dedupes by filename stem so
    each tissue image is used exactly once; the KFold split then runs on
    unique stems, so copies cannot leak across train/val/test folds.
    """
    EXTS = ("png", "jpg", "jpeg", "tif", "tiff", "bmp")

    def __init__(self, root, img_keep=None):
        # Single pass over the tree; index every image-like file by stem.
        # (The previous version re-globbed the whole tree per image -- with
        # 665 images x 2 lookups x 6 extensions that was ~8,000 recursive
        # scans of a 48k-file tree, which effectively hangs on Kaggle's
        # mounted storage. One walk + dict lookups is equivalent and fast.)
        #
        # 2026-10-20: the ipateam Kaggle upload holds ~11x too many image-like
        # files (7,315 found for 665 tissue images) -- extra copies of the
        # tissue images live in extra folders. Dedup by filename stem so each
        # tissue image is used exactly once; because the KFold split then runs
        # on unique stems, no copy can leak across train/val/test folds.
        # If the discovery still looks wrong, pass --img_keep "<folder words>"
        # to restrict image files to paths containing that substring.
        by_stem = {}
        for dirpath, _dirs, files in os.walk(root):
            for fn in files:
                if os.path.splitext(fn)[1].lower().lstrip(".") in self.EXTS:
                    by_stem.setdefault(os.path.splitext(fn)[0], []).append(os.path.join(dirpath, fn))
        # keep files that look like raw images, not masks/overlays/ambiguous maps
        imgs = []
        for paths in by_stem.values():
            for p in paths:
                pl = p.lower()
                if ("mask" not in pl and "ambiguous" not in pl and "overlay" not in pl
                        and "weight" not in pl and "distance" not in pl and "vague" not in pl):
                    if img_keep is None or img_keep.lower() in pl:
                        imgs.append(p)
        assert imgs, f"No raw images found under {root}"
        seen, uniq, ndup, kept_folders = set(), [], 0, {}
        # when duplicates of a stem exist, keep the authors' canonical
        # "tissue images" copy -- other folders (e.g. "nuclei border") may
        # hold processed variants rather than the raw H&E image
        for p in sorted(imgs, key=lambda q: (0 if "tissue images" in q.lower() else 1, q)):
            stem = os.path.splitext(os.path.basename(p))[0]
            if stem in seen:
                ndup += 1
                continue
            seen.add(stem)
            uniq.append(p)
            d = os.path.dirname(p)
            kept_folders[d] = kept_folders.get(d, 0) + 1
        imgs = uniq
        if ndup:
            print(f"dedup: {ndup} extra copies removed by filename stem; "
                  f"{len(imgs)} unique tissue images kept")
        print("image folders kept:")
        for d, n in sorted(kept_folders.items(), key=lambda kv: -kv[1])[:8]:
            print(f"   {n:5d}  {d}")
        self.items = []
        for ip in sorted(imgs):
            stem = os.path.splitext(os.path.basename(ip))[0]
            mp = self._find_mask(by_stem, stem)
            am = self._find_ambiguous(by_stem, stem)
            self.items.append((ip, mp, am))
        print(f"NuInsSeg: {len(self.items)} images; {sum(m is not None for _, m, _ in self.items)} with binary masks; "
              f"{sum(a is not None for _, _, a in self.items)} with ambiguous masks")

    def _find_mask(self, by_stem, stem):
        cands = []
        for s, paths in by_stem.items():
            if s.startswith(stem):
                cands += paths
        cands = [p for p in cands if "mask" in p.lower() and "ambiguous" not in p.lower()
                 and "overlay" not in p.lower() and "weight" not in p.lower() and "distance" not in p.lower()
                 and "erode" not in p.lower() and "border" not in p.lower()]
        # prefer the plain binary masks over label/instance variants
        def key(p):
            pl = p.lower()
            return (0 if "mask binary" in pl or "mask_binary" in pl else 1, len(p))
        cands.sort(key=key)
        return cands[0] if cands else None

    def _find_ambiguous(self, by_stem, stem):
        cands = []
        for s, paths in by_stem.items():
            if stem in s:
                cands += paths
        cands = [p for p in cands if "ambiguous" in p.lower() or "vague" in p.lower()]
        return cands[0] if cands else None

    def __len__(self): return len(self.items)

    def get(self, i):
        from PIL import Image
        ip, mp, am = self.items[i]
        img = np.array(Image.open(ip).convert("RGB"))
        lab = np.zeros(img.shape[:2], np.uint8)
        if mp is not None:
            m = np.array(Image.open(mp)); m = m.max(-1) if m.ndim == 3 else m
            lab[m > 0] = 1
        if am is not None:
            a = np.array(Image.open(am)); a = a.max(-1) if a.ndim == 3 else a
            lab[a > 0] = IGNORE
        return img, lab


def nuinsseg_splits(n, seed=19):
    """Authors' recommended 5-fold CV (KFold, n_splits=5, shuffle, random_state=19).
    Single quick run: fold 4 -> test, fold 3 -> val, folds 0-2 -> train."""
    try:
        from sklearn.model_selection import KFold
        parts = [tst for _, tst in KFold(n_splits=5, shuffle=True, random_state=seed).split(np.arange(n))]
    except ImportError:  # deterministic fallback without sklearn
        rng = np.random.default_rng(seed); idx = rng.permutation(n)
        parts = [idx[i * n // 5:(i + 1) * n // 5] for i in range(5)]
    te, va = np.asarray(parts[4]), np.asarray(parts[3])
    tr = np.setdiff1d(np.arange(n), np.concatenate([va, te]))
    return tr, va, te


class SubSet:
    def __init__(self, src, idx): self.src, self.idx = src, np.asarray(idx)
    def __len__(self): return len(self.idx)
    def get(self, i): return self.src.get(int(self.idx[i]))


# =====================================================================================
# ==== TORCH ====
# =====================================================================================
import torch, torch.nn as nn, torch.nn.functional as F
import segmentation_models_pytorch as smp


def seg_loss(logits, y, fg_classes, smooth=0.1):
    """CE with label smoothing + soft Dice over foreground classes.
    logits (P,K), y (P,) with IGNORE (255) pixels excluded from both terms."""
    valid = y != IGNORE
    if valid.sum() == 0:
        return logits.sum() * 0.0
    logits, y = logits[valid], y[valid].clone()
    y[y == IGNORE] = 0
    ce = F.cross_entropy(logits, y, label_smoothing=smooth)
    p = logits.softmax(1); K = logits.shape[1]
    oh = F.one_hot(y, K).to(p.dtype)
    inter = (p * oh).sum(0); den = p.sum(0) + oh.sum(0)
    dice = ((2 * inter + 1e-5) / (den + 1e-5))[fg_classes].mean()
    return ce + (1 - dice)


def build_model(encoder, n_cls):
    # smp 0.5.0 verified signature: DeepLabV3Plus(encoder_name, encoder_weights,
    # in_channels, classes, activation, ...). activation=None -> raw logits.
    # decoder_atrous_rates=(6,12,18): the paper's rates for output_stride=16.
    return smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights="imagenet",
                             in_channels=3, classes=n_cls, activation=None,
                             decoder_atrous_rates=(6, 12, 18))


def preprocess_fn(encoder):
    fn = smp.encoders.get_preprocessing_fn(encoder, pretrained="imagenet")
    return lambda img: fn(img.astype(np.float32))  # -> float32, ImageNet mean/std


class TrainDS(torch.utils.data.Dataset):
    """Random crops with foreground-biased window sampling (rho=0.33), same
    augments as pathbonnet.py (rot90, flip, stain/intensity jitter)."""
    def __init__(self, src, pp, crop=256, rho=0.33, length=1000, seed=0):
        self.src, self.pp, self.crop, self.rho, self.length, self.seed = src, pp, crop, rho, length, seed

    def __len__(self): return self.length

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed + torch.initial_seed() + i) % (2 ** 32))
        img, lab = self.src.get(int(rng.integers(len(self.src))))
        k = int(rng.integers(4)); img, lab = np.rot90(img, k), np.rot90(lab, k)
        if rng.random() < 0.5: img, lab = img[:, ::-1], lab[:, ::-1]
        img = np.clip(img.astype(np.float32) * rng.uniform(0.9, 1.1, 3) + rng.uniform(-15, 15, 3), 0, 255).astype(np.uint8)
        H, W = lab.shape[:2]; c = min(self.crop, H, W)
        fg = (lab > 0) & (lab != IGNORE)
        if rng.random() < self.rho and fg.any():
            ys, xs = np.nonzero(fg); j = int(rng.integers(len(ys)))
            y0 = int(np.clip(ys[j] - c // 2, 0, H - c)); x0 = int(np.clip(xs[j] - c // 2, 0, W - c))
        else:
            y0 = int(rng.integers(0, H - c + 1)); x0 = int(rng.integers(0, W - c + 1))
        img = np.ascontiguousarray(img[y0:y0 + c, x0:x0 + c]); lab = np.ascontiguousarray(lab[y0:y0 + c, x0:x0 + c])
        x = self.pp(img).transpose(2, 0, 1).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(lab.astype(np.int64))


# ----------------------------------- metrics ------------------------------------------
def conf_matrix(pred, gt, K):
    return np.bincount(gt.ravel().astype(np.int64) * K + pred.ravel().astype(np.int64), minlength=K * K).reshape(K, K)


def dice_from_conf(conf):
    tp = np.diag(conf).astype(np.float64); fp = conf.sum(0) - tp; fn = conf.sum(1) - tp
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)


@torch.no_grad()
def infer_image(model, img, pp, dev):
    """Full-image forward (pad to multiple of 16 for output_stride=16), argmax."""
    H0, W0 = img.shape[:2]
    Hp, Wp = math.ceil(H0 / 16) * 16, math.ceil(W0 / 16) * 16
    x = pp(img).transpose(2, 0, 1)[None]
    if (Hp, Wp) != (H0, W0):
        x = np.pad(x, ((0, 0), (0, 0), (0, Hp - H0), (0, Wp - W0)))
    if dev.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    p = model(torch.from_numpy(x.astype(np.float32)).to(dev)).softmax(1)
    if dev.type == "cuda": torch.cuda.synchronize()
    t1 = time.perf_counter()
    return p[0].argmax(0).cpu().numpy().astype(np.uint8)[:H0, :W0], t1 - t0


def evaluate(model, src, pp, dev, fg_classes, limit=0):
    model.eval(); n = len(src) if not limit else min(limit, len(src))
    K = int(max(fg_classes)) + 1; conf = np.zeros((K, K), np.int64); ts = []
    for i in range(n):
        img, lab = src.get(i)
        pred, t = infer_image(model, img, pp, dev)
        valid = lab != IGNORE
        conf += conf_matrix(pred[valid], lab[valid].astype(np.uint8), K)
        ts.append(t)
    d = dice_from_conf(conf)
    fg = float(np.mean([d[c] for c in fg_classes]))
    return dict(n_images=n, fg_dice=fg,
                per_class={("nuclei" if K == 2 else PANNUKE_CLASSES[c]): float(d[c]) for c in fg_classes},
                ms_per_image=float(np.mean(ts[1:]) * 1000) if n > 1 else float(np.mean(ts) * 1000))  # skip warm-up


# ----------------------------------- train --------------------------------------------
def get_dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["nuinsseg", "pannuke"], default="nuinsseg")
    ap.add_argument("--pannuke_root", default=None, help="PanNuke root (3 folds of images.npy/masks.npy)")
    ap.add_argument("--kaggle_dir", default=None, help="existing NuInsSeg dir (else downloaded via kagglehub)")
    ap.add_argument("--img_keep", default=None,
                    help="NuInsSeg only: keep only image files whose path contains this "
                         "substring (e.g. 'tissue images'); use if the discovery printout "
                         "shows image files outside the true tissue-image folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", default="resnet101")
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val_limit", type=int, default=400); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed); os.makedirs(a.out, exist_ok=True)
    dev = get_dev()
    pp = preprocess_fn(a.encoder)

    if a.dataset == "pannuke":
        assert a.pannuke_root, "--pannuke_root required for --dataset pannuke"
        f1, f2, f3 = find_pannuke(a.pannuke_root)
        tr_src, va_src, te_src = f1, f2, f3
        n_cls, fg_classes, class_names = 6, [1, 2, 3, 4, 5], PANNUKE_CLASSES
        split_note = "PanNuke fold1 train / fold2 val / fold3 test (her matched protocol)"
    else:
        root = download_nuinsseg(a.kaggle_dir)
        full = NuInsSegSet(root, a.img_keep)
        tr_i, va_i, te_i = nuinsseg_splits(len(full))
        tr_src, va_src, te_src = SubSet(full, tr_i), SubSet(full, va_i), SubSet(full, te_i)
        n_cls, fg_classes, class_names = 2, [1], NUINSSEG_CLASSES
        split_note = "NuInsSeg 5-fold CV (authors' scheme, seed 19): folds 0-2 train / fold 3 val / fold 4 test"
    print(split_note, "| classes:", class_names)

    model = build_model(a.encoder, n_cls).to(dev)
    print("params (M):", sum(p.numel() for p in model.parameters()) / 1e6)
    cfg = dict(dataset=a.dataset, encoder=a.encoder, n_cls=n_cls, class_names=class_names,
               epochs=a.epochs, iters=a.iters, batch=a.batch, crop=a.crop, lr=a.lr, seed=a.seed,
               split_note=split_note)
    json.dump(cfg, open(os.path.join(a.out, "cfg.json"), "w"), indent=1)

    dl = torch.utils.data.DataLoader(TrainDS(tr_src, pp, a.crop, 0.33, a.iters * a.batch, a.seed),
                                     batch_size=a.batch, num_workers=a.workers, drop_last=True,
                                     persistent_workers=a.workers > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * a.iters)
    best, log = -1, []
    for ep in range(a.epochs):
        model.train(); tl, nb, t0 = 0.0, 0, time.time()
        for x, y in dl:
            out = model(x.to(dev))                       # (B,K,H,W) logits
            P, K = out.shape[0] * out.shape[2] * out.shape[3], out.shape[1]
            loss = seg_loss(out.permute(0, 2, 3, 1).reshape(P, K), y.to(dev).reshape(-1), fg_classes)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            opt.step(); sched.step(); tl += loss.item(); nb += 1
        r = evaluate(model, va_src, pp, dev, fg_classes, a.val_limit)
        score = r["fg_dice"]
        row = dict(epoch=ep + 1, loss=tl / max(nb, 1), val_fg_dice=score)
        log.append(row)
        print(f"ep {ep+1:3d}/{a.epochs} loss {tl/max(nb,1):.4f} | val fgDice {score:.4f} | {time.time()-t0:.0f}s")
        if score > best:
            best = score
            torch.save(dict(cfg=cfg, state=model.state_dict()), os.path.join(a.out, "best.pt"))
        json.dump(log, open(os.path.join(a.out, "train_log.json"), "w"), indent=1)
    print("best val fg Dice:", round(best, 4))
    print("test split:", {"pannuke": "--pannuke_root fold 3", "nuinsseg": "fold 4"}[a.dataset],
          "-> run evaluate.py --ckpt", os.path.join(a.out, "best.pt"))


if __name__ == "__main__":
    main()
