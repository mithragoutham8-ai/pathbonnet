#!/usr/bin/env python3
"""
Evaluate a trained DeepLabV3+ checkpoint.

Reports foreground Dice (same definition as the PathBonnet report: mean Dice
over the 5 foreground nuclear classes for PanNuke, background excluded; Dice of
the single foreground class for NuInsSeg) and mean inference time in ms/image
(first-image warm-up excluded, same convention as pathbonnet.py).

Reference numbers from her PathBonnet report (PanNuke fold 3 test, 2,722 imgs):
  dense U-Net baseline : fg Dice 0.797,  12.2 ms/img   <- the model to beat on accuracy
  sparse-hema          : fg Dice 0.781,  59.6 ms/img   <- the negative result
Expected for DeepLabV3+ (ResNet-101): HIGHER accuracy than 0.797 (much bigger
model, ImageNet-pretrained backbone), SLOWER than 12.2 ms/img (atrous convs are
compute-heavy). That outcome tests architecture strength, NOT sparsity.

Usage
  python evaluate.py --ckpt runs/dl3p_nuinsseg/best.pt
  python evaluate.py --ckpt runs/dl3p_pannuke/best.pt --dataset pannuke --pannuke_root DATA/PanNuke
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_deeplabv3plus import (PanNukeFold, NuInsSegSet, SubSet, nuinsseg_splits,
                                 find_pannuke, download_nuinsseg, preprocess_fn,
                                 build_model, evaluate, get_dev, PANNUKE_CLASSES, NUINSSEG_CLASSES)
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", choices=["nuinsseg", "pannuke"], default=None,
                    help="default: read from checkpoint cfg")
    ap.add_argument("--pannuke_root", default=None)
    ap.add_argument("--kaggle_dir", default=None)
    ap.add_argument("--limit", type=int, default=0, help="0 = full test split")
    a = ap.parse_args()

    dev = get_dev()
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    cfg = ck["cfg"]
    dataset = a.dataset or cfg["dataset"]
    n_cls, class_names = cfg["n_cls"], cfg["class_names"]
    fg_classes = [1, 2, 3, 4, 5] if n_cls == 6 else [1]
    pp = preprocess_fn(cfg["encoder"])

    model = build_model(cfg["encoder"], n_cls).to(dev)
    model.load_state_dict(ck["state"]); model.eval()

    if dataset == "pannuke":
        assert a.pannuke_root, "--pannuke_root required for --dataset pannuke"
        te_src = find_pannuke(a.pannuke_root)[2]
        split = "PanNuke fold 3 (internal test)"
    else:
        full = NuInsSegSet(download_nuinsseg(a.kaggle_dir))
        _, _, te_i = nuinsseg_splits(len(full))
        te_src = SubSet(full, te_i)
        split = "NuInsSeg fold 4 (test)"

    r = evaluate(model, te_src, pp, dev, fg_classes, a.limit)
    r["split"] = split
    print(f"TEST ({split}): n={r['n_images']}")
    print(f"  fg Dice (background excluded): {r['fg_dice']:.4f}")
    for name, d in r["per_class"].items():
        print(f"    {name:12s} {d:.4f}")
    print(f"  inference: {r['ms_per_image']:.1f} ms/image (warm-up excluded)")
    print("  reference - dense U-Net: fg Dice 0.797, 12.2 ms/img | sparse-hema: 0.781, 59.6 ms/img")

    out = os.path.join(os.path.dirname(a.ckpt), "eval_results.json")
    json.dump(r, open(out, "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
