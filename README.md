# PathBonnet: sparse-prior segmentation for digital pathology

Adapts the Bonnet pipeline (Zhu, Martelleto Rezende, Yang, Ye, Gao, Luo, Huang, Yang,
*Bonnet: Ultra-Fast Whole-Body Bone Segmentation from CT Scans*, arXiv:2601.22576, 2026,
ISBI 2026 Oral — CT bone) to H&E digital pathology.

## Datasets used

- **PanNuke** — training, validation, and internal test (fold 1 / fold 2 / fold 3).
  https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke
- **CPM-17** — external, cross-dataset test with no fine-tuning (32 test images).
  Used in place of CoNSeP: CoNSeP's official download now sits behind a Warwick
  institutional sign-on that was not accessible, so CPM-17 was substituted. It comes
  from the same HoVer-Net data ecosystem and uses the same `.mat` `inst_map` label
  format, so no code changes were needed. This substitution is stated openly in the
  project report rather than silently swapped.
  (Vu, Q. D., Graham, S., Kurc, T., et al. *Methods for Segmentation and Classification
  of Digital Microscopy Tissue Images*. Frontiers in Bioengineering and Biotechnology,
  7:53, 2019.)

| Bonnet (CT) | PathBonnet (H&E) |
|---|---|
| HU threshold [200, 3000] gives sparse voxels | Stain prior threshold gives sparse pixels. Threshold is calibrated on the train fold to keep 98% of nuclear pixels |
| Sparse 3D U-Net (spconv) | Sparse 2D U-Net (spconv 2D), SparseInstanceNorm, LeakyReLU(0.01), MLP head |
| 128^3 windows, overlap 0.5, Gaussian fusion (sigma 0.5) | 128x128 windows, overlap 0.5, Gaussian fusion (sigma 0.5) |
| CE (label smoothing) + Soft Dice | Same |
| Train TotalSegmentator, test RibSeg / Pelvic1K / Spine1K with no fine-tuning | Train PanNuke fold 1, validate fold 2, test fold 3 (internal), then CPM-17 (external, no fine-tuning) |
| nnU-Net and nnU-Net (RAW) baselines | `--model dense` and `--prior raw` |

## 1. Environment (needs a CUDA GPU for the sparse model; Colab T4 is enough)

```bash
pip install torch numpy scipy pillow matplotlib
pip install spconv-cu121      # pick the build that matches your CUDA: cu118 / cu121 / ...
```

The dense baseline runs without spconv, and on CPU (slowly). spconv's JIT compile
currently fails on Python 3.13; a Python 3.10 environment (e.g. via miniforge) is the
reliable path on recent Colab runtimes.

The dense baseline runs without spconv, and on CPU (slowly).

## 2. Data

- **PanNuke** (train and internal test): https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke. Download the 3 fold zips and unzip them into one folder. The script finds `images.npy` and `masks.npy` in each fold recursively. There are 256x256 patches, 19 tissue types and 5 nuclear classes. A Hugging Face mirror also exists (`lz222/PanNuke`).
- **CPM-17** (external): freely downloadable (e.g. via the HoVer-Net project's shared Google Drive). Expected layout is `CPM17/Test/Images/*.png` and `CPM17/Test/Labels/*.mat`, each `.mat` holding an `inst_map` key. Its class set differs from PanNuke, so the external test is binary (nuclei vs background). The same loader path also works for CoNSeP if you have access to it — pass its root the same way.
- **Your own samples**: `my_data/images/*.png`, and optionally `my_data/masks/*.png` (non-zero means nucleus). Use `--external_dir my_data` to score them, or `predict` to just segment them.
- Licences are research and non-commercial (CC BY-NC-SA for PanNuke). Cite the dataset papers in the report.

## 3. Run

```bash
# main method
python pathbonnet.py train --pannuke_root DATA/PanNuke --model sparse --prior hema   --out runs/sparse_hema
# ablations (same as the paper's baselines / RAW variant)
python pathbonnet.py train --pannuke_root DATA/PanNuke --model sparse --prior tissue --out runs/sparse_tissue
python pathbonnet.py train --pannuke_root DATA/PanNuke --model sparse --prior raw    --out runs/sparse_raw
python pathbonnet.py train --pannuke_root DATA/PanNuke --model dense  --prior raw    --out runs/dense_raw

# evaluate every run (internal + external, no fine-tuning)
for r in sparse_hema sparse_tissue sparse_raw dense_raw; do
  python pathbonnet.py eval --ckpt runs/$r/best.pt --pannuke_root DATA/PanNuke --consep_root DATA/CPM17 --out runs/$r
done

# one markdown table for the report
python pathbonnet.py table runs/*/results.json

# segment your own images
python pathbonnet.py predict --ckpt runs/sparse_hema/best.pt --images my_samples/ --out preds/
```

Note: the CLI flag is still named `--consep_root` for backward compatibility with the
original Bonnet-style naming; point it at your CPM-17 (or CoNSeP) root — the loader
only cares about the `Test/Images` + `Test/Labels` (`.mat`, key `inst_map`) layout.

Defaults are 40 epochs of 200 batches of 8 crops, which is small enough for a Colab session. Raise `--epochs` for the final numbers. Add `--seed 1 2` style repeats if you want mean and std across seeds.

## 4. Results (PanNuke fold 3 internal / CPM-17 external, no fine-tuning)

| Config | Pixels kept (int.) | Mask recall (int.) | PanNuke fg Dice | Macro Dice | CPM-17 fg Dice | Internal ms/img | External ms/img |
|---|---|---|---|---|---|---|---|
| dense, raw | 100% | 100% | 79.72% | 0.385 | 83.55% | 12.2 | 94.3 |
| sparse, hema | 64.4% | 97.7% | 78.09% | 0.355 | 82.26% | 60.1 | 361.8 |
| sparse, tissue | 69.0% | 97.8% | 76.30% | 0.369 | 80.08% | 63.2 | 378.6 |
| sparse, raw (control) | 100% | 100% | 77.64% | 0.380 | 81.25% | 76.5 | 395.9 |

Headline finding: the sparse model is consistently slower than the dense baseline on
both datasets (~5-6x internally, ~3.8-4.2x externally), with accuracy comparable to or
trailing dense rather than improved. Foreground Dice is higher on CPM-17 than on
PanNuke for every configuration, which is the encouraging generalization signal — the
models were not retrained or tuned on CPM-17 at all. Full discussion, figures, and
caveats are in [`PathBonnet_Project_Report.pdf`](PathBonnet_Project_Report.pdf) at
the repo root. A plain markdown version of the results table above is in
[`outputs/results_table.md`](outputs/results_table.md), and the qualitative figures
(dense vs. sparse predictions, representative training samples) are in
[`outputs/figures/`](outputs/figures/).

## 5. Outputs used in the report

- `runs/*/results.json`, `train_log.json`, `vis_internal.png`, `vis_external_consep.png`
- Table columns: kept-pixel %, GT recall of the mask, foreground Dice, per-class Dice, macro nuclei Dice, external (CPM-17) Dice, and forward seconds per image (`t_forward_s`; preprocessing is reported separately as `t_prep_s`)

## 6. Report outline (mirrors the paper)

1. Introduction: dense WSI-scale networks are slow. Nuclei occupy a small share of pixels, so a stain prior gives a sparse input.
2. Method: 2.1 stain-prior thresholding and calibration, 2.2 sparse sliding window and Gaussian fusion (Eq. 1-3), 2.3 sparse U-Net, 2.4 loss (Eq. 5-7).
3. Experiments: datasets, setup, baselines (dense U-Net, `raw` prior), Table 1 (PanNuke fold 3), Table 2 (internal vs. CPM-17 external), qualitative figures.
4. Discussion: sparsity is only worth it if the kept-pixel % is low and mask recall is high. Report both honestly. Hardware and dimensionality (2D/T4 here vs. 3D/A6000 in Bonnet) are not matched, so speedup magnitudes are not directly comparable.
5. Limitations and future work: semantic segmentation only, so add instance separation (e.g. distance-map post-processing). External test is CPM-17, not CoNSeP, and still small (32 images). Use WSI-level tiling with tissue detection, where background gives the real speedup.

## 7. Honest caveats to state in the report

- H&E patches are much denser than CT bone volumes. The `hema` prior at 98% recall keeps a fraction of pixels, but the speedup you measure on PanNuke patches will be smaller than the paper's 25x. Report the measured number. The larger gains should appear on whole slides with white background.
- The threshold ceiling matters: Dice cannot exceed what the mask recall allows. The calibration step reports this.
- The external test is binary because the label sets differ. The internal test is 6-class.
- Bonnet's reported ~25x speedup is 3D-on-A6000; this study's ~5-6x slowdown is 2D-on-T4. Different dimensionality and hardware, not a like-for-like comparison.
- All numbers come from the author's own runs (see `runs/*/results.json` and the report).

## License

MIT (see `LICENSE`). Cite the Bonnet paper, PanNuke, and CPM-17 if you reuse this work.
