# Phase 2 — A Strong Dense Baseline: DeepLabV3+ on NuInsSeg

Phase 2 of PathBonnet establishes a reference point from a modern,
high-capacity dense segmentation network, against which the sparse approach
of Phase 1 (and any future variant) can be measured. A DeepLabV3+ network
with a ResNet-101 encoder (45.7M parameters, ImageNet-pretrained) was trained
on the NuInsSeg benchmark using the dataset authors' five-fold split scheme
(seed 19): folds 0–2 for training (399 images), fold 3 for validation (133),
fold 4 as the held-out test set (133).

One curation step matters: the public archive stores each image several
times over (7,315 files for 665 distinct images). The loader deduplicates by
filename stem and partitions over the 665 unique stems, so no tissue image
can appear in two subsets. Ambiguous-area pixels (regions the annotators
left undecided) are excluded from the loss and the metrics.

## Results (single fold — the preserved run)

| Measure | Value |
|---|---|
| Validation foreground Dice (fold 3, n = 133) | 0.836 |
| Test foreground Dice (fold 4, n = 133) | 0.837 |
| Inference time (Tesla T4) | 58.2 ms per image |

For context, the best figure in the dataset paper's own published benchmark
is a Dice of 0.814 (a five-fold mean); completing the full five-fold
cross-validation of this pipeline is the first item of future work. The
semantic output does not separate touching nuclei, so instance-level metrics
(AJI, PQ) are not reported for this phase.

Full details, discussion and limitations are in the project report:
**PathBonnet_Phase2_Report.pdf** (in this folder).

## Contents

- `PathBonnet_Phase2_Report.pdf` — the Phase 2 project report
- `train_deeplabv3plus.py` — training script (NuInsSeg and PanNuke)
- `evaluate.py` — checkpoint evaluation (foreground Dice + ms/image)
- `requirements.txt` — dependencies (segmentation-models-pytorch 0.5.0)
- `results/train_log.json` — per-epoch training loss and validation Dice
- `results/eval_results.json` — test-fold metrics of the reported run
- `figures/` — pipeline diagram and training curves used in the report

## Reproduce

```bash
pip install -r requirements.txt
python train_deeplabv3plus.py --dataset nuinsseg --kaggle_dir /path/to/nuinsseg --out runs/dl3p_nuinsseg
python evaluate.py --ckpt runs/dl3p_nuinsseg/best.pt
```

The trained checkpoint (`best.pt`, ~175 MB) is not stored in this repository;
it is preserved with the Kaggle notebook's saved-version outputs.
