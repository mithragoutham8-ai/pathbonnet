# PathBonnet — final results

Internal test: PanNuke fold 3 (2,722 images). External test: CPM-17 (32 images, no fine-tuning).

| Run | Model | Prior | Kept px % | GT recall % | PanNuke fg Dice | Nuclei macro Dice | Neo | Inf | Con | Dead | Epi | External (CPM-17) fg Dice | Internal fwd ms/img | External fwd ms/img |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense_raw | dense | raw | 100.00 | 100.00 | 79.72 | 38.54 | 64.81 | 45.79 | 41.36 | 0.00 | 40.75 | 83.55 | 12.2 | 94.3 |
| sparse_hema | sparse | hema | 64.42 | 97.73 | 78.09 | 35.48 | 64.47 | 39.45 | 40.76 | 0.00 | 32.72 | 82.26 | 60.1 | 361.8 |
| sparse_raw | sparse | raw | 100.00 | 100.00 | 77.64 | 38.00 | 65.51 | 45.83 | 41.13 | 0.00 | 37.54 | 81.25 | 76.5 | 395.9 |
| sparse_tissue | sparse | tissue | 68.97 | 97.78 | 76.30 | 36.92 | 63.68 | 40.45 | 40.48 | 0.00 | 39.97 | 80.08 | 63.2 | 378.6 |

**Headline:** sparse is consistently slower than dense on both datasets (~5-6x internally, ~3.8-4.2x externally), with accuracy comparable to or trailing dense rather than improved. Foreground Dice is higher on CPM-17 than on PanNuke for every configuration — an encouraging generalization signal, since none of the models were retrained or tuned on CPM-17.

Full discussion, method, and figures are in `PathBonnet_Project_Report.pdf` at the repo root. Qualitative figures referenced there are in `outputs/figures/`.
