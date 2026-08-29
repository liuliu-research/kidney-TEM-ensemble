# Aggregate results

De-identified aggregate outputs. Nothing in this directory is patient-level.

## Present now

| File | Produced by | Contents |
|---|---|---|
| `dataset_stats.json` | `scripts/build_manifest.py` | Per-class patient, biopsy and image counts; images per patient and per biopsy; empty-folder audit. |
| `image_quality.json` | `scripts/strict_oof/image_quality_v2.py` | RGB SSIM, PSNR and LPIPS for resize only, resize plus the bottom-28-row mask, and the actual decoded JPEG cache, overall and per class on the same thirty source TIFF images per class; includes sample errors and cache-byte identity. |

## Completed model and systems outputs

These files were copied only after the core-result validator passed. They contain
aggregate metrics and profiler summaries, never row-level predictions or identifiers.
The release does not include checkpoints, prediction tensors, workflow markers or
raw inference logs, raw per-epoch histories or high-frequency GPU-utilisation samples.
The `train_log_*.json` files are projected run-level summaries. Each embeds the SHA-256
of its unmodified controlled source and records the removed array fields and counts.

| File | Produced by | Contents |
|---|---|---|
| `analysis.json` | `scripts/strict_oof/analyze.py` + local release gate | Headline and per-class metrics with person-clustered bootstrap CIs, Train-OOF-fitted fusion, independent Meta-Cal selection/calibration and prespecified referral assessment, terminal Test TTA sweep, diversity, McNemar and patient-level aggregation. For privacy, the public copy omits per-item prediction thresholds and retains at most 512 deterministic display points per precision-recall curve; its metadata binds the unmodified controlled source by SHA-256. |
| `ablation_analysis.json` | `scripts/strict_oof/analyze_ablation.py` | Nine descriptive class-imbalance and augmentation arms under the common TTA1 evaluation schema. |
| `frozen_baseline.json` | `scripts/frozen_baseline.py` | Frozen ImageNet features with conventional classifiers. |
| `io_profile.json`, `profiler_*.txt` | `scripts/profile_io.py` | Throughput with and without the data-governance layer, in triplicate. |
| `train_log_*.json` | `scripts/strict_oof/train.py` + local release gate | Projected run-level summaries of stopping/terminal epochs, wall-clock time, hyperparameters, seeds, environments and aggregate GPU utilisation for six backbone/comparator runs and nine ablation arms; raw epoch histories and high-frequency GPU samples are removed and source-hash bound. |

## Protocol boundary represented in these aggregates

- Soft-voting weights and the primary logistic-regression stacker are fitted on complete
  Train OOF predictions at TTA8.
- The independent Meta-Cal partition selects between `voting_tta8` and `stack_lr_tta8`
  and fits the selected candidate's global temperature. It also assesses the prespecified
  referral rule; no threshold met the 0.90 target, so no operating threshold was selected.
- The held-out Test partition is used only after those choices are frozen.
- All nine ablation arms are evaluated at TTA1. Their differences are descriptive
  fixed-budget single-run contrasts, not paired or across-seed causal estimates.

## Files that must never be copied here

The run directory also holds outputs that carry quasi-identifiers. They stay on the
controlled host:

| File | Why it is blocked |
|---|---|
| `manifest.json` | One row per image, each naming a biopsy folder `PT######_YYYYMMDD` — pseudonymous patient key plus an individual-level date. |
| `split.json` | Maps every biopsy folder to its subset; same identifiers. |
| `split_stats.json` | Aggregate in most fields, but `cross_class_persons` lists thirty-nine `PT` numbers together with their diagnoses. |
| `cache_report.json` | Aggregate counts, but its error records carry absolute archive paths including `PT` numbers. |
| `gradcam_index.json` | Carries a `patient` field per panel. Publishable only after that field is dropped. |
| `results/gradcam/*.png`, `big/` | Grad-CAM overlays on real frames, model weights, prediction tensors and the JPEG cache. |

`PRIVACY_AUDIT.py` blocks each of these — by file name, by content pattern, by size and by
file type — so an accidental copy fails the gate rather than reaching GitHub. Run it after
filling this directory, and again before pushing.

Selected numbers from `split_stats.json` are safe to quote and appear in `README.md`
(patients per subset, images per subset, counts of multi-biopsy and cross-class people);
what is withheld is the identifier list, not the statistics.
