# Aggregate results

De-identified aggregate outputs. Nothing in this directory is patient-level.

## Present now

| File | Produced by | Contents |
|---|---|---|
| `dataset_stats.json` | `scripts/build_manifest.py` | Per-class patient, biopsy and image counts; images per patient and per biopsy; empty-folder audit. |
| `image_quality.json` | `scripts/image_quality.py` | SSIM, PSNR and LPIPS for the downsampling and JPEG-cache steps, overall and per class, thirty sampled images per class. |

## Pending — filled in after the GPU batch completes

The training and inference batch was still running when this repository was staged. The
files below are written by the pipeline into the run directory `results/`, and are copied
here by the release step of the project RUNBOOK ("retrieve real results", then rebuild
figures and manuscript) once `results/markers/ALL.done` exists.

| File | Produced by | Contents |
|---|---|---|
| `analysis.json` | `scripts/fusion_analysis.py` | Headline and per-class metrics with person-clustered bootstrap CIs, TTA sweep, soft voting and stacking, temperature scaling, calibration, diversity, McNemar, referral thresholds, patient-level aggregation. |
| `ablation_analysis.json` | `scripts/ablation_analysis.py` | The six class-imbalance ablation arms. |
| `frozen_baseline.json` | `scripts/frozen_baseline.py` | Frozen ImageNet features with conventional classifiers. |
| `io_profile.json`, `profiler_*.txt` | `scripts/profile_io.py` | Throughput with and without the data-governance layer, in triplicate. |

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
