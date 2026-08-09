# A heterogeneous deep-learning ensemble for 11-class diagnosis of kidney disease on transmission electron microscopy

Analysis code and de-identified aggregate results for:

> Liu L., Liu J., Zhang Q., Li X., Liu Z., Li J., Wang H., Wang W., Wang H., and Wang J.
> *A heterogeneous deep-learning ensemble for 11-class diagnosis of kidney disease on
> transmission electron microscopy.* Submitted to **Scientific Reports** (major revision).

Locked dataset: 60,512 transmission electron microscopy (TEM) images from 8,306 biopsies
in 8,250 patients across 11 glomerular disease classes, held at a single tertiary
ultrastructural pathology service. Three quality-control failures were excluded, leaving
60,509 images for modelling. Assignment is person-level and stratified (seed 42):
training 5,940 patients, meta-calibration 661 patients, held-out test 1,649 patients.
Every image of a person — including repeat biopsies — falls in exactly one subset.

---

## Privacy and ethics

**This repository contains no patient-level data.**

The TEM images are clinical records and cannot be redistributed. All image-level and
patient-level processing — manifest construction, person-level splitting, cache building,
training, inference, fusion and analysis — ran on the authorised controlled host; only
aggregate JSON and CSV summaries are published here. The tables released are counts,
proportions, distribution summaries, image-fidelity statistics and model metrics.

Specifically, this repository contains no images, no model weights, no prediction
tensors, no per-image or per-patient manifests, and no `PT` archive numbers. `PT` numbers
are pseudonymous patient keys, and the biopsy folder names that carry them
(`PT######_YYYYMMDD`) additionally encode an individual-level date; both are treated as
quasi-identifiers and are blocked from release. It contains no names, no medical record
or admission numbers, no dates of birth, and no contact details.

The gate that enforces this is `PRIVACY_AUDIT.py`, which blocks on file type (raw images,
weights, tensors), on file name (`split.json`, `manifest.json`), on JSON size, and on
content patterns (`PT` numbers, identifiers, credentials, absolute host paths). It must
report zero blocking findings before anything is pushed.

The study was approved by the Clinical Research Ethics Committee of Xiangya Hospital,
Central South University (Approval No. 2025030242) and conducted in accordance with the
Declaration of Helsinki. The requirement for informed consent was waived by the Ethics
Committee given the retrospective use of de-identified archival material.

Requests for controlled access to the underlying images should be directed to the
corresponding author and remain subject to ethics, institutional data-governance and
legal approval.

---

## Layout

| Path | Contents |
|---|---|
| `scripts/config.py` | Single source of truth: class set, paths, split fractions, image settings, backbones, hyperparameters, ablation grid, TTA sweep. |
| `scripts/build_manifest.py` | Enumerates the archive, parses person and biopsy units, writes the (unpublished) image manifest and the published dataset summary. |
| `scripts/make_split.py` | Person-level stratified train / meta-calibration / test split with an explicit leakage self-test. |
| `scripts/build_cache.py` | Full-frame isotropic downsampling to 384 px, JPEG cache, and masking of the bottom annotation strip. |
| `scripts/image_quality.py` | SSIM / PSNR / LPIPS quantification of the downsampling and JPEG steps. |
| `scripts/dataset.py` | Dataset and sampler definitions. |
| `scripts/train_backbone.py` | Backbone training (focal loss, weighted sampling, MixUp/CutMix, AdamW, early stopping, epoch-level resume). |
| `scripts/frozen_baseline.py` | Frozen ImageNet features with conventional classifiers. |
| `scripts/predict.py` | Test-time augmentation sweep and softmax export. |
| `scripts/fusion_analysis.py` | Soft voting, stacking, temperature scaling, referral thresholds, bootstrap CIs, calibration, diversity, McNemar, patient-level aggregation. |
| `scripts/ablation_analysis.py` | Class-imbalance ablation arms. |
| `scripts/gradcam.py` | Grad-CAM panels for correct and misclassified cases. |
| `scripts/profile_io.py` | Throughput profiling with and without the data-governance layer. |
| `scripts/orchestrator.py` | Step sequencing with marker-based resume. |
| `scripts/run_all.sh` | Entry point; starts the orchestrator detached. |
| `scripts/watchdog.sh` | Crash recovery, stall detection, bounded automatic retry. |
| `scripts/hourly_report.sh` | Time-weighted progress reporting. |
| `scripts/push.sh` | Optional notification shim; silently exits when no backend is configured. |
| `aggregate_results/` | De-identified aggregate results. Dataset composition and image-fidelity tables are present; model results are **pending** until the GPU batch completes. |
| `figures/` | Publication figures. **Pending** — generated once model results land. |
| `EXTRACTION_RULES.md` | English reference for the Chinese-language class folders and the archive parsing, quality-control and masking rules. |
| `METHODS.md` | Condensed frozen analysis specification. |
| `data_dictionary.csv` | Variable definitions for the published aggregate tables. |

The class-folder names in `scripts/config.py` are Chinese because the archive is Chinese.
`EXTRACTION_RULES.md` gives each folder name, its English class code and disease name, and
the deterministic rules that turn the archive layout into analysis units.

## Reproducing the reported figures and tables

```bash
python -m pip install -r requirements.txt

# On the controlled host, with the image archive mounted:
export SR_WORKSPACE=/path/to/workspace   # archive root; see scripts/config.py
export SR_PYTHON=/path/to/python         # interpreter used by run_all.sh
bash scripts/run_all.sh                  # full pipeline, detached and resumable
```

`scripts/run_all.sh` starts `scripts/orchestrator.py`, which runs the whole sequence —
manifest, split, cache, image quality, five backbones plus the ResNet-50 baseline,
16-fold test-time augmentation on the meta-calibration and test subsets, six ablation
arms, frozen-feature baseline, fusion, ablation analysis, Grad-CAM and I/O profiling —
recording a marker per step so an interrupted run resumes where it stopped. Individual
steps can also be run directly, for example `python scripts/fusion_analysis.py`, once the
predictions they consume exist.

The pipeline requires the controlled host and the image archive; it cannot be run against
public data. The scripts are included for inspection and audit, and the aggregate outputs
they produce are what this repository publishes.

Absolute paths from the original host have been replaced by the environment variables
`SR_WORKSPACE`, `SR_EXP_DIR`, `SR_DATA_ROOT`, `SR_PYTHON` and `SR_NOTIFY_SCRIPT`; every
other byte of the analysis code is as it ran. No other edit was made for release.

Random seed 42 is fixed throughout: split construction, sampler, augmentation and
bootstrap resampling. Software versions are pinned in `requirements.txt`; the analysis
environment used Python 3.13.9, PyTorch 2.10.0+cu130, timm 1.0.26, NumPy 2.4.2,
SciPy 1.17.1, scikit-learn 1.8.0, scikit-image 0.26.0, Pillow 12.1.1, LPIPS 0.1.4 and
matplotlib 3.10.8, on one NVIDIA GB10 (121 GB, CUDA 13.0).

## Reported dataset composition

| Quantity | Value |
|---|---|
| Classes | IgAN, IgAVN, MCD, MN, MsPGN, LN, EPGN, DKD, TBMN, FSGS, MPGN |
| Patients / biopsies / images | 8,250 / 8,306 / 60,512 |
| Images after quality control | 60,509 (3 unreadable or truncated files excluded) |
| Split unit | Person (`PT` prefix); repeat biopsies of one person stay together |
| Train / meta-calibration / test | 5,940 / 661 / 1,649 patients |
| Class imbalance | 12.5:1 at image level, 14.3:1 at patient level (IgAN vs FSGS) |
| Persons with more than one biopsy | 55, of which 39 carry more than one class label |
| Downsampling fidelity (mean) | SSIM 0.644, PSNR 26.5 dB |
| JPEG cache fidelity (mean) | SSIM 0.986, PSNR 39.4 dB, LPIPS 0.0027 |

Model results — per-class metrics, bootstrap confidence intervals, calibration, fusion,
ablations, referral analysis and Grad-CAM panels — are **not yet in this repository**.
They are written by the pipeline into `aggregate_results/` and `figures/`; see
`aggregate_results/README.md` for what lands there and when.

## Licence

Released under the MIT Licence (see `LICENSE`). The licence covers the code and the
aggregate tables in this repository only; it does not extend to the underlying clinical
images, which are not distributed here.
