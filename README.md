# A heterogeneous deep-learning ensemble for assistive 11-class classification of kidney disease on transmission electron microscopy

Analysis code and de-identified aggregate results for:

> Liu L., Liu J., Zhang Q., Li X., Liu Z., Li J., Wang H., Wang W., Wang H., and Wang J.
> *A heterogeneous deep-learning ensemble for assistive 11-class classification of
> kidney disease on transmission electron microscopy.* Submitted to **Scientific
> Reports** (major revision).

## Permanent archive

Version **v1.0.0** was released and archived on 14 October 2026:
[Zenodo DOI 10.5281/zenodo.23354305](https://doi.org/10.5281/zenodo.23354305),
[GitHub Release](https://github.com/liuliu-research/kidney-TEM-ensemble/releases/tag/v1.0.0).
The archive preserves commit `171c2f9c2f5de6fd24902f17abefa027e34b5b56`.
All 109 archived files were downloaded and verified against that commit. Its historical
repository-state metadata describes the status at snapshot creation; this live README,
the citation metadata and the Zenodo record document the subsequently completed archive.
Post-archive metadata updates do not change the frozen code, aggregate results or figures.

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
aggregate JSON and CSV summaries, together with visually reviewed, de-identified
publication figures, are published here. The tables released are counts,
proportions, distribution summaries, image-fidelity statistics and model metrics.

Specifically, this repository contains no redistributable raw images, no model weights,
no prediction tensors, no per-image or per-patient manifests, and no `PT` archive
numbers. It also contains no step markers, raw per-epoch histories or high-frequency
GPU-utilisation samples. The `train_log_*.json` files are projected run-level summaries
that retain aggregate timing, stopping, seed, environment and hyperparameter fields.
Source-microscopy content appears only in visually reviewed, de-identified Grad-CAM
publication panels. `PT` numbers
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
| `scripts/strict_oof/run_all.sh` | Sole strict-continuation orchestration entry after the validated legacy prerequisite snapshot has been materialised. |
| `scripts/strict_oof/workflow.py` | Frozen 94-step plan: patient-grouped Train OOF, terminal retraining, inference, nine-arm ablation, analysis and validation. |
| `scripts/strict_oof/protocol.py` | Protocol constants and fail-closed contracts for partitions, folds, TTA depths and result lineage. |
| `scripts/strict_oof/train.py` | Outer-fold, terminal and additional-ablation training with checkpoint validation and exact resume. |
| `scripts/strict_oof/predict.py` | Train OOF TTA8 and terminal Meta-Cal/Test TTA16 inference with hash-bound shards. |
| `scripts/strict_oof/analyze.py` | Train-OOF fusion fitting, independent Meta-Cal selection/calibration and prespecified referral assessment, followed by terminal Test analysis. |
| `scripts/strict_oof/analyze_ablation.py` | Combines the six retained arms and three additional arms under a common TTA1 evaluation schema. |
| `scripts/strict_oof/validate.py` | Deep, fail-closed validation of protocol, artifacts, probabilities and aggregate analysis. |
| `scripts/config.py`, `build_manifest.py`, `make_split.py`, `build_cache.py`, `image_quality.py` | Shared frozen cohort, split, preprocessing and image-fidelity dependencies used by the strict workflow. |
| `scripts/frozen_baseline.py`, `profile_io.py`, `validate_core_results.py` | Contract-bound comparator, systems summary and compatibility validation dependencies. |
| `scripts/run_all.sh`, `orchestrator.py`, `train_backbone.py`, `predict.py`, `fusion_analysis.py`, `ablation_analysis.py`, `gradcam.py` | Legacy prerequisite workflow retained for provenance; it generates most prerequisite artifacts but does not publish the complete prerequisite receipt or the revised strict results by itself. |
| `aggregate_results/` | Completed, de-identified aggregate results used for the revision. |
| `figures/` | Publication figure exports in PNG and PDF; all files are hash-bound to the visual-review sign-off. |
| `EXTRACTION_RULES.md` | English reference for the Chinese-language class folders and the archive parsing, quality-control and masking rules. |
| `METHODS.md` | Condensed frozen analysis specification. |
| `data_dictionary.csv` | Variable definitions for the published aggregate tables. |

The class-folder names in `scripts/config.py` are Chinese because the archive is Chinese.
`EXTRACTION_RULES.md` gives each folder name, its English class code and disease name, and
the deterministic rules that turn the archive layout into analysis units.

## Reproducing the reported figures and tables

```bash
python -m pip install -r requirements.txt

# On the controlled host, after restoring and validating the legacy prerequisite snapshot:
export SR_DATA_ROOT=/path/to/restricted/archive
export SR_EXP_DIR="$PWD/scripts"        # controlled experiment root
export SR_PYTHON=/path/to/python
(cd scripts && "$SR_PYTHON" -m strict_oof.orchestrator --dry-run)
# Inspect preflight_ready/preflight_blockers; this command never starts training.
bash scripts/strict_oof/run_all.sh        # strict continuation, not raw-data bootstrap
```

`scripts/strict_oof/run_all.sh` is the frozen strict-continuation launcher, not a raw-data
bootstrap. It requires a validated legacy prerequisite snapshot containing the restricted
patient-level manifest/split and cache, legacy aggregate analysis and ablation inputs, and
the machine-verifiable core-validation markers. Those private/run-control artifacts are
withheld from this public repository. Within the original
Train partition it creates patient-grouped five-fold out-of-fold (OOF) predictions, using
only an inner patient-grouped validation split for early stopping. Soft-voting weights
and the primary logistic-regression stacker are fitted on the complete Train OOF
predictions at TTA8. The independent Meta-Cal partition then selects between the two
pre-specified fusion candidates and fits the global temperature. The prespecified referral
rule is assessed on Meta-Cal; no threshold in {0.5, ..., 0.9} reaches autonomous accuracy
0.90, so no operating threshold is selected. Only after these choices are frozen is the
held-out Test partition used in terminal analysis. The same
workflow evaluates all nine ablation arms at TTA1 and performs deep artifact and
lineage validation before publishing its ready pointer.

The parent-level `scripts/run_all.sh` and its associated training, prediction and fusion
modules generated most of the legacy prerequisite snapshot. They remain inspectable for
provenance, but they neither publish the complete prerequisite validation receipt nor
reproduce the revised strict analysis by themselves.

The pipeline requires the controlled host and the image archive; it cannot be run against
public data. The scripts are included for inspection and audit, and the aggregate outputs
they produce are what this repository publishes.

The scientific scripts are synchronised from the validated active workflow. Host-only
defaults are replaced by `SR_WORKSPACE`, `SR_EXP_DIR`, `SR_DATA_ROOT` and `SR_PYTHON`.
One release-only protocol descriptor changes `referral-threshold fitting` to the accurate
non-numeric wording `prespecified referral-rule assessment`; no numeric setting changes.
Every changed source byte, its controlled and public SHA-256, and its classification are
recorded in `RELEASE_TRANSFORMATIONS.json`; the executed source tree remains bound in the
controlled strict contract.

The partition uses base seed 42. Training seeds are deterministic offsets by model,
fold, stage and ablation arm; analysis resampling uses separately recorded derived
seeds. Every run records the seed that produced it.

Software versions are pinned in `requirements.txt`. The controlled runs used Python
3.13.9, PyTorch 2.10.0+cu130, timm 1.0.26, NumPy 2.4.2, SciPy 1.17.1,
scikit-learn 1.8.0, scikit-image 0.26.0, LPIPS 0.1.4 and matplotlib 3.10.8 on one
NVIDIA GB10 (121 GB, CUDA 13.0). The imported `PIL` module reported 12.1.1, whereas
installed-distribution metadata recorded Pillow 12.2.0; that mismatch is retained in
the audit records. The clean release pin remains Pillow 12.1.1 to match the imported
runtime implementation, and every new run must freeze a fresh, internally consistent
environment. The cache-byte audit passed 330/330 comparisons with zero mismatch.

## Reported dataset composition

| Quantity | Value |
|---|---|
| Classes | IgAN, IgAVN, MCD, MN, MsPGN, LN, EPGN, DKD, TBMN, FSGS, MPGN |
| Patients / biopsies / images | 8,250 / 8,306 / 60,512 |
| Images after quality control | 60,509 (3 unreadable or truncated files excluded) |
| Split unit | Person (`PT` prefix); repeat biopsies of one person stay together |
| Train / meta-calibration / test | 5,940 / 661 / 1,649 patients |
| Class imbalance | 12.5:1 at image level (IgAN vs FSGS), 14.3:1 at patient level (IgAN vs MPGN) |
| Persons with more than one biopsy | 55, of which 39 carry more than one class label |
| Image-fidelity audit | Stage-wise resize-only, resize-plus-mask and actual-cache results are in `aggregate_results/image_quality.json` |
| Deliberately masked area | Bottom 28/384 rows (7.29%); a shortcut-risk mitigation, not proof that all metadata are absent |

Model results — per-class metrics, bootstrap confidence intervals, calibration, fusion,
ablations, referral analysis and Grad-CAM panels — are provided in
`aggregate_results/` and `figures/`. Only aggregate outputs on the explicit release
allowlist are included.

## Licence

Released under the MIT Licence (see `LICENSE`). The licence covers the code and the
aggregate tables in this repository only; it does not extend to the underlying clinical
images, which are not distributed here.
