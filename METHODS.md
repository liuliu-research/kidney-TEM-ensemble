# Frozen analysis specification

## Cohort and unit of analysis

- Source of truth: the curated TEM image archive of a single tertiary ultrastructural
  pathology service, v2.0 freeze (16 June 2026).
- Classes: IgAN, IgAVN, MCD, MN, MsPGN, LN, EPGN, DKD, TBMN, FSGS and MPGN — the
  eleven-class full-power set. Four rare categories present in the archive (AMY, AAGN,
  HN and unclassified material) are outside this class set and are not modelled.
- Archive unit: one folder per biopsy, named `PT######_YYYYMMDD`. The `PT` prefix
  identifies the person; a person may hold more than one biopsy folder.
- Analysis unit for splitting: the **person**. All images of all biopsies of one person
  fall in exactly one subset. Fifty-five people hold more than one biopsy and thirty-nine
  of those carry more than one class label; such people are stratified by their majority
  diagnosis but are never divided across subsets.
- Analysis unit for the primary metrics: the image. Patient-level aggregation is reported
  as a secondary analysis.
- Scale: 8,250 people, 8,306 biopsies, 60,512 images; 60,509 images after quality
  control.

## Split

Person-level stratified three-way split at seed 42: 72 % training, 8 % meta-calibration,
20 % test — 5,940 / 661 / 1,649 people. A leakage self-test asserts that the three person
sets are pairwise disjoint before the split is written.

The meta-calibration subset is the only data used to set soft-voting weights, to fit the
stacking meta-learner, to fit the temperature, and to choose the referral threshold. The
test subset is touched once, at final evaluation.

## Image preprocessing

Whole frames are downsampled isotropically to a 384-pixel short side with Lanczos
resampling; no cropping is applied, so no field of view is discarded. The bottom
28 pixels of every frame — the microscope annotation strip — are masked to black before
caching, which removes the printed magnification and scale text as a possible shortcut.
Frames are cached as quality-92 JPEG. Downsampling and JPEG fidelity are quantified
against the originals with SSIM, PSNR and LPIPS on thirty sampled images per class.
Files that fail to decode are excluded and counted.

## Models and training

Five heterogeneous ImageNet-pretrained backbones are fine-tuned: EfficientNet-B4,
ViT-B/16, Swin-B, ConvNeXt-B and DeiT-B, all at 384 pixels. A supervised ResNet-50 and a
frozen-feature classifier over ImageNet representations serve as baselines.

Training uses focal loss with gamma 1.5 and label smoothing 0.05, class-weighted
sampling, MixUp (alpha 0.2) and CutMix (alpha 1.0) applied with probability 0.5, AdamW at
base learning rate 1e-4 with a 0.1 multiplier on backbone parameters, weight decay 1e-4,
a 10 % warm-up fraction, bfloat16 mixed precision, batch size 32 (40 for EfficientNet-B4),
at most 40 epochs, and early stopping on meta-calibration macro-F1 with patience 8.
Checkpoints are written every epoch so an interrupted run resumes at the epoch boundary
rather than from scratch.

Six class-imbalance ablation arms are trained on Swin-B under a common 30-epoch budget:
plain cross-entropy; plus weighted sampling; focal only; focal with sampling; the full
recipe; and the full recipe with class-balanced re-weighting.

## Inference, fusion and calibration

Deterministic test-time augmentation is swept at 1, 2, 4, 8 and 16 folds. Fusion
compares soft voting with meta-calibration-derived weights against stacked
generalisation with a logistic-regression meta-learner, with multilayer-perceptron and
gradient-boosted-tree meta-learners as non-linear controls. Temperature scaling is fitted
on the meta-calibration subset, globally and per class.

## Evaluation

Reported metrics are accuracy, balanced accuracy, macro-F1, weighted F1, macro one-vs-rest
AUC, macro average precision, MCC and Cohen's kappa, with per-class recall, precision, F1
and average precision. Uncertainty comes from person-clustered bootstrap resampling, so
that repeat biopsies of one person resample together. Calibration is reported as Brier
score, negative log-likelihood, per-class and balanced expected calibration error, and
reliability diagrams. Ensemble diversity is reported as pairwise backbone correlation,
disagreement rate and mutual information. The ensemble is compared against the best single
backbone by McNemar's paired test. A referral analysis sweeps confidence thresholds from
0.5 to 0.9 and reports the disease composition of referred cases.

Throughput is profiled with and without the data-governance layer, in triplicate, with the
PyTorch profiler.

## Validation boundary

The referral analysis is a retrospective triage simulation on archival images; it is not a
prospective clinical evaluation and does not establish clinical utility. All images come
from one centre and one electron-microscopy service, so no external validation is claimed.
Grad-CAM panels are qualitative: the printed annotation strip is masked, but in-image
measurement marks drawn during routine reporting cannot be removed, and panels must be
inspected for activation tracking those marks before any interpretive claim is made.
