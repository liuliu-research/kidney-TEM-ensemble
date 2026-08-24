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

The original Train partition is further divided into five patient-grouped outer folds.
Within each outer-training portion, a patient-grouped inner validation split is used for
early stopping and epoch selection; the outer-heldout fold is used only to generate that
fold's out-of-fold (OOF) predictions. The five outer folds therefore provide complete
Train OOF predictions without using the independent Meta-Cal or Test partitions for
backbone training or epoch selection.

Soft-voting weights and the primary logistic-regression stacking meta-learner are fitted
on the complete Train OOF predictions at TTA8. The independent Meta-Cal partition is then
used to select between the two pre-specified candidates (`voting_tta8` and
`stack_lr_tta8`), fit the selected candidate's global temperature, and select the
referral threshold. These choices are frozen before the held-out Test partition is used
once for terminal evaluation.

## Image preprocessing

All source rows and columns participate in Lanczos resampling to 384 x 384; no spatial
crop precedes resizing. The bottom 28 rows (7.29% of the resized image area) are then
masked to black before quality-92 JPEG caching. This fixed mask reduces exposure to
commonly located footer metadata, but it does not prove that scale text or measurement
marks outside the strip are absent. Stage-wise fidelity is quantified on the same thirty
source TIFF images per class after resize only, after resize plus masking, and after
decoding the actual cached JPEG. RGB SSIM, PSNR and LPIPS are reported for all three
cumulative comparisons, together with the incremental JPEG-only comparison; regenerated
JPEG bytes must match the cache used by the dataset.
Files that fail to decode are excluded and counted.

## Models and training

Five heterogeneous ImageNet-pretrained backbones are fine-tuned: EfficientNet-B4,
ViT-B/16, Swin-B, ConvNeXt-B and DeiT-B, all at 384 pixels. A supervised ResNet-50 and a
frozen-feature classifier over ImageNet representations serve as baselines.

Training uses focal loss with gamma 1.5 and label smoothing 0.05, class-weighted
sampling, MixUp (alpha 0.2) and CutMix (alpha 1.0) applied with probability 0.5, AdamW at
base learning rate 1e-4 with a 0.1 multiplier on backbone parameters, weight decay 1e-4,
a 10 % warm-up fraction, bfloat16 mixed precision, batch size 32 (40 for EfficientNet-B4),
and at most 40 outer-fold epochs. Outer-fold early stopping uses only the inner-validation
macro-F1 with patience 8. For each backbone and ResNet-50, the terminal epoch count is the
median of its five outer-fold best epochs; the model is then retrained on the complete
Train partition for that fixed count, without early stopping. The scheduler retains its
40-epoch horizon. In total, the strict workflow contains 30 outer-fold and six terminal
training runs. Checkpoints are written every epoch so an interrupted run resumes at the
epoch boundary rather than from scratch.

The ablation analysis contains nine arms under a common 30-epoch budget: plain
cross-entropy; cross-entropy plus weighted sampling; focal loss only; focal loss plus
sampling; the full recipe; the full recipe plus class-balanced re-weighting;
class-balanced re-weighting only; MixUp only; and CutMix only. The original six
contract-bound arms and three additional isolated runs use the same selected ablation
backbone and are all evaluated at TTA1. They are descriptive single-run contrasts, not
paired or across-seed estimates of component effects. The three additional arms retain
the original ablation convention of Train fitting with Meta-Cal monitoring; they do not
enter the main fusion-candidate selection.

## Inference, fusion and calibration

Outer-fold models generate complete Train OOF predictions at TTA8. Terminal models,
trained on the complete Train partition at their fixed epoch counts, generate Meta-Cal
and Test predictions for deterministic TTA depths 1, 2, 4, 8 and 16. The pre-specified
primary candidates are soft voting and logistic-regression stacking at TTA8; both the
voting weights and logistic-regression coefficients are fitted on complete Train OOF
predictions. Their independent Meta-Cal macro-F1 selects the final candidate. A single
global temperature is fitted to the selected candidate's independent Meta-Cal
probabilities, and the referral threshold is selected on those temperature-scaled
Meta-Cal probabilities. Multilayer-perceptron and gradient-boosted-tree stackers are
fitted on Train OOF TTA1 features and reported only as post-hoc descriptive sensitivity
analyses. Per-class temperature scaling is likewise a sensitivity analysis; the primary
system uses the global temperature.

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
Grad-CAM panels are qualitative: the bottom strip is masked, but scale text or in-image
measurement marks can remain outside it, and panels must be
inspected for activation tracking those marks before any interpretive claim is made.
