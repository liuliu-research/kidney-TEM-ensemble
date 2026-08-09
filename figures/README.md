# Figures

**Pending.** The publication figures are generated from `aggregate_results/` once the GPU
batch completes, by the figure step of the project RUNBOOK, and are copied here at 300 dpi
as PNG and PDF.

Planned set: cohort and class composition; per-class performance with confidence
intervals; confusion matrix; calibration and reliability; TTA and fusion comparison;
ablation arms; referral threshold sweep; Grad-CAM panels — plus the supplementary figures.

## Before adding anything here

Grad-CAM panels are overlays on real electron-microscopy frames. Two checks are mandatory
and neither can be automated:

1. **No identifiers in the panel.** The bottom annotation strip is masked during caching,
   but confirm no case number, date or instrument text survives anywhere in the frame, and
   that no filename or title text carries a `PT` number.
2. **Activation is not tracking an annotation.** Measurement marks drawn inside the frame
   during routine reporting cannot be removed. If the highlighted region follows such a
   mark rather than tissue, the panel must not be presented as evidence of morphological
   attention, and the manuscript text must say so.

`PRIVACY_AUDIT.py` blocks raster images everywhere except this directory, and flags every
file placed here for manual sign-off. A flag here is not a failure; it is the reminder that
these two checks are owed.
