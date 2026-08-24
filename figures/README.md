# Figures

This directory contains the three main and eleven supplementary publication figures as
600-dpi PNG and vector/raster PDF exports. Every file is hash-bound to the external
visual-review sign-off used by the local release gate.

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
