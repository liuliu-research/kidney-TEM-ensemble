# Figures

This directory contains eight main and six supplementary publication figures as
600-dpi PNG and vector/raster PDF exports. Every file is hash-bound to the Codex visual
QA record used by the local release gate; this is not author confirmation.

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

`PRIVACY_AUDIT.py` accepts only the exact 28 PNG/PDF paths and SHA-256 values recorded
in `privacy_policy.json`; missing, extra or byte-drifted figures fail closed. The hashes
originate from the Codex visual QA record checked by the local release gate. Authors must
still complete their own privacy and scientific-interpretation confirmation before upload.
