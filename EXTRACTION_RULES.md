# Archive parsing rules

The image archive is organised in Chinese, because the source service is Chinese. This
file is the English reference for `scripts/config.py` and `scripts/build_manifest.py`:
every class folder, its English code, and the deterministic rules that turn the folder
tree into analysis units. No patient-identifying value appears here; `PT######` below is
a placeholder for the pseudonymous archive number, never a real one.

## Directory grammar

```
<SR_DATA_ROOT>/<class folder>/电镜图片/PT######_YYYYMMDD/*.tif|*.jpg|*.png
                              └ "TEM images"  └ biopsy folder
```

- `SR_DATA_ROOT` is set by environment variable; the released code carries no host path.
- `电镜图片` means "electron microscopy images" and is the fixed sub-folder name.
- A biopsy folder is `PT` + archive number + `_` + biopsy date in `YYYYMMDD`.
- **Person key** = the `PT######` prefix, everything before the first underscore.
  Two folders sharing a prefix are two biopsies of the same person.
- Accepted image extensions: `.tif`, `.tiff`, `.TIF`, `.TIFF`, `.jpg`, `.jpeg`, `.png`.

Both the person key and the biopsy folder name are quasi-identifiers — the key is a
pseudonymous patient identifier and the suffix is an individual-level date — so neither is
published. `PRIVACY_AUDIT.py` blocks them by pattern.

## Class folders

| Code | Archive folder (Chinese) | Disease |
|---|---|---|
| IgAN | `IgAN-IgA肾病` | IgA nephropathy |
| IgAVN | `IgAVN-IgA血管炎相关性肾炎（紫癜性肾炎）` | IgA vasculitis-associated nephritis (Henoch–Schönlein purpura nephritis) |
| MCD | `MCD-微小病变病` | Minimal change disease |
| MN | `MN-膜性肾病` | Membranous nephropathy |
| MsPGN | `MsPGN-系膜增生性肾小球肾炎（非IgA）` | Mesangial proliferative glomerulonephritis (non-IgA) |
| LN | `LN-狼疮肾炎` | Lupus nephritis |
| EPGN | `EPGN-毛细血管内增生性肾小球肾炎` | Endocapillary proliferative glomerulonephritis |
| DKD | `DKD-糖尿病肾脏病` | Diabetic kidney disease |
| TBMN | `TBMN-薄基底膜肾病与Alport综合征（IV型胶原相关肾病）` | Thin basement membrane nephropathy and Alport syndrome (type IV collagen nephropathies) |
| FSGS | `FSGS-局灶节段性肾小球硬化` | Focal segmental glomerulosclerosis |
| MPGN | `MPGN-膜增生性肾小球肾炎（IC-MPGN）` | Membranoproliferative glomerulonephritis (immune-complex type) |

The mapping is exact-match on folder name; a folder outside this table is not enumerated,
which is how the four out-of-scope categories in the archive are excluded without any
per-case judgement.

## Quality control

Applied in `scripts/build_cache.py` while writing the cache:

1. A file that Pillow cannot identify is excluded and its error recorded.
2. A file that Pillow reports as truncated is excluded and its error recorded.
3. Empty biopsy folders are counted; none were found at the v2.0 freeze.

Three of 60,512 files failed (one unidentifiable TIFF, two truncated JPEGs), leaving
60,509 images. The per-file exclusion record is kept on the controlled host and is not
published, because it is a list of file paths containing archive numbers; only the counts
appear here and in `aggregate_results/`.

## Annotation masking

Routine electron-microscopy frames carry a printed strip along the bottom edge with
magnification, scale bar and instrument text. Because that text correlates with
acquisition habits and could act as a shortcut, the bottom 28 pixels of every frame are
set to black before caching (`ANNOT_STRIP_PX = 28`). This is applied uniformly to every
image of every class, so it cannot itself introduce a class-dependent artefact.

Measurement marks drawn inside the frame during routine reporting — lines and numerals
placed by the reporting pathologist — cannot be removed without destroying image content.
They remain, and Grad-CAM panels must be inspected for activation that tracks them before
any interpretive claim is made.

## Split rules

Implemented in `scripts/make_split.py`:

1. Group all biopsies by person key.
2. Give each person one stratification label: the class contributing the largest number of
   that person's images. For a person whose repeat biopsies disagree, this is the majority
   diagnosis by image count.
3. Within each class, shuffle the people with a seed-42 generator and cut at the rounded
   20 % and 8 % boundaries, giving test, meta-calibration and training sets of people.
4. Let every biopsy folder inherit its person's subset, and every image its folder's.
5. Assert three things: the three person sets are pairwise disjoint; every folder's subset
   equals its person's subset; and no multi-biopsy person straddles two subsets.

Step 5 is the leakage guard. It runs before the split is written, so a split that could
place one person on both sides of the evaluation boundary never reaches disk. Because the
stratification label is per person rather than per biopsy, the per-class proportions are
approximate for the thirty-nine people whose biopsies disagree — the alternative, splitting
those people across subsets, would reintroduce exactly the leakage this guards against.
