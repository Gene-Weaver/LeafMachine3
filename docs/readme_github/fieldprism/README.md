# FieldPrism images for the LeafMachine3 README

This folder holds the figures for the README section on using LeafMachine3 with FieldPrism images.

## Citing FieldPrism

- Website: https://fieldprism.org/
- iOS app: https://apps.apple.com/us/app/fieldprism/id6761267750
- Android app (Google Play testing program): https://play.google.com/apps/testing/com.leafmachine.fieldprism
- Publication: Weaver, W. N., and S. A. Smith. 2023. FieldPrism: A system for creating snapshot vouchers
  from field images using photogrammetric markers and QR codes. *Applications in Plant Sciences* 11(5):
  e11545. doi:[10.1002/aps3.11545](https://doi.org/10.1002/aps3.11545).
  https://bsapubs.onlinelibrary.wiley.com/doi/10.1002/aps3.11545

## Images

Every image here is real LeafMachine3 output from a full pipeline run on the two example FieldPrism
images in `examples/images/`:
- `15_1_FPfit.JPG`: a Letter sheet with all 4 markers readable. Published CF 114.72 px/cm.
- `5_1_FPfit.JPG`: a Letter sheet whose top-left marker is crossed by twigs. Published CF 91.76 px/cm.

| File | What it shows |
|---|---|
| `summary_15_1_FPfit.jpg` | Full Summary overlay for 15_1. FieldPrism markers are shown without detector boxes: each marker's TL, TR, C and BL squares are labeled the way the FieldPrism app labels them, with a green square on the empty BR cell. |
| `summary_15_1_FPfit_topleft.jpg` | Top-left corner of that overlay: the CF banner ("measured from FieldPrism"), the 1 cm / 1 inch raft, and the sheet badge next to it ("FieldPrism Letter \| 4 markers"). |
| `summary_5_1_FPfit.jpg` | Full Summary overlay for 5_1. |
| `summary_5_1_FPfit_topleft.jpg` | Top-left corner of the 5_1 overlay. The twig-crossed marker failed validation, so it was reconstructed from the other three (dashed magenta, "inferred"), and the badge reads "3 + 1 inferred". |
| `fieldprism_overlay_15_1_FPfit.jpg`, `fieldprism_overlay_5_1_FPfit.jpg` | The Overlay_FieldPrism output: a clone of the FieldPrism app's overlay, with the "1 cm = N px" legend and the sheet badge. |
| `qc_marker_15_1_FPfit_valid.png` | Overlay_Ruler_Lattice QC section for a marker that passed every geometric check. |
| `qc_marker_5_1_FPfit_failed_validation.png` | QC section for the twig-crossed 5_1 marker. The apps' square finder sees only 3 squares, and the marker fails the pitch and plateau checks, so it is not used. |
| `qc_sheet_15_1_FPfit.png` | Sheet identification for 15_1: Letter from 4 markers, with the ranked sheet hypotheses and the page schematic. |
| `qc_sheet_5_1_FPfit.png` | Sheet identification for 5_1: Letter from 3 markers plus 1 inferred. |
