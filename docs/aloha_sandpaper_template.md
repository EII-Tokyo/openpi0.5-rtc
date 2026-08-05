# ALOHA Sandpaper Template

This workflow builds a local-only, zero-thickness review model for one-piece
sandpaper wraps on the installed handed ALOHA ViperX-300 fingers. It derives
all dimensions from the frozen supplier assembly; the reference photograph is
not used as a dimensional source.

## Current review contract

- One continuous piece per finger, with separate left and right outputs.
- Full main inward profile plus four adjacent longitudinal panels.
- Four CAD-derived folds and no overlap tabs.
- A balanced zero-width relief cut resolves the otherwise overlapping inner
  panels.
- Material thickness, adhesive thickness, bend compensation, edge clearance,
  DXF, PDF, and final print approval remain intentionally pending.

The supplier STEP license is `UNKNOWN_HARD_BLOCKER` for redistribution. The
source and all generated FCStd/SVG/PNG geometry therefore remain under the
ignored `.codex/artifacts/` tree and must not be committed or redistributed.
Only the generic generator, contracts, tests, and documentation are tracked.

## Generate the first review

Use the pinned project-local FreeCAD 1.1.1 wrapper:

```bash
REVIEW_DIR="$PWD/.codex/artifacts/<review-directory>"
ALOHA_SANDPAPER_STEP="$PWD/.codex/artifacts/20260729-aloha-finger-palm-orientation/gdrive_source_readonly/Simple Aloha Viper 2024-5-13.step" \
ALOHA_SANDPAPER_OUTPUT_DIR="$REVIEW_DIR" \
  local_tools/freecad-tessellation/freecadcmd \
  tools/aloha1_mapping/build_sandpaper_template_freecad.py

.venv/bin/python tools/render_aloha_sandpaper_template_review.py \
  "$REVIEW_DIR/aloha_sandpaper_zero_thickness_review.json" \
  --output-dir "$REVIEW_DIR"
```

The builder rejects a changed source hash, writable source CAD, wrong
FreeCAD/OCCT versions, changed handed objects or face topology, nonplanar
panels, failed unfold alignment, and residual flat-pattern overlap.

## Review colors

- Orange: complete main inward profile.
- Green: the two outer longitudinal wrap panels.
- Blue: the two balanced inner wrap panels.
- Blue dashed: the four CAD-derived fold lines.
- Red: the zero-width inner relief cut.

The generated SVG files are physically sized in millimetres on A4, but are
watermarked `NOT FINAL PRINT TEMPLATE`. Do not use them as the final sandpaper
cutting template until the wrapped geometry is approved and the combined
sandpaper-plus-adhesive thickness is measured.
