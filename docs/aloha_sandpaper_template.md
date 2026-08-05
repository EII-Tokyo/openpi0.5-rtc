# ALOHA Sandpaper Template

This workflow builds a local-only, zero-thickness review model for one-piece
sandpaper wraps on the installed handed ALOHA ViperX-300 fingers. It derives
all dimensions from the frozen supplier assembly; the reference photograph is
not used as a dimensional source.

## Current review contract

- One continuous piece per finger, with separate left and right outputs.
- Full main inward profile plus two outer longitudinal wrap panels.
- Two CAD-derived folds and no overlap tabs.
- The inner wrap panels are intentionally excluded, so no inner relief cut is
  present.
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
- Blue dashed: the two CAD-derived outer fold lines.

The generated SVG files are physically sized in millimetres on A4, but are
watermarked `NOT FINAL PRINT TEMPLATE`. Do not use them as the final sandpaper
cutting template until the wrapped geometry is approved and the combined
sandpaper-plus-adhesive thickness is measured.

## Export the approved zero-thickness print templates

After the wrapped geometry is approved and the user explicitly accepts the
very thin material as a zero-thickness approximation, export the local-only
1:1 PDF and millimetre DXF files with:

```bash
PRINT_DIR="$PWD/.codex/artifacts/<print-directory>"
.venv/bin/python tools/aloha1_mapping/export_sandpaper_print_templates.py \
  "$REVIEW_DIR/aloha_sandpaper_zero_thickness_review.json" \
  --output-dir "$PRINT_DIR" \
  --approved-zero-thickness
```

Each A4 PDF contains one left- or right-finger template and a 50 x 50 mm
calibration square. Print with `Actual Size` / `100%`; disable fit, shrink,
and scale-to-page options. Measure the calibration square after printing and
reject the print if either side is not 50.0 mm.

Each DXF declares `$INSUNITS=4` (millimetres) and separates geometry into:

- `CUT`: the closed external and internal cutting contours;
- `FOLD`: the two outer fold lines;
- `REFERENCE`: a non-cutting 50 x 50 mm calibration square.

The final export manifest records the source review hash, dimensions, file
hashes, scale, zero bend compensation, and the user-approved material
assumption. Generated PDF/DXF files remain local-only while the supplier CAD
redistribution license is unresolved.

### Distal-only root-cut variant

To connect the two fold lines at their base-side endpoints, discard all
material toward the finger base, and retain only the tipward portion, add:

```bash
  --distal-only-at-fold-root
```

The exporter proves the finger tip has the greater CAD length coordinate,
constructs the root-cut line from the two lower-length fold endpoints, and
keeps only that line's greater-length half-plane. Because the original inner
opening intersects the root-cut line, the final retained material is exported
as one continuous closed cut contour: the root cut appears as two collinear
boundary segments separated by the now-open inner opening. The manifest
records the exact root-cut coordinates and the kept/discarded side semantics.
