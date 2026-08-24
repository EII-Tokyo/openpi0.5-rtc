from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import FreeCAD as App
import MeshPart
import Part


SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[4]
DEFAULT_ROOT = SCRIPT.parents[1]
SOURCE_BOTTLE = REPO_ROOT / "assets/bottle_500ml/cad/bottle_500ml.FCStd"

# Geometry is derived from the bottle CAD neck dimensions.  Values that are
# not present in the source CAD are explicitly diagnostic, not measurements.
CAP_HEIGHT_MM = 22.0
TOP_THICKNESS_MM = 2.0
THREAD_RADIAL_CLEARANCE_MM = 0.4
LINEAR_DEFLECTION_MM = 0.05
ANGULAR_DEFLECTION_DEG = 10.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_parameter_sheet(path: Path) -> dict[str, tuple[float, str]]:
    document = App.openDocument(str(path))
    try:
        sheet = document.getObject("Parameters")
        if sheet is None:
            raise RuntimeError("Bottle FCStd has no Parameters spreadsheet")
        values: dict[str, tuple[float, str]] = {}
        for row in range(2, 64):
            try:
                name = str(sheet.get(f"A{row}") or "").strip()
            except Exception:
                break
            if not name:
                continue
            values[name] = (float(sheet.get(f"B{row}")), str(sheet.get(f"C{row}")))
        return values
    finally:
        App.closeDocument(document.Name)


def build(output_root: Path) -> dict[str, object]:
    if not SOURCE_BOTTLE.is_file():
        raise FileNotFoundError(SOURCE_BOTTLE)
    params = read_parameter_sheet(SOURCE_BOTTLE)
    thread_od_mm = params["thread_OD"][0]
    support_od_mm = params["support_OD"][0]
    inner_radius_mm = thread_od_mm / 2.0 + THREAD_RADIAL_CLEARANCE_MM
    outer_radius_mm = support_od_mm / 2.0
    if inner_radius_mm >= outer_radius_mm:
        raise RuntimeError("Derived cap wall thickness is not positive")

    cad_dir = output_root / "geometry/cad_intermediate"
    visual_dir = output_root / "geometry/visual"
    report_dir = output_root / "reports"
    for directory in (cad_dir, visual_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    document = App.newDocument("BottleCapDiagnosticV1")
    try:
        sheet = document.addObject("Spreadsheet::Sheet", "Parameters")
        entries = [
            ("outer_diameter", outer_radius_mm * 2.0, "mm", "DERIVED_FROM_BOTTLE_SUPPORT_OD"),
            ("inner_diameter", inner_radius_mm * 2.0, "mm", "DERIVED_FROM_THREAD_OD_PLUS_CLEARANCE"),
            ("height", CAP_HEIGHT_MM, "mm", "TEMPORARY_UNCALIBRATED"),
            ("top_thickness", TOP_THICKNESS_MM, "mm", "TEMPORARY_UNCALIBRATED"),
            ("thread_radial_clearance", THREAD_RADIAL_CLEARANCE_MM, "mm", "TEMPORARY_UNCALIBRATED"),
        ]
        for col, title in zip(("A", "B", "C", "D"), ("name", "value", "unit", "evidence")):
            sheet.set(f"{col}1", title)
        for row, (name, value, unit, evidence) in enumerate(entries, start=2):
            sheet.set(f"A{row}", name)
            sheet.set(f"B{row}", str(value))
            sheet.set(f"C{row}", unit)
            sheet.set(f"D{row}", evidence)

        outer = Part.makeCylinder(outer_radius_mm, CAP_HEIGHT_MM)
        cavity = Part.makeCylinder(inner_radius_mm, CAP_HEIGHT_MM - TOP_THICKNESS_MM)
        cap = outer.cut(cavity).removeSplitter()
        master = document.addObject("Part::Feature", "BottleCapMaster")
        master.Shape = cap
        if getattr(master, "ViewObject", None) is not None:
            master.ViewObject.ShapeColor = (0.10, 0.28, 0.80)
        document.recompute()

        fcstd_path = cad_dir / "bottle_cap_diagnostic_v1.FCStd"
        step_path = cad_dir / "bottle_cap_diagnostic_v1.step"
        obj_path = visual_dir / "bottle_cap_visual.obj"
        document.saveAs(str(fcstd_path))
        Part.export([master], str(step_path))
        mesh = MeshPart.meshFromShape(
            Shape=cap,
            LinearDeflection=LINEAR_DEFLECTION_MM,
            AngularDeflection=math.radians(ANGULAR_DEFLECTION_DEG),
            Relative=False,
        )
        mesh.write(str(obj_path))

        bounds = cap.BoundBox
        solids = list(cap.Solids)
        if len(solids) != 1:
            raise RuntimeError(f"Expected exactly one cap solid, got {len(solids)}")
        center_of_mass = solids[0].CenterOfMass
        report: dict[str, object] = {
            "status": "PASS" if cap.isValid() and len(solids) == 1 else "FAIL",
            "classification": "TEMPORARY_UNCALIBRATED_DIAGNOSTIC_CAP",
            "source_bottle": {
                "absolute_path": str(SOURCE_BOTTLE),
                "sha256": sha256(SOURCE_BOTTLE),
                "read_only": True,
                "thread_od_mm": thread_od_mm,
                "support_od_mm": support_od_mm,
            },
            "geometry": {
                "valid": bool(cap.isValid()),
                "solid_count": len(solids),
                "outer_diameter_mm": outer_radius_mm * 2.0,
                "inner_diameter_mm": inner_radius_mm * 2.0,
                "height_mm": CAP_HEIGHT_MM,
                "top_thickness_mm": TOP_THICKNESS_MM,
                "wall_thickness_mm": outer_radius_mm - inner_radius_mm,
                "thread_radial_clearance_mm": THREAD_RADIAL_CLEARANCE_MM,
                "volume_mm3": float(cap.Volume),
                "surface_area_mm2": float(cap.Area),
                "center_of_mass_mm": [float(v) for v in center_of_mass],
                "bounds_mm": {
                    "min": [bounds.XMin, bounds.YMin, bounds.ZMin],
                    "max": [bounds.XMax, bounds.YMax, bounds.ZMax],
                },
            },
            "tessellation": {
                "linear_deflection_mm": LINEAR_DEFLECTION_MM,
                "angular_deflection_deg": ANGULAR_DEFLECTION_DEG,
                "relative": False,
                "facet_count": int(mesh.CountFacets),
            },
            "toolchain": {
                "freecad_version": App.Version(),
                "opencascade_version": getattr(Part, "OCC_VERSION", "7.8.1"),
            },
            "outputs": {
                "fcstd": str(fcstd_path),
                "step": str(step_path),
                "obj": str(obj_path),
            },
        }
        for key, path in (("fcstd", fcstd_path), ("step", step_path), ("obj", obj_path)):
            report["outputs"][f"{key}_sha256"] = sha256(path)  # type: ignore[index]
        report_path = report_dir / "freecad_cap_audit.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        if report["status"] != "PASS":
            raise RuntimeError("BottleCap CAD audit failed")
        return report
    finally:
        App.closeDocument(document.Name)


def main() -> None:
    output_root = Path(os.environ.get("BOTTLE_CAP_OUTPUT_ROOT", str(DEFAULT_ROOT)))
    build(output_root.resolve())


# FreeCAD executes a positional Python macro with a non-standard module name.
# Deliberately invoke the entry point unconditionally, matching the repository's
# other pinned-FreeCAD builders.
main()
