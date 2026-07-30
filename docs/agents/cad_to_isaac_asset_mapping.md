# CAD To Isaac Asset Mapping

This is the canonical project rule for supplier or measured CAD (`STEP`,
`STP`, `IGES`, `FCStd`, Parasolid, JT, or equivalent) used to build an Isaac
Sim asset. It supersedes `scene_reconstruction.md` for CAD ingestion,
assembly interpretation, tessellation, robot-link geometry mapping, and
CAD-derived USD. The photo reconstruction rules remain applicable only to
photo evidence and photo-derived workcell proxies.

## Scope And Source Precedence

- Treat CAD as geometric evidence, not as a robot description. CAD assembly
  placements do not by themselves define joint type, axis, limit, dynamics,
  mimic behavior, or controller order.
- Use the version-pinned URDF/Xacro and control source for kinematics and
  naming. Use supplier CAD, project evidence, and physical measurement to
  supply geometry and verified mounting transforms.
- Prefer evidence in this order for a claimed value:
  1. exact supplier/manufacturer source with immutable identity;
  2. version-pinned official robot description;
  3. project report with reproducible provenance;
  4. physical measurement or calibration;
  5. derived value with the derivation recorded;
  6. engineering inference;
  7. temporary diagnostic value.
- A photograph may validate appearance or expose an orientation error. It
  must not override an exact CAD surface or supply an unmeasured precision
  transform.
- If two sources conflict, preserve both, report the conflict, and do not
  silently choose the source that makes a downstream test pass.

## Mandatory Gates

Before converting or editing anything:

1. Read the repository `AGENTS.md`, this file, and
   `docs/agents/isaac_mcp_toolchain.md`.
2. Probe the current host. Do not trust a previous statement that FreeCAD,
   Isaac Sim, a CAD converter, OpenUSD, or a Python module is installed.
3. Record exact tool versions and executable/module paths. For Isaac work,
   pin the locally installed Isaac Sim, Kit, PhysX, extension, and schema
   versions. Do not substitute `latest`, Isaac Sim 6.x, or an online example
   for a pinned local 5.1 workflow.
4. Use NVIDIA's official Isaac capability through the reviewed
   `mcpjungle_lab` Gateway before changing Isaac code, USD, physics, GUI, or
   runtime behavior. If a needed CAD/host capability is not exposed by the
   selected Gateway group, report that a reviewed Gateway bridge is required;
   do not add a direct MCP server.
5. Resolve the user-approved active Stage by absolute path, SHA-256, root
   prim, sublayers, and required prims before switching it. Investigation
   stages never replace the review Stage implicitly.
6. Freeze the baseline inputs and outputs. A diagnostic variant must use a
   separate directory or layer and must not overwrite the source CAD,
   imported source USD, accepted configuration layer, or final default asset.

## External Source Manifest

Every external CAD resource must have a machine-readable manifest entry with:

- provider and original URL;
- repository plus branch/tag/commit when applicable;
- provider file ID or immutable revision when no Git commit exists;
- retrieval timestamp in UTC;
- original filename, MIME type, byte size, and SHA-256;
- local absolute path and storage class;
- embedded STEP filename, timestamp, originating system, preprocessor, and
  schema when available;
- license identifier, license evidence path/URL, attribution requirement,
  and redistribution status;
- `read_only: true`;
- relationships to parent folders, assemblies, and downstream outputs.

Public download access is not license evidence. When a license cannot be
confirmed, set `license_status: UNKNOWN_HARD_BLOCKER`, keep the resource out
of commits and redistributed packages, and continue read-only local analysis
only when the user's authorization and applicable policy permit it.

Store unknown-license or unreviewed downloads in a task artifact cache or
other ignored external-source cache. Do not mix them with finalized,
redistributable source assets.

## Directory And Layer Contract

For a versioned asset root such as `assets/<vendor>/<asset>/<version>/`, keep
these roles distinct:

```text
source/                    immutable, redistributable source only
geometry/cad_intermediate/ reproducible derived CAD, never the original
geometry/visual/           versioned tessellated visual meshes
geometry/collision/        separately justified collision geometry
usd/source/                raw imported/source geometry USD
usd/configuration/         naming, hierarchy, and placement overrides
usd/physics/               mass, collider, joint, drive, and material layers
diagnostics/<task>/        isolated hypotheses and comparison variants
```

Use `.codex/artifacts/<task>/` for logs, temporary source caches, screenshots,
and high-volume evidence. Put durable machine-readable reports under the
task's `reports/` tree. Every report must contain absolute input/output paths
and hashes.

## CAD Intake Audit

Audit the source before tessellation:

- format and schema, including STEP AP level;
- declared length, angle, and mass units;
- product/part names, product count, instance count, and assembly depth;
- local placement and world placement for every instance;
- transform determinant, handedness, non-uniform scale, and mirror state;
- color/material metadata and whether the importer preserved it;
- B-rep validity, null shapes, solids, shells, faces, edges, vertices, AABB,
  volume, area, and center of mass when observable;
- duplicate geometry and instance reuse;
- missing or suppressed parts;
- import warnings and parser/cooking logs.

Run `shape.isValid()` or the local equivalent and save detailed check output
when invalid. A successful GUI open or a nonempty viewport is not a geometry
audit.

For STEP assemblies, preserve product hierarchy, instance identity, and
placements in the source audit. Do not flatten before the hierarchy has been
recorded and verified.

## Robot-Link Mapping

Create an explicit mapping table from CAD product instances to robot links.
Each row must include:

- CAD file hash, product name, instance path, and source world transform;
- target robot, URDF link, and intended USD prim path;
- CAD-to-link matrix and translation/quaternion;
- unit conversion, axis conversion, determinant, and mirror flag;
- evidence class, evidence path, confidence, and approval state;
- visual role, collision role, or `REFERENCE_ONLY`;
- unresolved ambiguity or `HARD_BLOCKER`.

Never infer mapping by alphabetical order. Never mirror a chiral part merely
because it is convenient. A standalone finger CAD file establishes shape, not
its installed handedness or attachment transform. The installed transform
must come from an assembly that contains the instance, an official drawing,
or a measured/approved transform.

CAD assembly relations are not articulation joints. Joint frames and axes
must be reconciled against the version-pinned URDF/Xacro and then tested
one joint at a time after import.

## Deterministic Tessellation

Keep the exact B-rep source unchanged. Mesh generation must be scripted and
must record:

- CAD importer and geometry-kernel versions;
- mesher API actually probed on the current host;
- absolute/relative linear deflection;
- angular deflection and its unit;
- unit scale;
- sew/weld tolerance and policy;
- normal, smoothing, UV, color, and material policy;
- triangulation and degenerate-face policy;
- per-part instance/merge policy;
- output format and canonicalization method.

Do not rely on hidden GUI preferences. Probe the local API, save requested
values and runtime readback, and fail if required parameters are unavailable.

For every mesh, report vertex/triangle counts, AABB, surface area and volume
when comparable, manifold/watertight status, degenerate elements, connected
components, and distance/deviation from the source surface when the local
toolchain can compute it.

Run tessellation at least twice in fresh output directories. Compare file
hashes and a canonical geometric signature. If byte-identical output is not
available, document why and enforce explicit topology, bounds, and sampled
surface-deviation tolerances.

### Pinned Project-Local FreeCAD Runtime

Use the project-local, self-contained FreeCAD runtime for scripted CAD
inspection and tessellation:

```text
/home/eii/project/openpi0.5-rtc-reward-learning/local_tools/freecad-tessellation/
```

The required headless entry point is:

```bash
/home/eii/project/openpi0.5-rtc-reward-learning/local_tools/freecad-tessellation/freecadcmd --version
```

Do not use `/snap/bin/freecad` or the Snap `freecad.cmd` entry point for a new
deterministic tessellation run. The Snap path is retained only for explicitly
requested legacy reproduction. The project-local wrapper removes inherited
`LD_LIBRARY_PATH` and `PYTHONPATH` before starting the bundled `freecadcmd`;
this prevents the old Snap-specific libcurl override and Isaac Sim Python
paths from entering the CAD process.

Before use, require all of the following:

- `local_tools/freecad-tessellation/manifest.json` exists and has
  `status: PASS`;
- the wrapper is executable;
- the AppImage SHA-256 is
  `e2006138400b2fa85fa2e160e872d00767eb32964e85075830f7e198a3a876e1`;
- runtime readback is FreeCAD `1.1.1`, commit
  `0108fd4b4850cc46e625b60e53cea7a7bbe69f8d`, Python `3.11.14`, and
  OpenCascade `7.8.1`;
- `MeshPart.meshFromShape` accepts explicit linear and angular deflection
  without an external library override.

The official Linux x86_64 AppImage and its extracted runtime are local-only:

```text
local_tools/freecad-tessellation/FreeCAD_1.1.1-Linux-x86_64-py311.AppImage
local_tools/freecad-tessellation/runtime/
```

The host AppImageLauncher cannot read this official AppImage's zstd-compressed
SquashFS payload. The verified local installation therefore uses
`unsquashfs 4.6.1`, offset `944632`, instead of modifying AppImageLauncher or
system libraries.

Current installation validation is recorded at:

```text
local_tools/freecad-tessellation/manifest.json
local_tools/freecad-tessellation/validation/final_meshpart_probe.json
local_tools/freecad-tessellation/validation/final_fresh_tessellation/manifest.json
```

The final fresh supplier-finger run used `LinearDeflection=0.2 mm`,
`AngularDeflection=20 degrees`, and `Relative=False`. Its outputs match both
independent project-local runs and the earlier Snap diagnostic byte-for-byte:

- left finger OBJ SHA-256:
  `c6710d0fe5b2030a32722d9df5c0b553c771c9d61d92b8ddaec36c94c5963488`;
- right finger OBJ SHA-256:
  `b0979c5d55fee448dab512dc75b1251bab17d94892decd01de9a6e76c01482d1`;
- each finger has 831 vertices, 1662 triangles, one connected component, and
  zero degenerate triangles.

The earlier Snap plus local-libcurl override did produce a valid
angular-controlled `PASS`, but its override dependency was not completely
manifested. Treat old `PARTIAL` or `HARD_BLOCKER` text about unavailable
angular deflection as stale. The self-contained project-local runtime is the
default going forward.

The bundled AppImage, extracted runtime, and validation outputs are excluded
locally through `.git/info/exclude`; do not commit these large binaries.
Installing this toolchain did not modify `.venv_issac`, Snap FreeCAD, system
libraries, source CAD, source USD, or final/default colliders.

## USD And Isaac Composition

- Use meters, kilograms, seconds, right-handed coordinates, `+Z` up, and
  `+X` forward. Save transform matrices and translation/quaternion with the
  quaternion ordering named explicitly.
- Keep raw source geometry, configuration, physics, sensors, and controls in
  separate layers. Source geometry remains re-importable.
- Preserve one articulation per robot. Do not merge links across active
  joints. Do not turn a CAD assembly into one rigid mesh before link mapping.
- Visual meshes and collision meshes are separate products. A visual mesh is
  not automatically an accepted collider.
- Do not derive calibrated mass or inertia from default density. CAD volume
  may support a calculation only when material/density evidence is explicit.
- Do not optimize, merge, decimate, instance, flatten, or create payload
  variants until the unoptimized asset passes the same regression gates.
- Verify the composed Stage, not just input parameters: default prim,
  references, sublayers, resolved external assets, prim paths, stage units,
  axis, world transforms, articulation roots, DOF names/order, colliders, and
  applied schemas.

## Visual Evidence And Screenshot Contract

Screenshots are auxiliary evidence and must be paired with machine-readable
geometry or runtime data.

Before each capture, record the object, part, physical/kinematic phase, view,
and acceptance criterion. For an orientation or gripper test:

- capture open and closed states from four unambiguous directions that expose
  both fingers, their inner surfaces, the gripper bar, and any test object;
- do not label a view `top` unless its camera direction is proven against the
  asset/world axes;
- keep the same camera pose for the open/closed comparison;
- save an original and annotated image for every accepted capture;
- annotate left/right parts, inner contact surfaces, opening or displacement,
  test object, contact/normal when applicable, stage/frame/time, camera pose,
  asset hash, and PASS/FAIL/PARTIAL;
- visually inspect every image with the vision model; retake images with
  occlusion, excessive distance, wrong view, or indistinguishable states;
- save a screenshot-review JSON containing absolute paths, SHA-256, camera
  matrix, target, self-review result, and retake history.

Do not accept geometry, contact, or hold based on viewport appearance alone.

## Machine Validation

At minimum, automated validation must check:

- source and derived hashes plus manifest completeness;
- B-rep audit completion and unit resolution;
- assembly instance count and transform preservation;
- explicit CAD-to-link coverage with no ambiguous active link;
- repeatable tessellation signatures;
- all USD/mesh/texture references resolve;
- expected prim, link, articulation, and DOF structure;
- finite transforms, bounds, mass, inertia, limits, forces, and velocities
  where applicable;
- collision and initial-overlap reports;
- official Isaac asset/physics/robot rules available in the pinned local
  version;
- repeat-run determinism;
- screenshot pairs and their visual-review records.

Reports use only `PASS`, `FAIL`, `PARTIAL`, or `NOT_RUN`. A `PASS` must link
to the evidence that satisfies the gate. A persistent contact report is not
equivalent to a stable grasp, and a visual match is not equivalent to a
physical validation.

## Diagnostic Discipline

- Change one causal variable at a time. Do not simultaneously alter geometry,
  friction, drive, mimic, mass, timestep, or solver settings to make a result
  pass.
- Keep baseline and diagnostic variants immutable and hash-addressed.
- Promote a diagnostic result only after the baseline and candidate run the
  same regression suite.
- Record temporary values as `TEMPORARY_UNCALIBRATED` or
  `DIAGNOSTIC_ONLY_NOT_FINAL`.
- Missing proprietary geometry, license evidence, assembly transforms,
  material properties, or physical measurements become explicit
  `HARD_BLOCKER` entries. Continue all work that does not depend on them.

## Forbidden Shortcuts

- No source overwrite, silent healing, or unrecorded CAD repair.
- No filename-based part identity or orientation guess.
- No photo-derived precision dimension when exact CAD or measurement is
  required.
- No unrecorded scale, axis conversion, mirror, or transform bake.
- No flattened robot mesh as a substitute for per-link mapping.
- No collider copied from a visual mesh without an explicit collision audit.
- No default density, extreme friction, fixed constraint, or SurfaceGripper
  used to conceal a mapping or grasp failure.
- No claim of completion from file existence, viewport appearance, or a
  single successful run.
