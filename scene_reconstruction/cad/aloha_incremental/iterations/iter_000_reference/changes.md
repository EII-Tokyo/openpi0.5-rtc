# Iteration 000 Reference

## Scope

- Imported the current Isaac ALOHA source scene as read-only reference mesh geometry: dual arms, existing table, existing frame, and existing scene cameras.
- Created a transparent desktop reference plane as an orientation aid from the existing scene dimensions.
- Added world axes for orientation.

## Deliberately Not Done

- No water pipe.
- No new rack.
- No new external camera model.
- No photo-derived millimeter reconstruction.
- No reverse-engineered parametric CAD from Isaac USD meshes.
- No separate ALOHA2 `workcell_v2` STL is used as the primary reference.

## Reference Lock

All `REF_ALOHA_*` and `REF_SCENE_*` objects are read-only reference geometry by workflow rule. Scripts may regenerate them from source assets, but must not edit source files under `external/` or `local_eval_assets/`.
