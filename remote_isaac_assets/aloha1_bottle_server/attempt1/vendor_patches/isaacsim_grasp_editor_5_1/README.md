# Isaac Sim 5.1 Grasp/Gripper persistence

This directory records two independent fixes accepted on 2026-08-17.

1. `aloha1_human_accepted_gripper_control.usda` is a reusable USD override
   layer for both ALOHA1 follower grippers. It preserves one active
   `left_finger` drive and the passive `right_finger` PhysX mimic relationship.
2. `grasp_editor_selected_root_frame.patch` fixes the Isaac Sim 5.1 Grasp
   Editor frame dropdown when the articulation root is a sibling of the end
   effector link hierarchy. It changes only the extension's enumeration root;
   it does not change the robot USD hierarchy.

The Grasp Editor patch was made against:

- extension: `isaacsim.robot_setup.grasp_editor`
- installed file:
  `/home/eii/Applications/isaacsim-5.1.0/exts/isaacsim.robot_setup.grasp_editor/isaacsim/robot_setup/grasp_editor/ui_builder.py`
- original SHA-256:
  `bd48e3c3c0e5587d9f790e86a5b1b843aa4b64cde26420af09f5f1f4dbf29823`
- patched SHA-256:
  `1258241849257f2da3e82a20741785fc8eb9e392e9bdb134d991317e533901dc`

The installed extension is intentionally not restored while the current
Grasp Editor session is open. A normal Isaac restart keeps the patch because
the installed file is already patched. An Isaac upgrade/reinstall can replace
it; re-check the original hash before applying the patch to a new version.

The USD values are simulation parameters validated by human free-space
full-stroke tests, not direct DYNAMIXEL register values and not a calibrated
sim-to-real identification result.
