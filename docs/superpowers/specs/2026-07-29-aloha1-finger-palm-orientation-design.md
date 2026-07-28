# ALOHA 1 Finger Palm Orientation Diagnostic Design

## Scope

Correct only the installation-orientation diagnosis for the user-confirmed
Stationary ALOHA 1 follower custom fingers. Do not modify the original URDF,
imported source USD, final/default collider, physics parameters, Task 8, or
real hardware.

The current blue `left_finger` installation is rejected because its palm
faces away from the gripper center. Any earlier orientation or hold result
that used that installation is not acceptance evidence for the corrected
installation.

## Evidence boundary

Use these fixed inputs:

- left STL SHA-256
  `df73ae5b9058e5d50a6409ac2ab687dade75053a86591bb5e23ab051dbf2d659`;
- right STL SHA-256
  `56fb3cc1236d4193106038adf8e457c7252ae9e86c7cee6dabf0578c53666358`;
- the unchanged follower diagnostic articulation and legal finger q limits;
- Isaac Sim `5.1.0.0`, Kit `107.3.3`, PhysX `107.3.26`.

The two STL meshes are a measured geometric mirror pair about source Y. A
candidate that changes only the blue left finger is diagnostic-only because
it breaks bilateral mirror symmetry.

## Candidates

Create candidates only under a new palm-orientation diagnostic directory:

- `A_CURRENT_REJECTED`: unchanged current installation;
- `B_LEFT_ONLY_DIAGNOSTIC`: rotate only the blue left finger 180 degrees
  around the source-derived finger longitudinal axis;
- `C_BILATERAL_SYMMETRIC`: apply the corresponding 180-degree correction to
  both mirrored fingers while preserving bilateral symmetry.

Do not promote any candidate to the default asset before user visual review.

## Geometry gates

For each candidate, record:

- exact visual and collision prim paths;
- input STL hashes;
- full link-relative transforms;
- transform determinants and rigid-transform checks;
- left/right center-plane symmetry residual;
- palm-facing direction relative to the gripper centerline;
- open and closed legal finger q values;
- source and generated USD SHA-256.

The palm-facing gate must be geometry-derived. Image color, viewport
appearance, file existence, or a manually written label cannot make it pass.

## Screenshot contract

Each candidate must include at least:

- open gripper top view;
- closed gripper top view;
- open center-facing oblique view;
- closed center-facing oblique view.

Within a candidate, open and closed captures use the same fixed camera pose.
The top view must expose both fingers, both palm/contact surfaces, the
gripper centerline, and the aperture. Images that hide a palm, crop a finger,
or make open and closed states visually indistinguishable are rejected and
recaptured.

Save an original PNG and an annotated PNG for every accepted capture. The
annotation identifies the blue left finger, orange right finger, each palm,
the gripper centerline, the aperture, candidate, q state, camera view, key
numeric orientation evidence, and `PASS`/`FAIL`.

The machine-readable review report records absolute original/annotated paths,
camera pose, candidate transform, visual self-review conclusion, and any
retake reason.

## Acceptance

The assistant visually reviews every accepted image before delivery. The user
then selects or rejects the candidate from the screenshots. Until that
selection:

- corrected installation status is `AWAITING_USER_VISUAL_CONFIRMATION`;
- Task 5 corrected-install rerun is `NOT_RUN`;
- Task 7 rerun is `NOT_RUN`;
- Task 8 remains `NOT_RUN`;
- final/default collider remains unchanged.
