# ALOHA 20 cm five-pose downward-gripper acceptance v6

Status: **PASS**. All five frozen samples are machine `PASS` and evidence
`PASS`. The new sample 5 video passed full-frame visual-model review, and the
user confirmed on 2026-07-31 that the grasp is correct.

Samples 1 and 2 preserve the already accepted legacy initial-orientation
exceptions and were not rerecorded. Samples 3 and 4 preserve their already
successful downward-gripper runs. Only the previously failed sample 5 was
replanned and rerun, using candidate 119. Its initial gripper approach axis is
`7.189721450960664°` from world `-Z`, inside the frozen
`23.241131059202324°` gate.

Sample 5 reaches `0.20077485934609024 m` bottle clearance, holds for `2.0 s`,
and drops `0.0007390718475712432 m`, below the unchanged `0.010 m` gate. Its
fresh primary and repeat deterministic signature is
`879b4b88e25b1c54bf38b2713e59ad77b330d3805f1ac8d43561cf271512cff7`.

The previous timeout was a contact-gate interpretation error. PhysX reported
bilateral force-carrying contacts with finite positive impulse while geometric
separation remained slightly positive inside the contact envelope. The
controller now uses bilateral reported pairs with finite positive solver
impulse for the physical contact gate and retains `separation <= 0` as a
separate diagnostic. Collider, friction, drive, mimic, bottle parameters,
timestep, solver iterations and hold thresholds were not changed.

The sample 5 primary action video contains 912 frames at 60 fps. All frames
were covered exactly once by 46 contact sheets and reviewed by the visual
model. The whole arm is visible, the initial gripper points downward, the
horizontal bottle and both fingers remain visible, and open, contact, lift and
hold are distinct. A first collision-evidence batch captured `OPEN_PREGRASP`
before the open target was reached and was rejected; the fresh retake contains
24 reviewed records and passes.

The source Stage remains unchanged at SHA-256
`2b3f76365ed67532f478d995ae859a88b5639975ac07cb7ac8a53ac679e8205c`.
No real robot or `192.168.1.103` access occurred. Task 8 remains `NOT_RUN`.

Authoritative machine-readable report:

`/home/eii/project/openpi0.5-rtc-reward-learning/reports/aloha1_mapping/aloha1_grasp_20cm_five_pose_downward_acceptance_v6.json`

User-confirmed sample 5 annotated video:

`/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260731-aloha1-grasp-downward-contact-gate/formal_candidate119_v5/sample_05/primary/video_attempt_001/video/aloha1_grasp_20cm_annotated_candidate.mp4`
