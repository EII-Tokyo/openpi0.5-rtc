# ALOHA1 Grasp Frame Run13 Rejection

Status: **FAIL**

Classification:
`REJECTED_WHOLE_PAD_FACE_CENTROID_NOT_EFFECTIVE_GRASP_CENTER`.

Run13 did not directly reuse the official EE helper as the grasp origin, but
its replacement was still invalid: it used the midpoint of the complete
left/right inward pad-face centroids. The supplier STEP supplied the correct
handed finger B-Reps and faces 117/128, but the complete gripper shell,
sliding carriage, runtime bar envelope, and project Bottle500 B-Rep were not
included in the clearance calculation.

The native Grasp Editor returned success and confidence 1.0. That result is
rejected. There were no nonpositive-separation left or right finger contacts,
no bilateral finger contact, and the final mimic residual was
0.0096722543 m. The raw report contains 14,774 Bottle500/gripper-bar pair
events; their minimum separation is 0.00421 mm and their maximum impulse is
0.000305883 N·s. This is contact-envelope evidence, not a stable grasp.

Both run13 screenshots were individually reviewed and failed the visual gate.
The full-arm camera is too distant to prove distal-pad capture, and the
simulated image does not show the bottle held between both effective pad
regions. Machine contact evidence independently confirms the failure.

No new Grasp Editor or IK run is authorized until the complete supplier
gripper and project Bottle500 B-Rep clearance gate passes.
