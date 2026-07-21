# Phase133: ALOHA RL Readiness Four-Level Gate

## Conclusion

Fixed-initial-state replay is a calibration gate, not a proof that ALOHA can solve a grasping task from an unknown bottle pose.

The current replay work can prove that the Isaac control chain is internally consistent under one assumed initial state. It cannot prove that a policy can handle different bottle positions. For RL training, the environment must define reset, observation, reward, termination, and evaluation over a distribution of initial states.

## Why Replay Is Not Enough

An HDF5 replay fixes both the robot trajectory and the assumed object pose. If the same action sequence succeeds, the evidence is:

- DOF names, signs, limits, and target conversion are plausible.
- The drive target controller can track the replayed qpos under this initial state.
- The collision setup may be plausible for this exact object placement.

It does not prove:

- the bottle can start at another pose;
- the observation contains enough information to locate the bottle;
- a policy can choose different actions for different bottle poses;
- the reward distinguishes good and bad grasp attempts;
- camera perception is usable.

The official Isaac physics model advances a discrete simulation state from current state plus inputs. Therefore a training task must test state-dependent action selection, not only replay an already chosen action sequence.

## Four Gates

### Gate 1: Control Interface

Goal: prove robot control is correctly connected.

Acceptance:

- Joint target and readback correspond.
- DOF order, signs, and limits are correct.
- Gripper opens and closes stably.
- Reset does not cause jumps, explosions, or abnormal velocities.
- Same action input from the same initial state gives repeatable motion.

Current status: mostly passed for arm-only qpos replay and drive target readback. This does not imply grasping task readiness.

### Gate 2: Fixed-Pose Minimal Grasp

Goal: prove deterministic grasp physics at one known bottle pose.

Acceptance:

- Bottle starts from one explicitly authored fixed pose.
- A scripted controller, IK controller, or state machine approaches the bottle.
- Left gripper closes on the bottle body, not on the gripper base or a wrong mesh.
- Bottle is lifted above the table by a measurable threshold.
- Target contact is formed by semantically correct finger pad colliders.
- Non-target contacts are not allowed as a shortcut.

This gate should be passed before using RL. If a deterministic controller cannot solve one fixed-pose grasp, RL will hide simulator errors inside a reward curve.

### Gate 3: Randomized Bottle Pose With Simulator Truth

Goal: prove the task is learnable when bottle pose changes.

First use a small reset range, for example:

$$
\Delta x,\Delta y \in [-0.02,0.02]\ \mathrm{m}
$$

The policy observation should initially use simulator truth:

$$
o_t = [q_t,\dot q_t,p_{O,t}^{B},p_P^{B},g_t]
$$

Where:

- $q_t$ is robot joint position.
- $\dot q_t$ is robot joint velocity.
- $p_{O,t}^{B}$ is bottle/object position in the robot base frame.
- $p_P^{B}$ is the pipe or target position in the robot base frame.
- $g_t$ is gripper state.

Acceptance:

- Reset samples bottle pose from a known distribution.
- Training and evaluation use separate random seeds.
- The policy succeeds across held-out random poses.
- Reward and termination are stable and do not depend on hidden future labels.
- Different bottle poses lead to meaningfully different actions.

This is the main sign that ALOHA can begin RL in simulation.

### Gate 4: Camera-Based Perception

Goal: replace simulator-truth object pose with perception.

Possible inputs:

- RGB or RGB-D images.
- Keypoints.
- estimated bottle pose.
- asymmetric training where the actor sees images and the critic may see simulator truth.

Acceptance:

- Gate 3 remains stable as a non-vision control baseline.
- The camera pipeline estimates the bottle/pipe relation with bounded error.
- Failure can be attributed separately to perception, control, contact physics, or reward.

Do not start here. If visual training fails before Gate 3 is stable, the failure source is ambiguous.

## Code Representation

The conservative readiness report is implemented in:

- `aloha_isaac_replay/rl/readiness.py`

Regression tests:

- `aloha_isaac_replay/tests/test_rl_readiness.py`

Important status semantics:

- `NOT_READY_FOR_RL_TRAINING`: Gate 1 or Gate 2 or Gate 3 has not passed.
- `READY_FOR_PRIVILEGED_STATE_RL_TRAINING_NOT_CAMERA`: Gate 1-3 passed; simulator-truth RL can start, but camera policy is not ready.
- `READY_FOR_CAMERA_BASED_RL_TRAINING`: Gate 1-4 passed.

## Next Engineering Step

Continue the current replay work only as Gate 2 calibration:

1. Fix the bottle initial pose hypothesis for the manual grasp window.
2. Replace oversized or semantically wrong gripper colliders with true inner finger pad contact proxies.
3. Confirm that the first target contact is finger-pad-to-bottle, not gripper-bar-to-bottle.
4. Add a deterministic fixed-pose approach-close-lift script.

After Gate 2 passes, build a minimal Gate 3 task with small random bottle pose reset and privileged truth observation.

## Sources

- NVIDIA Isaac Sim MCP instruction set: `physics`, `robot_simulation`, `isaac_lab`, `omniverse_and_usd`.
- Local implementation: `aloha_isaac_replay/rl/readiness.py`.
- Local tests: `aloha_isaac_replay/tests/test_rl_readiness.py`.
