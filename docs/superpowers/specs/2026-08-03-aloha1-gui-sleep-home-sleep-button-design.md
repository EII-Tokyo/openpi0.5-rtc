# ALOHA1 GUI Sleep–Home–Sleep button design

## Scope

Add a small Isaac Sim 5.1 GUI control window to the isolated runtime-Sleep
review launcher. The first button runs the frozen
`runtime_measured_sleep` manifest inside the visible GUI process only. A
separate real-robot arm switch is present but starts disarmed and cannot create
a ROS publisher while disarmed.

## Data flow

The controller loads and hash-checks the existing Stage, finger-limit layer,
and runtime-measured manifest. On click it requires the digital articulation to
be at Sleep, pauses any prior run, and applies one manifest sample per physics
step. It records target/readback, segment, frame, and wall-clock timestamps to a
new GUI-run report. The visible GUI process is therefore the process that moves
and is recorded; a separate headless validator is not used as visual evidence.

## Real-robot safety boundary

The real path is disabled by default. An `ARM REAL ROBOT` checkbox must be
explicitly enabled, the operator must confirm a modal warning, and the
coordinator must re-check the real Sleep readback, command-topic publisher set,
manifest hash, and stop/hold readiness. Without all gates, the button performs
digital-only motion and reports `REAL_NOT_ARMED`. This change does not connect
to 103 or publish commands during GUI startup.

## Failure handling

Any invalid state, stale hash, missed sample deadline, or transport failure
aborts the digital run and writes a machine-readable failure report. A real
transport failure uses latest joint readback hold; it never sends an all-zero
command. The existing user GUI and Stage are not modified.

## Acceptance

- GUI button is visible in Isaac Sim 5.1 Full GUI on workspace 2.
- Digital-only click produces one complete Sleep–Home–Sleep run in that same
  visible process and a report/video path.
- Disarmed mode creates zero ROS publishers and zero real commands.
- Existing focused tests pass and frozen Stage hash remains unchanged.
