# Isaac Visual Tutor Extension

Extension name:

```text
my.isaac.visual_tutor
```

Path:

```text
visual_tutor/isaac_extensions/my.isaac.visual_tutor
```

Purpose:

- provide a simulation-only Isaac panel;
- keep timeline paused unless the user explicitly changes it;
- capture stage and selection snapshots;
- avoid ROS, real robot control, or original asset overwrite.

To enable in Isaac Sim, add `visual_tutor/isaac_extensions` to the Extension Manager search path, then enable `my.isaac.visual_tutor`.
