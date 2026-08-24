# ALOHA Codex Isaac Sim monitor

This monitor runs on `aloha` as a user-level systemd timer. It polls Isaac Sim
without inotify and records a JSON health snapshot every two minutes. A
read-only, ephemeral `codex exec` diagnosis is generated only on the first run,
on a health-state transition, or when new matching error lines appear. A
15-minute cooldown suppresses repeated non-urgent model calls.

The monitor never restarts or signals a process, controls a robot, edits a
file, or saves the USD Stage. Reports are stored under:

`remote_isaac_assets/aloha1_bottle_server/attempt1/reports/codex_monitor/`

Useful commands on `aloha`:

```bash
systemctl --user status aloha-codex-isaac-monitor.timer
systemctl --user start aloha-codex-isaac-monitor.service
journalctl --user -u aloha-codex-isaac-monitor.service -n 100 --no-pager
```
