# Isaac Sim 5.1 Private frp Streaming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Install Isaac Sim 5.1.0 and a boot-enabled private frpc provider on GPU workstation 103, then let a Mac reach SSH and Isaac WebRTC through the existing, unchanged AWS frps service.

**Architecture:** Workstation 103 publishes SSH and Isaac signaling as STCP and media as SUDP to the existing AWS frps listener. The Mac runs matching localhost-only visitors. AWS configuration is treated as immutable and checked by hashes before and after endpoint deployment.

**Tech Stack:** frp 0.66.0, systemd, NVIDIA Isaac Sim 5.1.0 Full Streaming App, Bash, macOS launch-on-demand client.

---

### Task 1: Freeze AWS and endpoint baselines

**Files:**
- Read only: AWS `/etc/frp/frps.toml`
- Read only: AWS `/etc/systemd/system/frps.service`
- Read only: 103 `/home/eii/Applications/`

- [ ] **Step 1: Record AWS hashes, metadata, service status, and bounded listeners**

Run from the control computer:

```bash
ssh ec2 'set -eu; sudo sha256sum /etc/frp/frps.toml /etc/systemd/system/frps.service; sudo stat -c "%n %U:%G %a %s" /etc/frp/frps.toml /etc/systemd/system/frps.service; systemctl is-active frps.service; sudo ss -lntup | sed -n "1,80p"'
```

Expected: `frps.service` is active, TCP 7000 is present, and no AWS mutation occurs.

- [ ] **Step 2: Assert clean 103 installation targets**

```bash
ssh aloha 'set -eu; test ! -e /home/eii/Applications/isaacsim-5.1.0; test ! -e /home/eii/Applications/isaacsim-5.1.0.staging; ! command -v frpc; ! systemctl is-active --quiet frpc.service'
```

Expected: exit status 0. Any existing target is a hard stop.

### Task 2: Deploy frpc 0.66.0 provider on 103

**Files:**
- Create: 103 `/usr/local/bin/frpc`
- Create: 103 `/etc/frp/frpc.toml`
- Create: 103 `/etc/frp/frps-token`
- Create: 103 `/etc/systemd/system/frpc.service`
- Create temporarily: 103 `/home/eii/openpi0.5-rtc-reward-learning/.codex/artifacts/isaac-frp-deploy/`

- [ ] **Step 1: Download the exact Linux archive and verify its embedded version**

```bash
curl -fL --retry 3 -o frp_0.66.0_linux_amd64.tar.gz \
  https://github.com/fatedier/frp/releases/download/v0.66.0/frp_0.66.0_linux_amd64.tar.gz
tar -xzf frp_0.66.0_linux_amd64.tar.gz
./frp_0.66.0_linux_amd64/frpc --version
```

Expected: `0.66.0`.

- [ ] **Step 2: Generate three independent endpoint keys and transfer the existing AWS token without terminal output**

Use `openssl rand -hex 32` once per endpoint. Stream the existing token from the root-readable AWS configuration into a mode-0600 staging file; never include its value in stdout, Git, or chat.

- [ ] **Step 3: Render provider TOML and systemd unit exactly as approved in the design**

Use the provider identities `eii-103-isaac` and `eii-mac`, localhost targets TCP 22, TCP 49100, and UDP 47998. Use token-file authentication and TLS. Run the service as the non-login `frp` system user.

- [ ] **Step 4: Validate before system mutation**

```bash
./frpc verify -c ./frpc.toml
```

Expected: configuration validation succeeds.

- [ ] **Step 5: Run the prepared installer once with interactive sudo**

```bash
cd /home/eii/openpi0.5-rtc-reward-learning
sudo bash .codex/artifacts/isaac-frp-deploy/install-103.sh
```

Expected: the script installs only the four declared system files, creates the `frp` user/group, re-validates the installed configuration, then enables and starts `frpc.service`.

- [ ] **Step 6: Verify service and boot registration**

```bash
systemctl is-enabled frpc.service
systemctl is-active frpc.service
sudo journalctl -u frpc.service -n 60 --no-pager
```

Expected: `enabled`, `active`, successful login and all three proxy registrations. Do not reboot 103.

### Task 3: Configure the Mac visitors and pass network gates

**Files:**
- Create: Mac `~/.local/bin/frpc`
- Create: Mac `~/.config/frp/frpc.toml`
- Create: Mac `~/.config/frp/frps-token`

- [ ] **Step 1: Produce a protected Mac setup bundle**

The setup script detects `arm64` versus `x86_64`, downloads the corresponding official frp 0.66.0 archive, installs `frpc`, renders the approved visitor configuration using the already-generated endpoint keys, and applies mode 0600 to both configuration and token.

- [ ] **Step 2: Verify and start Mac frpc in a visible terminal**

```bash
~/.local/bin/frpc --version
~/.local/bin/frpc verify -c ~/.config/frp/frpc.toml
~/.local/bin/frpc -c ~/.config/frp/frpc.toml
```

Expected: version `0.66.0`, validation succeeds, and all three visitors register.

- [ ] **Step 3: Verify private SSH**

```bash
ssh -p 22022 eii@127.0.0.1
```

Expected: the host key identifies workstation 103 and no AWS public SSH proxy port is involved.

- [ ] **Step 4: Run a bounded bidirectional SUDP probe**

Expected: packets pass both ways through the localhost UDP visitor without sustained loss. Stop before Isaac installation if this gate fails.

- [ ] **Step 5: Recheck AWS immutability**

Re-run Task 1 Step 1. Expected: both SHA-256 values are byte-for-byte identical, `frps.service` remains active, and no public TCP 49100 or UDP 47998 listener exists.

### Task 4: Install and qualify Isaac Sim 5.1.0 on 103

**Files:**
- Create: 103 `/home/eii/Applications/isaacsim-downloads/isaac-sim-standalone-5.1.0-linux-x86_64.zip`
- Create then rename: 103 `/home/eii/Applications/isaacsim-5.1.0.staging/`
- Create: 103 `/home/eii/Applications/isaacsim-5.1.0/`

- [ ] **Step 1: Download resumably from NVIDIA**

```bash
mkdir -p /home/eii/Applications/isaacsim-downloads
curl -fL -C - --retry 5 -o /home/eii/Applications/isaacsim-downloads/isaac-sim-standalone-5.1.0-linux-x86_64.zip \
  https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip
```

Expected: exactly 8,768,419,777 bytes.

- [ ] **Step 2: Record digest and test every ZIP member**

```bash
sha256sum /home/eii/Applications/isaacsim-downloads/isaac-sim-standalone-5.1.0-linux-x86_64.zip
unzip -t /home/eii/Applications/isaacsim-downloads/isaac-sim-standalone-5.1.0-linux-x86_64.zip
```

Expected: the digest is recorded and `unzip -t` reports no errors.

- [ ] **Step 3: Extract atomically and run post-install**

Extract only to the staging directory, assert `isaac-sim.streaming.sh`, `isaac-sim.compatibility_check.sh`, and `post_install.sh`, then rename staging to the final path and execute `post_install.sh` there.

- [ ] **Step 4: Run the compatibility gate**

```bash
cd /home/eii/Applications/isaacsim-5.1.0
./isaac-sim.compatibility_check.sh --/app/quitAfter=10 --no-window
```

Expected: GPU, driver, RTX/Kit, storage, and OS checks pass; no compatibility process remains.

### Task 5: Prove Full Streaming and add the disabled on-demand service

**Files:**
- Create after foreground acceptance: 103 `/home/eii/.config/systemd/user/isaac-sim-streaming.service`

- [ ] **Step 1: Start the full streaming app in the foreground**

```bash
cd /home/eii/Applications/isaacsim-5.1.0
./isaac-sim.streaming.sh \
  --/app/livestream/publicEndpointAddress=127.0.0.1 \
  --/app/livestream/port=49100
```

Expected: `Isaac Sim Full Streaming App is loaded.`, TCP 49100 and UDP 47998 listen locally, no project Stage is selected, and the timeline remains paused.

- [ ] **Step 2: Run Mac client acceptance**

Connect NVIDIA Isaac Sim WebRTC Streaming Client to `127.0.0.1`. Verify picture, keyboard, mouse, reconnect, latency, and clean exit.

- [ ] **Step 3: Install but do not enable the user unit**

Create the exact unit from the approved design, run `systemctl --user daemon-reload`, and assert:

```bash
systemctl --user is-enabled isaac-sim-streaming.service
```

Expected: `disabled`.

- [ ] **Step 4: Verify on-demand start and stop**

```bash
systemctl --user start isaac-sim-streaming.service
systemctl --user is-active isaac-sim-streaming.service
systemctl --user stop isaac-sim-streaming.service
pgrep -af 'isaac-sim|kit' || true
```

Expected: service becomes active, Mac reconnects, stop completes, and no orphan Kit process remains.

### Task 6: Commit operational documentation in isolated batches

**Files:**
- Create: `docs/superpowers/plans/2026-08-02-isaac-sim-frp-remote-streaming.md`
- Modify only if durable facts differ: `docs/superpowers/specs/2026-08-02-isaac-sim-frp-remote-streaming-design.md`

- [ ] **Step 1: Review only task-owned diffs**

```bash
git diff -- docs/superpowers/plans/2026-08-02-isaac-sim-frp-remote-streaming.md docs/superpowers/specs/2026-08-02-isaac-sim-frp-remote-streaming-design.md
```

- [ ] **Step 2: Commit the plan separately**

```bash
git add docs/superpowers/plans/2026-08-02-isaac-sim-frp-remote-streaming.md
git commit -m "docs: plan private Isaac streaming deployment"
```

- [ ] **Step 3: Commit any verified durable-fact correction separately**

Stage only the exact spec file after reviewing its diff. Never stage unrelated dirty files or any secret/runtime artifact.
