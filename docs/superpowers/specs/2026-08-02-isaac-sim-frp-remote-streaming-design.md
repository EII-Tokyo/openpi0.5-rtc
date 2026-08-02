# Isaac Sim 5.1 Remote Streaming Through frp Design

## Goal

Run the Isaac Sim 5.1 Full Streaming App on GPU workstation
`192.168.1.103` and access its GUI from a home Mac through the existing AWS
frps server. Both endpoint machines are behind NAT. The 103 frpc service must
start at boot. Isaac Sim itself remains an on-demand service.

This work is simulation-only. It must not start robot containers, connect MCP
tools to the real robot, command robot joints, or change the active simulation
timeline.

## Approved decisions

- Reuse the existing AWS frps `0.66.0` listener on TCP port `7000`.
- Do not modify or restart the existing AWS frps in the primary design.
- Install the matching frpc `0.66.0` on 103 and the Mac.
- Use private STCP/SUDP visitors instead of exposing Isaac WebRTC ports on the
  AWS public interface.
- Install Isaac Sim at
  `/home/eii/Applications/isaacsim-5.1.0` on 103.
- Keep Isaac Sim 5.1.0 for project compatibility even though NVIDIA now marks
  the release unsupported.
- Start frpc automatically at boot, but start Isaac Sim only on demand.
- Use NVIDIA Isaac Sim WebRTC Streaming Client on macOS.

## Verified baseline

### AWS

- SSH alias: `ec2`, host `18.183.41.244`, user `ubuntu`.
- Architecture: `x86_64`.
- frps version: `0.66.0`.
- Service: active `frps.service`.
- Unit: `/etc/systemd/system/frps.service`.
- Config: `/etc/frp/frps.toml`.
- Existing bind port: TCP `7000`.
- Existing frps configuration uses token authentication and an `allowPorts`
  list for public TCP/UDP proxies.
- Existing public proxy and application listeners must remain intact.

### 103

- SSH alias: `aloha`, host `192.168.1.103`, user `eii`.
- OS: Ubuntu 24.04.3 LTS, `x86_64`, systemd 255.
- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB VRAM.
- Driver: `580.159.03`.
- NVENC library package: `libnvidia-encode-580 580.159.03`.
- RAM: 62 GiB.
- Available storage on the target filesystem: 317 GiB.
- Display manager and a seat0 desktop session are active.
- No frpc binary or active frpc service exists.
- No Isaac Sim installation was found in the project, Applications directory,
  Omniverse package directory, command path, or Docker image list.
- The official Isaac Sim 5.1.0 ZIP is reachable from 103 and reports
  `Content-Length: 8768419777`.

## Network architecture

```text
103
  Isaac SSH             127.0.0.1:22/TCP
  Isaac WebRTC signal   127.0.0.1:49100/TCP
  Isaac WebRTC media    127.0.0.1:47998/UDP
            |
            | frpc provider: STCP + STCP + SUDP
            v
AWS existing frps 18.183.41.244:7000/TCP
            ^
            | Mac frpc visitors: STCP + STCP + SUDP
            |
Mac
  SSH                    127.0.0.1:22022/TCP
  WebRTC signal          127.0.0.1:49100/TCP
  WebRTC media           127.0.0.1:47998/UDP
            |
            v
  NVIDIA Isaac Sim WebRTC Streaming Client -> 127.0.0.1
```

STCP and SUDP do not declare `remotePort`; therefore the existing AWS
`allowPorts` list is not changed and the Isaac streaming endpoints are not
public listeners on AWS. The Mac must run its visitor frpc before SSH or
WebRTC access.

## AWS configuration

The primary design makes no AWS file or service change. The relevant existing
configuration remains conceptually:

```toml
bindPort = 7000
auth.method = "token"
auth.token = "<EXISTING_SECRET_NOT_PRINTED_OR_COMMITTED>"

allowPorts = [
  { start = 20022, end = 20032 },
  { start = 6000, end = 6002 },
]
```

Before endpoint deployment, record SHA-256 and file metadata for
`/etc/frp/frps.toml` and `/etc/systemd/system/frps.service`. After deployment,
recompute both hashes and require them to be identical. Do not restart or
reload `frps.service`.

The existing frps token is provisioned to the two frpc hosts without printing
it in terminal output, writing it into this repository, or including it in
logs. Token rotation is out of scope because it would disrupt existing frpc
clients.

## 103 frpc installation and configuration

### Files and ownership

```text
/usr/local/bin/frpc              frpc 0.66.0 binary
/etc/frp/frpc.toml               root:frp 0640
/etc/frp/frps-token              root:frp 0640
/etc/systemd/system/frpc.service root:root 0644
```

A dedicated, non-login system user and group named `frp` run the service.
Real secrets never enter Git. Generate three independent 32-byte random
secret keys for SSH, WebRTC signaling, and WebRTC media.

### `/etc/frp/frpc.toml`

```toml
user = "eii-103-isaac"
serverAddr = "18.183.41.244"
serverPort = 7000
loginFailExit = false
udpPacketSize = 1500

auth.method = "token"
auth.tokenSource.type = "file"
auth.tokenSource.file.path = "/etc/frp/frps-token"

transport.tls.enable = true
log.to = "console"
log.level = "info"

[[proxies]]
name = "isaac-ssh"
type = "stcp"
secretKey = "<SSH_STCP_SECRET>"
localIP = "127.0.0.1"
localPort = 22
allowUsers = ["eii-mac"]

[[proxies]]
name = "isaac-webrtc-signal"
type = "stcp"
secretKey = "<WEBRTC_SIGNAL_STCP_SECRET>"
localIP = "127.0.0.1"
localPort = 49100
allowUsers = ["eii-mac"]

[[proxies]]
name = "isaac-webrtc-media"
type = "sudp"
secretKey = "<WEBRTC_MEDIA_SUDP_SECRET>"
localIP = "127.0.0.1"
localPort = 47998
allowUsers = ["eii-mac"]
```

### `/etc/systemd/system/frpc.service`

```ini
[Unit]
Description=frp Client for Private Isaac Sim Access
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=0

[Service]
Type=simple
User=frp
Group=frp
ExecStart=/usr/local/bin/frpc -c /etc/frp/frpc.toml
Restart=always
RestartSec=5
UMask=0077
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectControlGroups=true
RestrictSUIDSGID=true
RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6

[Install]
WantedBy=multi-user.target
```

Validate the TOML with `frpc verify -c /etc/frp/frpc.toml` before running
`systemctl enable --now frpc.service`. A config verification failure is a hard
stop. `systemctl is-enabled`, `systemctl is-active`, bounded journal output,
and the AWS frps client view must all confirm the service state.

An actual 103 reboot is not implicit. It requires a separate approval because
it affects the robot workstation. Without a reboot, `is-enabled` and the
multi-user target link verify systemd boot registration, but not a real boot
cycle.

## Mac frpc visitor configuration

Install frpc `0.66.0` for the Mac architecture. The user must confirm whether
the Mac is Apple Silicon (`darwin_arm64`) or Intel (`darwin_amd64`) before the
download command is frozen.

Suggested files:

```text
~/.local/bin/frpc
~/.config/frp/frpc.toml       mode 0600
~/.config/frp/frps-token      mode 0600
```

### `~/.config/frp/frpc.toml`

```toml
user = "eii-mac"
serverAddr = "18.183.41.244"
serverPort = 7000
loginFailExit = false
udpPacketSize = 1500

auth.method = "token"
auth.tokenSource.type = "file"
auth.tokenSource.file.path = "<ABSOLUTE_MAC_HOME>/.config/frp/frps-token"

transport.tls.enable = true
log.to = "console"
log.level = "info"

[[visitors]]
name = "isaac-ssh-visitor"
type = "stcp"
serverUser = "eii-103-isaac"
serverName = "isaac-ssh"
secretKey = "<SSH_STCP_SECRET>"
bindAddr = "127.0.0.1"
bindPort = 22022

[[visitors]]
name = "isaac-webrtc-signal-visitor"
type = "stcp"
serverUser = "eii-103-isaac"
serverName = "isaac-webrtc-signal"
secretKey = "<WEBRTC_SIGNAL_STCP_SECRET>"
bindAddr = "127.0.0.1"
bindPort = 49100

[[visitors]]
name = "isaac-webrtc-media-visitor"
type = "sudp"
serverUser = "eii-103-isaac"
serverName = "isaac-webrtc-media"
secretKey = "<WEBRTC_MEDIA_SUDP_SECRET>"
bindAddr = "127.0.0.1"
bindPort = 47998
```

The Mac visitor is initially started on demand:

```bash
~/.local/bin/frpc verify -c ~/.config/frp/frpc.toml
~/.local/bin/frpc -c ~/.config/frp/frpc.toml
```

A macOS LaunchAgent is optional and out of the first deployment scope. The
first acceptance path keeps the terminal visible so connection and SUDP
errors are observable.

## Isaac Sim installation on 103

### Paths

```text
/home/eii/Applications/isaacsim-downloads/
/home/eii/Applications/isaacsim-5.1.0.staging/
/home/eii/Applications/isaacsim-5.1.0/
```

Never unzip directly over the final directory. Refuse installation if either
the staging or final path already exists. Download only from:

```text
https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip
```

Use resumable HTTPS download, record a local SHA-256, run `unzip -t`, extract
to staging, assert the expected launch scripts exist and are executable, then
rename staging to the final directory. Run `post_install.sh` only after these
checks.

The NVIDIA page does not expose a separate published SHA-256 for this ZIP.
HTTPS origin validation, the recorded local hash, exact HTTP content length,
and a complete ZIP integrity test form the available integrity evidence. Do
not treat the multipart S3 ETag as a content hash.

### Compatibility gate

Before the full app, run:

```bash
cd /home/eii/Applications/isaacsim-5.1.0
./isaac-sim.compatibility_check.sh --/app/quitAfter=10 --no-window
```

Require successful GPU, driver, RTX/Kit, storage, and OS checks. Inspect the
bounded log for Vulkan, RTX, NVENC, display, or crash errors. Confirm no
compatibility-check process remains after exit.

### First streaming launch

The first launch is foreground and manually observed:

```bash
cd /home/eii/Applications/isaacsim-5.1.0
./isaac-sim.streaming.sh \
  --/app/livestream/publicEndpointAddress=127.0.0.1 \
  --/app/livestream/port=49100
```

This is the full streaming application, not a trimmed Python experience. It
must reach `Isaac Sim Full Streaming App is loaded.` and listen on TCP 49100
and UDP 47998. It opens without an automatically selected project Stage, and
the timeline remains paused.

Advertising `127.0.0.1` is an engineering adaptation for the Mac-local frp
visitors. NVIDIA documents localhost for a local client and a public endpoint
for direct remote clients, but does not document STCP/SUDP encapsulation.
Therefore the combination is not accepted until a real Mac end-to-end test
passes.

## On-demand Isaac service

After a successful foreground launch, create but do not enable:

```text
/home/eii/.config/systemd/user/isaac-sim-streaming.service
```

```ini
[Unit]
Description=Isaac Sim 5.1 Full Streaming App
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/eii/Applications/isaacsim-5.1.0
ExecStart=/home/eii/Applications/isaacsim-5.1.0/isaac-sim.streaming.sh --/app/livestream/publicEndpointAddress=127.0.0.1 --/app/livestream/port=49100
Restart=on-failure
RestartSec=10
TimeoutStopSec=180

[Install]
WantedBy=default.target
```

Use:

```bash
systemctl --user daemon-reload
systemctl --user start isaac-sim-streaming.service
systemctl --user status isaac-sim-streaming.service
journalctl --user --unit isaac-sim-streaming.service --follow
systemctl --user stop isaac-sim-streaming.service
```

Do not run `systemctl --user enable`. Isaac remains on demand and a normal
File -> Exit is not automatically restarted because `Restart=on-failure` only
handles abnormal exits.

## Deployment sequence and gates

1. Snapshot AWS frps config, unit hashes, status, and listeners.
2. Install frpc `0.66.0` on 103 without starting it.
3. Create protected token and endpoint config files; run `frpc verify`.
4. Enable and start 103 `frpc.service`; verify provider registration.
5. Install and configure the matching Mac frpc; run `frpc verify`.
6. Start the Mac visitor and verify private SSH through
   `127.0.0.1:22022` before any Isaac work.
7. Run a bounded UDP probe through the SUDP visitor before installing Isaac.
   Reject packet loss, one-way traffic, or unstable reconnect behavior.
8. Recompute AWS config and unit hashes. They must match the baseline.
9. Download and install Isaac Sim 5.1.0 through the staging workflow.
10. Run the compatibility checker and inspect bounded evidence.
11. Start the full streaming app in the foreground and verify the two ports.
12. Connect the Mac NVIDIA client to `127.0.0.1`; verify video, keyboard,
    mouse, reconnect, and clean shutdown.
13. Measure latency, packet loss, AWS bandwidth, and stream stability.
14. Only after the foreground test passes, add the disabled user service and
    repeat the start/connect/stop test.

Each gate must pass before the next phase. No robot or project runtime service
is started as part of these tests.

## Acceptance criteria

- AWS `frps.service` remains active and its config/unit hashes are unchanged.
- No new public AWS listener is created for TCP 49100 or UDP 47998.
- Existing AWS listeners and clients remain available.
- 103 runs exact frpc `0.66.0`; config verification passes.
- 103 `frpc.service` is active and enabled for `multi-user.target`.
- Mac runs exact frpc `0.66.0`; config verification passes.
- STCP SSH works only through Mac localhost port 22022.
- SUDP probe is bidirectional and stable enough for the WebRTC trial.
- Isaac compatibility check passes on RTX 5090 with NVENC available.
- Full Streaming App reaches its loaded marker and exposes the expected local
  TCP/UDP ports.
- The Mac client receives a usable picture and keyboard/mouse control through
  `127.0.0.1`.
- Isaac starts and stops through the user service without orphan Kit
  processes; the service remains disabled at boot.
- The timeline stays paused and no real robot service or device is touched.

## Security properties and limitations

- frps token authentication controls frpc registration, while independent
  STCP/SUDP secret keys control visitor access.
- Config and token files are never committed and must have restrictive modes.
- Global frp TLS is explicitly enabled. The existing frps has no pinned CA in
  the inspected non-secret configuration, so encryption does not establish
  strong server identity verification. Adding a CA or rotating authentication
  would affect existing clients and is outside this deployment.
- Isaac WebRTC endpoints do not provide their own user authentication. They
  are protected by localhost binding and secret frp visitors rather than
  public AWS listeners.
- Media is relayed through AWS. Latency, bandwidth, and AWS egress cost must be
  measured; success cannot be inferred from a connected signaling channel.
- NVIDIA documents that only one streaming client can use an Isaac instance
  at a time.
- Isaac Sim 5.1 is no longer supported upstream. It is retained solely for
  project compatibility; a 6.x migration is a separate project.

## Failure handling and fallback

- If frpc config verification fails, do not install the unit or start frpc.
- If private SSH fails, stop before SUDP or Isaac work.
- If SUDP fails or has unacceptable media performance, stop and report. Do
  not modify AWS automatically.
- The first fallback is a separately reviewed design that adds a dedicated
  `frps-isaac` instance with a separate token and port, leaving the existing
  frps untouched.
- A public TCP/UDP proxy on AWS is the last fallback because Isaac streaming
  endpoints have no native authentication. It would require explicit AWS
  firewall restrictions and separate approval.
- If the compatibility checker fails, do not start the full streaming app.
- If the stream app starts but the Mac has no image, preserve bounded logs,
  confirm signaling and UDP separately, and stop before changing ports or
  firewalls.

## Rollback

Primary AWS rollback is unnecessary because AWS is unchanged. Verify its
post-run hashes and status against the baseline.

On 103, rollback stops and disables `frpc.service` and restores any preexisting
paths from timestamped backups. Stop the Isaac user service if created. Keep
downloaded archives, the versioned Isaac directory, configs, and logs in
place for inspection; deleting them requires separate approval.

On the Mac, stop the visitor process. Keep protected config files for review;
removal is a separate user action.

## Sources

- NVIDIA Isaac Sim 5.1 Livestream Clients:
  <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/manual_livestream_clients.html>
- NVIDIA Isaac Sim 5.1 Quick Install:
  <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/quick-install.html>
- frp STCP and SUDP:
  <https://gofrp.org/en/docs/features/stcp-sudp/>
- frp secure visitor example:
  <https://gofrp.org/en/docs/examples/stcp/>
- frp authentication and token source:
  <https://gofrp.org/en/docs/features/common/authentication/>
- frp systemd guidance:
  <https://gofrp.org/en/docs/setup/systemd/>
- frp v0.66.0 release:
  <https://github.com/fatedier/frp/releases/tag/v0.66.0>
