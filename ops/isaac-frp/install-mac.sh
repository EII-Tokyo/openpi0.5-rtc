#!/usr/bin/env bash
set -euo pipefail

die() {
  printf 'install-mac: %s\n' "$*" >&2
  exit 1
}

[[ $# -eq 1 ]] || die "usage: $0 SETUP_BUNDLE"
bundle_dir=$(cd -- "$1" 2>/dev/null && pwd -P) || die "setup bundle is not a directory: $1"
target_home=${ISAAC_FRP_TEST_HOME:-${HOME:?HOME is not set}}
machine_arch=${ISAAC_FRP_TEST_ARCH:-$(uname -m)}

case "$machine_arch" in
  arm64) frp_arch=darwin_arm64 ;;
  x86_64) frp_arch=darwin_amd64 ;;
  *) die "unsupported Mac architecture: $machine_arch" ;;
esac

for name in frps-token endpoint-secrets.env; do
  [[ -s "$bundle_dir/$name" ]] || die "missing required bundle file: $bundle_dir/$name"
done

# The generated file contains only three shell-safe hexadecimal assignments.
# shellcheck disable=SC1091
source "$bundle_dir/endpoint-secrets.env"
: "${SSH_STCP_SECRET:?missing SSH_STCP_SECRET}"
: "${WEBRTC_SIGNAL_STCP_SECRET:?missing WEBRTC_SIGNAL_STCP_SECRET}"
: "${WEBRTC_MEDIA_SUDP_SECRET:?missing WEBRTC_MEDIA_SUDP_SECRET}"

binary_target="$target_home/.local/bin/frpc"
config_dir="$target_home/.config/frp"
config_target="$config_dir/frpc.toml"
token_target="$config_dir/frps-token"
for target in "$binary_target" "$config_target" "$token_target"; do
  [[ ! -e "$target" ]] || die "refusing to overwrite existing target: $target"
done

work_dir=$(mktemp -d)
cleanup() {
  rm -rf -- "$work_dir"
}
trap cleanup EXIT

archive=${ISAAC_FRP_ARCHIVE_PATH:-$work_dir/frp_0.66.0_${frp_arch}.tar.gz}
if [[ -z ${ISAAC_FRP_ARCHIVE_PATH:-} ]]; then
  curl -fL --retry 3 --retry-delay 2 \
    -o "$archive" \
    "https://github.com/fatedier/frp/releases/download/v0.66.0/frp_0.66.0_${frp_arch}.tar.gz"
fi
[[ -s "$archive" ]] || die "frp archive is empty: $archive"
tar -xzf "$archive" -C "$work_dir"
candidate="$work_dir/frp_0.66.0_${frp_arch}/frpc"
[[ -x "$candidate" ]] || die "frpc missing from archive"
[[ $("$candidate" --version) == 0.66.0 ]] || die "archive contains unexpected frpc version"

umask 077
install -d -m 0700 "$target_home/.local/bin" "$config_dir"
install -m 0755 "$candidate" "$binary_target"
install -m 0600 "$bundle_dir/frps-token" "$token_target"

{
  printf '%s\n' \
    'user = "eii-mac"' \
    'serverAddr = "18.183.41.244"' \
    'serverPort = 7000' \
    'loginFailExit = false' \
    'udpPacketSize = 1500' \
    '' \
    'auth.method = "token"' \
    'auth.tokenSource.type = "file"'
  printf 'auth.tokenSource.file.path = "%s"\n' "$token_target"
  printf '%s\n' \
    '' \
    'transport.tls.enable = true' \
    'log.to = "console"' \
    'log.level = "info"' \
    '' \
    '[[visitors]]' \
    'name = "isaac-ssh-visitor"' \
    'type = "stcp"' \
    'serverUser = "eii-103-isaac"' \
    'serverName = "isaac-ssh"'
  printf 'secretKey = "%s"\n' "$SSH_STCP_SECRET"
  printf '%s\n' \
    'bindAddr = "127.0.0.1"' \
    'bindPort = 22022' \
    '' \
    '[[visitors]]' \
    'name = "isaac-webrtc-signal-visitor"' \
    'type = "stcp"' \
    'serverUser = "eii-103-isaac"' \
    'serverName = "isaac-webrtc-signal"'
  printf 'secretKey = "%s"\n' "$WEBRTC_SIGNAL_STCP_SECRET"
  printf '%s\n' \
    'bindAddr = "127.0.0.1"' \
    'bindPort = 49100' \
    '' \
    '[[visitors]]' \
    'name = "isaac-webrtc-media-visitor"' \
    'type = "sudp"' \
    'serverUser = "eii-103-isaac"' \
    'serverName = "isaac-webrtc-media"'
  printf 'secretKey = "%s"\n' "$WEBRTC_MEDIA_SUDP_SECRET"
  printf '%s\n' \
    'bindAddr = "127.0.0.1"' \
    'bindPort = 47998'
} >"$config_target"

chmod 0600 "$config_target" "$token_target"
"$binary_target" verify -c "$config_target"
printf 'installed frpc %s for %s\n' "$("$binary_target" --version)" "$machine_arch"
printf 'configuration: %s\n' "$config_target"
