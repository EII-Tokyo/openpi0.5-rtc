#!/usr/bin/env bash
set -euo pipefail

die() {
  printf 'install-103: %s\n' "$*" >&2
  exit 1
}

[[ $# -eq 1 ]] || die "usage: $0 DEPLOYMENT_SOURCE"

source_dir=$(cd -- "$1" 2>/dev/null && pwd -P) || die "deployment source is not a directory: $1"
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
test_root=${ISAAC_FRP_TEST_ROOT:-}

for name in frpc frpc.toml frps-token; do
  [[ -f "$source_dir/$name" ]] || die "missing required deployment file: $source_dir/$name"
done
[[ -x "$source_dir/frpc" ]] || die "frpc is not executable: $source_dir/frpc"
[[ -f "$script_dir/frpc.service" ]] || die "missing service unit: $script_dir/frpc.service"

if [[ -n "$test_root" ]]; then
  prefix=${test_root%/}
  install -d -m 0755 "$prefix/usr/local/bin" "$prefix/etc/systemd/system"
  install -d -m 0750 "$prefix/etc/frp"
  install -m 0755 "$source_dir/frpc" "$prefix/usr/local/bin/frpc"
  install -m 0640 "$source_dir/frpc.toml" "$prefix/etc/frp/frpc.toml"
  install -m 0640 "$source_dir/frps-token" "$prefix/etc/frp/frps-token"
  install -m 0644 "$script_dir/frpc.service" "$prefix/etc/systemd/system/frpc.service"
  "$prefix/usr/local/bin/frpc" verify -c "$prefix/etc/frp/frpc.toml"
  exit 0
fi

[[ $EUID -eq 0 ]] || die "run this installer with sudo"

for target in \
  /usr/local/bin/frpc \
  /etc/frp/frpc.toml \
  /etc/frp/frps-token \
  /etc/systemd/system/frpc.service; do
  [[ ! -e "$target" ]] || die "refusing to overwrite existing target: $target"
done

if ! getent group frp >/dev/null; then
  groupadd --system frp
fi
if ! getent passwd frp >/dev/null; then
  useradd --system --gid frp --home-dir /nonexistent --shell /usr/sbin/nologin frp
fi

install -d -o root -g frp -m 0750 /etc/frp
install -o root -g root -m 0755 "$source_dir/frpc" /usr/local/bin/frpc
install -o root -g frp -m 0640 "$source_dir/frpc.toml" /etc/frp/frpc.toml
install -o root -g frp -m 0640 "$source_dir/frps-token" /etc/frp/frps-token
install -o root -g root -m 0644 "$script_dir/frpc.service" /etc/systemd/system/frpc.service

runuser -u frp -- /usr/local/bin/frpc verify -c /etc/frp/frpc.toml
systemctl daemon-reload
systemctl enable --now frpc.service

[[ $(systemctl is-enabled frpc.service) == enabled ]] || die "frpc.service is not enabled"
[[ $(systemctl is-active frpc.service) == active ]] || die "frpc.service is not active"
