#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: $0 --dry-run or --apply" >&2
    exit 2
}

if [[ $# -ne 1 ]]; then
    usage
fi

case "$1" in
    --dry-run)
        apply_changes=false
        ;;
    --apply)
        apply_changes=true
        ;;
    *)
        usage
        ;;
esac

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source_root=$(cd -- "$script_dir/.." && pwd -P)
install_root=/opt/aloha-foot-pedal
udev_target=/etc/udev/rules.d/99-aloha-foot-pedal.rules
service_target=/etc/systemd/system/aloha-foot-pedal.service
environment_target=/etc/aloha-foot-pedal.env

required_sources=(
    "$source_root/aloha/__init__.py"
    "$source_root/aloha/foot_pedal_relay.py"
    "$source_root/scripts/foot_pedal_relay.py"
    "$source_root/deploy/foot_pedal/99-aloha-foot-pedal.rules"
    "$source_root/deploy/foot_pedal/aloha-foot-pedal.service"
    "$source_root/deploy/foot_pedal/foot-pedal.env.example"
)

for source_path in "${required_sources[@]}"; do
    if [[ ! -f "$source_path" ]]; then
        echo "missing source file: $source_path" >&2
        exit 1
    fi
done

print_plan() {
    echo "install $source_root/aloha -> $install_root/aloha"
    echo "install $source_root/scripts/foot_pedal_relay.py -> $install_root/scripts/foot_pedal_relay.py"
    echo "install udev rule -> $udev_target"
    echo "install systemd unit -> $service_target"
    echo "create environment when absent -> $environment_target"
    echo "add eii to input group"
    echo "udevadm control --reload-rules"
    echo "udevadm trigger --subsystem-match=input --action=change"
    echo "systemctl daemon-reload"
    echo "service remains disabled until pedal enrollment"
}

print_plan
if [[ "$apply_changes" == false ]]; then
    exit 0
fi

if [[ $(id -u) -ne 0 ]]; then
    echo "--apply must run as root" >&2
    exit 1
fi

if [[ $(hostname) != "ubuntu" ]]; then
    echo "refusing to install: expected machine 101 hostname ubuntu" >&2
    exit 1
fi

resolved_aloha=$(runuser -u eii -- ssh -G aloha 2>/dev/null | awk '$1 == "hostname" {print $2; exit}')
if [[ "$resolved_aloha" != "192.168.1.103" ]]; then
    echo "refusing to install: SSH host aloha must resolve to 192.168.1.103" >&2
    exit 1
fi

backup_root="/var/backups/aloha-foot-pedal/$(date +%Y%m%d-%H%M%S)"
temporary_paths=()

cleanup() {
    local temporary_path
    for temporary_path in "${temporary_paths[@]}"; do
        rm -f -- "$temporary_path"
    done
}
trap cleanup EXIT

backup_existing() {
    local target=$1
    if [[ -e "$target" ]]; then
        local backup_path="$backup_root$target"
        install -D -m 0600 -- "$target" "$backup_path"
    fi
}

install_atomic() {
    local source=$1
    local target=$2
    local mode=$3
    local temporary_path="${target}.tmp.$$"
    backup_existing "$target"
    install -D -m "$mode" -- "$source" "$temporary_path"
    temporary_paths+=("$temporary_path")
    mv -f -- "$temporary_path" "$target"
}

install_atomic "$source_root/aloha/__init__.py" "$install_root/aloha/__init__.py" 0644
install_atomic "$source_root/aloha/foot_pedal_relay.py" "$install_root/aloha/foot_pedal_relay.py" 0644
install_atomic "$source_root/scripts/foot_pedal_relay.py" "$install_root/scripts/foot_pedal_relay.py" 0755
install_atomic "$source_root/deploy/foot_pedal/99-aloha-foot-pedal.rules" "$udev_target" 0644
install_atomic "$source_root/deploy/foot_pedal/aloha-foot-pedal.service" "$service_target" 0644

if [[ ! -e "$environment_target" ]]; then
    install_atomic "$source_root/deploy/foot_pedal/foot-pedal.env.example" "$environment_target" 0600
fi

usermod -a -G input eii
udevadm control --reload-rules
udevadm trigger --subsystem-match=input --action=change
systemctl daemon-reload
systemctl disable --now aloha-foot-pedal.service >/dev/null 2>&1 || true

echo "installation complete; service disabled pending pedal enrollment"
