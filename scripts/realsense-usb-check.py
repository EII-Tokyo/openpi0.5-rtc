#!/usr/bin/env python3
"""Read-only, one-second RealSense USB link monitor for the four ALOHA cameras."""

import argparse
from datetime import datetime
from pathlib import Path
import re
import subprocess
import sys
import time

CONFIG = Path(__file__).resolve().parents[1] / "config/realsense_usb.yaml"
LABELS = {
    "camera_high": "高位", "camera_low": "低位",
    "camera_wrist_right": "右腕", "camera_wrist_left": "左腕",
}


def expected_cameras(config):
    """Read only the camera_instances name/serial pairs; no YAML dependency."""
    text = config.read_text(encoding="utf-8").split("camera_instances:", 1)
    if len(text) != 2:
        raise ValueError("配置缺少 camera_instances")
    result = {}
    name = None
    for line in text[1].splitlines():
        match = re.match(r"\s*(?:-\s*)?name:\s*(camera_\w+)", line)
        if match:
            name = match.group(1)
        match = re.match(r'\s*serial_no:\s*["\']?([0-9]+)', line)
        if match and name in LABELS:
            result[name] = match.group(1)
            name = None
    if set(result) != set(LABELS) or len(set(result.values())) != 4:
        raise ValueError("配置中的四台相机名称或序列号不完整／重复")
    return result


def parse_devices(output):
    devices = {}
    for block in re.split(r"(?=^Device info:\s*$)", output, flags=re.MULTILINE):
        fields = dict(re.findall(r"^\s*(Serial Number|Physical Port|Usb Type Descriptor)\s*:\s*(.*?)\s*$", block, re.MULTILINE))
        serial = fields.get("Serial Number")
        if serial:
            devices[serial] = fields
    return devices


def usb_port(physical):
    # /sys/.../4-3/4-3.4/4-3.4:1.0/video4linux/video6 -> 4-3.4
    matches = re.findall(r"/(\d+-[\d.]+):\d+\.\d+(?:/|$)", physical)
    return matches[-1] if matches else "未知"


def inspect(config, runner=subprocess.run):
    completed = runner(["rs-enumerate-devices"], capture_output=True, text=True, timeout=8, check=True)
    devices = parse_devices(completed.stdout)
    rows = []
    for name, serial in expected_cameras(config).items():
        device = devices.get(serial)
        if device is None:
            rows.append((LABELS[name], serial, "未发现", "-", "断开／枚举失败"))
            continue
        usb = device.get("Usb Type Descriptor", "未知")
        port = usb_port(device.get("Physical Port", ""))
        status = "正常" if usb.startswith("3.") else "降速：检查USB3线缆／端口／供电" if usb.startswith("2.") else "速度未知"
        rows.append((LABELS[name], serial, usb, port, status))
    return rows


def format_screen(rows, *, timestamp, error=None, terminal=False):
    """Build one complete frame. ANSI control codes are only for a terminal."""
    lines = [
        f"RealSense D405 USB 连接诊断   {timestamp}   每秒刷新 · Ctrl+C 退出",
        "运行采集时频繁枚举相机可能增加 USB 负载。",
        "",
        "相机       序列号         USB    物理端口          状态",
        "─" * 69,
    ]
    if error is not None:
        lines.append(f"枚举失败：{error}（不能据此认定四台都断开）")
    else:
        for label, serial, usb, port, status in rows:
            line = f"{label:<8} {serial:<14} {usb:<6} {port:<16} {status}"
            if terminal:
                color = "\033[32m" if usb.startswith("3.") else "\033[31m" if usb.startswith("2.") or usb == "未发现" else "\033[33m"
                line = color + line + "\033[0m"
            lines.append(line)
        count = sum(row[2].startswith("3.") for row in rows)
        lines.extend(("", f"结果：{count}/4 台以 USB 3 连接"))
    frame = "\n".join(lines) + "\n"
    return ("\033[H\033[2J" if terminal else "") + frame


def main(argv=None):
    parser = argparse.ArgumentParser(description="每秒只读诊断四台D405 USB连接；Ctrl+C退出")
    parser.add_argument("--once", action="store_true", help="只诊断一次")
    parser.add_argument("--interval", type=float, default=1.0, help="刷新间隔秒数，默认1")
    parser.add_argument("--config", type=Path, default=CONFIG, help="相机配置路径")
    args = parser.parse_args(argv)
    if args.interval < 1:
        parser.error("刷新间隔不得小于1秒")
    try:
        expected_cameras(args.config)
    except (OSError, ValueError) as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2
    terminal = sys.stdout.isatty()
    try:
        while True:
            started = time.monotonic()
            try:
                rows = inspect(args.config)
            except (OSError, subprocess.SubprocessError) as exc:
                frame = format_screen([], timestamp=datetime.now().strftime("%F %T"), error=exc, terminal=terminal)
                sys.stdout.write(frame)
                sys.stdout.flush()
                if args.once:
                    return 1
            else:
                frame = format_screen(rows, timestamp=datetime.now().strftime("%F %T"), terminal=terminal)
                sys.stdout.write(frame)
                sys.stdout.flush()
                if args.once:
                    return 0 if all(row[2].startswith("3.") for row in rows) else 1
            time.sleep(max(0, args.interval - (time.monotonic() - started)))
    except KeyboardInterrupt:
        print("\n监控已结束。")
        return 0


if __name__ == "__main__":
    sys.exit(main())
