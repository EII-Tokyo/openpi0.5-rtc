"""Host-103 monitor: stop only after completed training and verified final S3 inference export."""
import datetime as dt
import json
from pathlib import Path
import subprocess
import time

ROOT = Path('/home/eii/vast-training-backups/hxz15-no-mem-5k-20261118')
STATE = ROOT / 'completion-monitor.json'
INSTANCE = 61149764
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-p', '51564', 'root@65.58.62.178']
PREFIX = ('inference/eii_hxz_15repo_cam4_fullft_h200x2_448_no_mem_nearw4_5k_20261118/'
          'hxz15_h200x2_448_no_mem_bs128_ga2_5k_20261118/4999/')
REPORT = ROOT / 'final-cost-report.txt'


def run(args, timeout=90):
    return subprocess.check_output(args, text=True, stderr=subprocess.PIPE, timeout=timeout)


def remote(command):
    output = run(SSH + [command])
    # Login banners may precede machine-readable output.
    return output


def aws(*args):
    return run(['/usr/local/bin/aws', '--region', 'ap-northeast-1', *args])


def parse_json(output):
    for i, char in enumerate(output):
        if char == '{':
            try:
                return json.loads(output[i:])
            except json.JSONDecodeError:
                pass
    raise ValueError('No complete JSON object returned')


def write_state(state):
    state['checked_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
    tmp = STATE.with_suffix('.next')
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(STATE)


def verified_final():
    if not (ROOT / '4999' / '.s3-inference-complete').exists():
        return False
    body = aws('s3', 'cp', 's3://openpi-tokyo/' + PREFIX + '_inference_manifest.json', '-', '--only-show-errors')
    manifest = json.loads(body)
    assert manifest['type'] == 'inference_only' and manifest['step'] == 4999
    files = manifest['files']
    paths = {item['path'] for item in files}
    assert any(p.startswith('params/') for p in paths)
    assert any(p.startswith('assets/') and p.endswith('norm_stats.json') for p in paths)
    assert 'inference_config.json' in paths
    assert not any(p.startswith('train_state/') for p in paths)
    for item in files:
        relative = Path(item['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        head = json.loads(aws('s3api', 'head-object', '--bucket', 'openpi-tokyo', '--key', PREFIX + item['path']))
        assert head['ContentLength'] == item['size']
        assert head.get('Metadata', {}).get('sha256') == item['sha256']
    (ROOT / 'final-s3-verified.json').write_text(body)
    return True


def snapshot():
    command = '''vastai show instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" --raw | python3 -c 'import sys,json; d=json.load(sys.stdin); keys=["id","actual_status","intended_status","start_date","dph_base","dph_total","storage_total_cost","internet_up_cost_per_tb","internet_down_cost_per_tb"]; print(json.dumps({k:d.get(k) for k in keys}))' '''
    value = parse_json(remote(command))
    assert value['id'] == INSTANCE
    return value


def report(state):
    rate = state['instance_before_stop']['dph_total']
    stop = state['stop_requested_epoch']
    start = state['instance_before_stop']['start_date']
    hours = (stop - start) / 3600
    # Training service started 2026-11-18 01:18:32 UTC, excluding preparation/benchmarks.
    train_start = dt.datetime(2026, 11, 18, 1, 18, 32, tzinfo=dt.timezone.utc).timestamp()
    train_hours = (state['training_completed_epoch'] - train_start) / 3600
    yen_time = dt.datetime.fromtimestamp(stop, dt.timezone(dt.timedelta(hours=9))).isoformat()
    status = state['phase']
    REPORT.write_text(
        f'Vast.ai 实例 {INSTANCE} 训练结束费用报告\n'
        f'状态：{status}\n停机请求时间（日本）：{yen_time}\n'
        f'最终推理模型：s3://openpi-tokyo/{PREFIX}\n最终 S3 文件清单、大小和 SHA256 元数据已复核。\n'
        f'实例自创建至停机请求：{hours:.3f} 小时\n双 GPU 加存储租价：${rate:.6f}/小时\n'
        f'本实例总租用费估算：${hours * rate:.2f}（包含准备、测试和等待上传）\n'
        f'正式 5000 步训练运行时长：{train_hours:.3f} 小时\n对应租用费估算：${train_hours * rate:.2f}\n'
        '以上按实际记录时间和租价计算，非账单结算金额；不含额外流量和 S3 费用。\n'
        '实例密钥缺少 billing_read 权限，无法取得最终账单。\n'
        f'停止后保留磁盘，存储仍约 ${state["instance_before_stop"]["storage_total_cost"]:.6f}/小时。\n'
        'API 接受停机指令与平台确认 stopped 状态会分别记录；SSH 断开本身不当作停机证明。\n'
    )


def main():
    state = json.loads(STATE.read_text()) if STATE.exists() else {'phase': 'watching_training'}
    if state['phase'] == 'stopped_confirmed':
        return
    if state.get('stop_accepted'):
        try:
            current = snapshot()
            if current['actual_status'] == 'stopped':
                state['phase'] = 'stopped_confirmed'
            elif current['intended_status'] == 'stopped':
                state['phase'] = 'stop_accepted_platform_stopping'
            else:
                raise RuntimeError('Stop request accepted but instance not stopping; manual review required')
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            state['phase'] = 'stop_accepted_ssh_offline_confirmation_unavailable'
        write_state(state)
        report(state)
        return
    probe = '''python3 - <<'INNER'
import json,pathlib
p=pathlib.Path('/workspace/logs/no_mem_5k-status.json'); m=pathlib.Path('/workspace/logs/no_mem_5k-metrics.jsonl')
print(json.dumps({'status':json.loads(p.read_text()) if p.exists() else None,'status_mtime':p.stat().st_mtime if p.exists() else None,'latest_metric':json.loads(m.read_text().splitlines()[-1]) if m.exists() else None}))
INNER'''
    status = parse_json(remote(probe))
    state['latest_metric'] = status['latest_metric']
    if not status['status']:
        state['phase'] = 'watching_training'
        write_state(state)
        return
    if status['status']['state'] != 'COMPLETED':
        state['phase'] = 'training_not_completed_no_stop'
        write_state(state)
        return
    state['training_completed_epoch'] = status['status_mtime']
    if not verified_final():
        state['phase'] = 'waiting_for_final_verified_s3_export'
        write_state(state)
        subprocess.run(['systemctl', '--user', 'start', '--no-block', 'vast-hxz15-no-mem-5k-20261118-backup.service'], check=True)
        return
    # Refuse to stop if another GPU workload has appeared.
    processes = remote('nvidia-smi --query-compute-apps=pid --format=csv,noheader')
    assert not any(line.strip().isdigit() for line in processes.splitlines()), 'GPU process still running'
    for filename in ['no_mem_5k.log', 'no_mem_5k-metrics.jsonl', 'no_mem_5k-status.json']:
        run(['scp', '-q', '-P', '51564', 'root@65.58.62.178:/workspace/logs/' + filename, str(ROOT / filename)])
    state['instance_before_stop'] = snapshot()
    state['stop_requested_epoch'] = time.time()
    state['phase'] = 'final_verified_sending_stop'
    write_state(state)
    report(state)
    output = remote('vastai stop instance 61149764 --api-key "$CONTAINER_API_KEY"')
    if 'stopping instance 61149764' not in output:
        raise RuntimeError('Vast CLI did not confirm acceptance of stop request')
    state['stop_accepted'] = True
    state['phase'] = 'stop_accepted_pending_platform_confirmation'
    write_state(state)
    report(state)
    subprocess.run(['notify-send', 'Vast 训练完成，S3 已校验，停机指令已接受', '费用报告：' + str(REPORT)], check=False)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Never log credentials or remote stderr; errors block automatic stopping.
        (ROOT / 'monitor-last-error.json').write_text(json.dumps({'time': time.time(), 'type': type(exc).__name__}))
        raise SystemExit('Monitor check failed: ' + type(exc).__name__)
