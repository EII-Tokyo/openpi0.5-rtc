"""Exercise pedal events without opening a physical input device."""
import os
import time

from openpi.robot.aloha_real.foot_switch import FootSwitch


def test_pedal_press_release_and_repeat_do_not_duplicate_commands():
    pedal = FootSwitch(None)
    reader, writer = os.pipe()
    pedal.fd = reader
    os.set_blocking(reader, False)
    try:
        now = time.time()
        seconds = int(now)
        microseconds = int((now - seconds) * 1e6)
        events = [(1, 48, 1), (1, 48, 2), (1, 48, 1), (1, 48, 0)]
        os.write(writer, b''.join(pedal._event.pack(seconds, microseconds, *event) for event in events))
        assert pedal.poll() == 'b'
        assert pedal.poll() is None
        os.write(writer, pedal._event.pack(seconds, microseconds, 1, 48, 1))
        assert pedal.poll() == 'b'
    finally:
        pedal.close()
        os.close(writer)


def test_old_events_and_other_keys_do_not_start_recording():
    pedal = FootSwitch(None)
    reader, writer = os.pipe()
    pedal.fd = reader
    os.set_blocking(reader, False)
    try:
        old = int(time.time()) - 10
        os.write(writer, pedal._event.pack(old, 0, 1, 48, 1))
        os.write(writer, pedal._event.pack(old, 0, 1, 48, 0))
        os.write(writer, pedal._event.pack(int(time.time()), 0, 1, 30, 1))
        assert pedal.poll() is None
        os.close(writer)
        writer = None
        assert pedal.poll() is None
        assert pedal.fd is None
    finally:
        pedal.close()
        if writer is not None:
            os.close(writer)


def test_missing_pedal_leaves_keyboard_control_available(tmp_path):
    pedal = FootSwitch(str(tmp_path / 'missing-device'))
    assert pedal.fd is None
    assert pedal.poll() is None
    pedal.close()
