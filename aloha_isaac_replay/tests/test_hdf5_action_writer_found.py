from __future__ import annotations

from pathlib import Path


def test_rlt_hdf5_action_writer_is_located() -> None:
    path = Path("examples/aloha_real/rlt_key_region_recorder.py")
    text = path.read_text()
    assert "root.create_dataset(\"action\"" in text
    assert "action=_extract_action_array(action, \"actions\", \"action\")" in text
    assert "reference_action=_extract_action_array(" in text
    assert "\"reference_actions\", \"reference_action\", \"vla_reference_action\"" in text
    assert "def _write_hdf5" in text


def test_standard_hdf5_savers_store_runtime_action_key() -> None:
    saver = Path("examples/aloha_real/h5df_saver.py").read_text()
    utils = Path("examples/aloha_real/hdf5_utils.py").read_text()
    video_saver = Path("examples/aloha_real/video_hdf5_saver.py").read_text()
    assert "action[\"actions\"]" in saver
    assert "data_dict[\"/action\"]" in utils
    assert "root.create_dataset(\"action\"" in utils
    assert "self._actions.append(np.asarray(action[\"actions\"], dtype=np.float32))" in video_saver
