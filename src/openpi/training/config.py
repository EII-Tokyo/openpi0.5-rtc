"""See _CONFIGS for the list of available configs."""

from collections.abc import Sequence
import dataclasses
import difflib
import pathlib
from typing import Any, Literal, TypeAlias

import flax.nnx as nnx
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
from openpi.data import transforms as _transforms

Filter: TypeAlias = nnx.filterlib.Filter

_TROSSEN_RESET_POSE = {"reset_pose": [0, -1.5, 1.5, 0, 0, 0]}
_PI05_BASE_PARAMS = "gs://openpi-assets/checkpoints/pi05_base/params"

_TWIST_AND_STATIC_REPO_IDS = [
    "lyl472324464/2026-03-09-inference-with-and-without-cap",
    "lyl472324464/2026-03-09-no-cap-inference",
    "lyl472324464/2026-03-05-two-direction",
    "lyl472324464/2026-03-04-one-direction",
    "lyl472324464/2026-02-03-no-cap-and-direction",
    "lyl472324464/2026-01-28-twist-many-bottle",
    "lyl472324464/2026-01-20-twist-one-bottle",
    "lyl472324464/2025-12-23-twist-one-bottle",
    "lyl472324464/2025-12-10-twist-one-bottle",
    "lyl472324464/2025-11-26-twist-two-bottles",
    "lyl472324464/2025-11-18-twist-two-bottles",
    "lyl472324464/2025-11-14-twist-two-bottles",
    "lyl472324464/2025-11-06-twist-many-bottles",
    "lyl472324464/2025-09-15-twist-one-bottle-no-box-in-the-front",
    "lyl472324464/aloha_static_battery",
    "lyl472324464/aloha_static_candy",
    "lyl472324464/aloha_static_coffee",
    "lyl472324464/aloha_static_coffee_new",
    "lyl472324464/aloha_static_cups_open",
    "lyl472324464/aloha_static_fork_pick_up",
    "lyl472324464/aloha_static_pingpong_test",
    "lyl472324464/aloha_static_pro_pencil",
    "lyl472324464/aloha_static_screw_driver",
    "lyl472324464/aloha_static_tape",
    "lyl472324464/aloha_static_thread_velcro",
    "lyl472324464/aloha_static_towel",
    "lyl472324464/aloha_static_vinh_cup",
    "lyl472324464/aloha_static_vinh_cup_left",
    "lyl472324464/aloha_static_ziploc_slide",
]

_TWIST_ONLY_REPO_IDS = [
    "lyl472324464/2026-03-09-inference-with-and-without-cap",
    "lyl472324464/2026-03-09-no-cap-inference",
    "lyl472324464/2026-03-05-two-direction",
    "lyl472324464/2026-03-04-one-direction",
    "lyl472324464/2026-02-03-no-cap-and-direction",
    "lyl472324464/2026-01-28-twist-many-bottle",
    "lyl472324464/2026-01-20-twist-one-bottle",
    "lyl472324464/2025-12-23-twist-one-bottle",
    "lyl472324464/2025-12-10-twist-one-bottle",
    "lyl472324464/2025-11-26-twist-two-bottles",
    "lyl472324464/2025-11-18-twist-two-bottles",
    "lyl472324464/2025-11-14-twist-two-bottles",
    "lyl472324464/2025-11-06-twist-many-bottles",
    "lyl472324464/2025-09-15-twist-one-bottle-no-box-in-the-front",
]

_TWIST_WATER_TEAR_REPO_IDS = [
    "lyl472324464/2026.04.10_twist_tear_water-direction_tabacco-top",
    "lyl472324464/2026.04.10_twist-and-tear-and-water-direction-and-tabacco-top",
    "lyl472324464/2026.04.13_twist_tear_water-direction_tabacco-top-2",
    "lyl472324464/2026-04-14_tear_twist_water-direction_tabacco-mid_lower",
    "lyl472324464/2026-04-14_twist_tear_water-direction_tabacco-mid_lower",
    "lyl472324464/2026-04-14_twist_tear_water-direction_tabacco-all"
]

# Hugging Face Hub dataset roots under `eii-data-system-prod/data/huggingface/hub` on 192.168.1.40
# (datasets--org--name), excluding any repo_id whose name contains "tear" (case-insensitive).
_EII_DATA_SYSTEM_HUB_NO_TEAR_REPO_IDS = [
    "lyl472324464/2025-09-15-twist-one-bottle-no-box-in-the-front",
    "lyl472324464/2025-11-06-twist-many-bottles",
    "lyl472324464/2025-11-14-twist-two-bottles",
    "lyl472324464/2025-11-18-twist-two-bottles",
    "lyl472324464/2025-11-26-twist-two-bottles",
    "lyl472324464/2025-12-10-twist-one-bottle",
    "lyl472324464/2025-12-23-twist-one-bottle",
    "lyl472324464/2026-01-20-twist-one-bottle",
    "lyl472324464/2026-01-28-twist-many-bottle",
    "lyl472324464/2026-02-03-no-cap-and-direction",
    "lyl472324464/2026-03-04-one-direction",
    "lyl472324464/2026-03-05-two-direction",
    # 2026-03-09 two repos ~28.3k frames total on Hub; repeat 4x each (~113k weighted) to target ~100k.
    "lyl472324464/2026-03-09-inference-with-and-without-cap",
    "lyl472324464/2026-03-09-no-cap-inference",
    "lyl472324464/2026-03-09-inference-with-and-without-cap",
    "lyl472324464/2026-03-09-no-cap-inference",
    "lyl472324464/2026-03-09-inference-with-and-without-cap",
    "lyl472324464/2026-03-09-no-cap-inference",
    "lyl472324464/2026-03-09-inference-with-and-without-cap",
    "lyl472324464/2026-03-09-no-cap-inference",
    "lyl472324464/2026-03-12-one-have-cap",
    "lyl472324464/2026-03-12-one-have-cap-direction",
    "lyl472324464/2026-03-12-one-havent-cap",
    "lyl472324464/2026-03-12-one-havent-cap-direction",
    "lyl472324464/2026-03-12-two-have-all-left",
    "lyl472324464/2026-03-12-two-have-cap-all-right",
    "lyl472324464/2026-03-12-two-have-cap-one-right",
    "lyl472324464/2026.03.16_twist_many",
    # 2026-04-21 (HF Hub ids use `-lerobot` suffix) — rotate-heavy; listed twice to up-weight in training.
    "lyl472324464/2026-04-21_direction-lerobot",
    "lyl472324464/2026-04-21_direction_2-lerobot",
    "lyl472324464/2026-04-21_direction_havent_cap-lerobot",
    "lyl472324464/2026-04-21_direction_havent_cap_water-lerobot",
    "lyl472324464/2026-04-21_direction-lerobot",
    "lyl472324464/2026-04-21_direction_2-lerobot",
    "lyl472324464/2026-04-21_direction_havent_cap-lerobot",
    "lyl472324464/2026-04-21_direction_havent_cap_water-lerobot",
]

_EII_RINSE_REPO_IDS = [
    "lyl472324464/2026-04-21_direction-lerobot-with-rinse",
    "lyl472324464/2026-04-21_direction_2-lerobot-with-rinse",
    "lyl472324464/2026-04-21_direction_havent_cap-lerobot-with-rinse",
    "lyl472324464/2026-04-21_direction_havent_cap_water-lerobot-with-rinse",
    "lyl472324464/2026-04-23_direction_havent_cap_water-lerobot-with-rinse",
    "lyl472324464/2026.03.18_twist-and-water_one_no_cap-with-rinse",
    "lyl472324464/2026.03.30_twist-and-water_two_have_cap-with-rinse",
]

_EII_RINSE_9REPO_REPO_IDS = [
    "lyl472324464/2026-05-07_water-lerobot-with-rinse",
    "lyl472324464/2026-05-05_water-lerobot-with-rinse",
    "lyl472324464/2026-05-05_direction-water-lerobot-with-rinse",
    "lyl472324464/2026-05-04_direction-lerobot-with-rinse",
    "lyl472324464/2026-05-04_turn_over-lerobot-with-rinse",
    "lyl472324464/2026-05-04_direction-twist-water-lerobot-with-rinse",
    "lyl472324464/2026-05-03_turn_over-lerobot-with-rinse",
    "lyl472324464/2026-05-01_water1-lerobot-with-rinse",
    "lyl472324464/2026-05-01_turn_over-lerobot-with-rinse",
]

_EII_RINSE_11REPO_REPO_IDS = [
    "lyl472324464/2026-05-01_turn_over-lerobot-with-rinse",
    "lyl472324464/2026-05-01_water1-lerobot-with-rinse",
    "lyl472324464/2026-05-03_turn_over-lerobot-with-rinse",
    "lyl472324464/2026-05-04_direction-twist-water-lerobot-with-rinse",
    "lyl472324464/2026-05-04_turn_over-lerobot-with-rinse",
    "lyl472324464/2026-05-04_direction-lerobot-with-rinse",
    "lyl472324464/2026-05-05_direction-water-lerobot-with-rinse",
    "lyl472324464/2026-05-05_water-lerobot-with-rinse",
    "lyl472324464/2026-05-07_water-lerobot-with-rinse",
    "lyl472324464/2026-05-12_insert-to-nozzle_realign-lerobot-with-rinse",
    "lyl472324464/2026-05-13-insert-to-nozzle-no-cap-with-rinse",
]

_EII_RINSE_INSERT_REALIGN_REPO_ID = "lyl472324464/2026-05-12_insert-to-nozzle_realign-lerobot-with-rinse"
_EII_RINSE_INSERT_NO_CAP_REPO_ID = "lyl472324464/2026-05-13-insert-to-nozzle-no-cap-with-rinse"

_EII_RINSE_11REPO_INSERT_X5_REPO_IDS = [
    *_EII_RINSE_11REPO_REPO_IDS,
    *([_EII_RINSE_INSERT_REALIGN_REPO_ID] * 4),
    *([_EII_RINSE_INSERT_NO_CAP_REPO_ID] * 4),
]

_EII_DATA_SYSTEM_WITHOUT_RINSE_36_REPO_IDS = [
    "lyl472324464/2025-09-15-twist-one-bottle-no-box-in-the-front-without-rinse",
    "lyl472324464/2025-11-06-twist-many-bottles-without-rinse",
    "lyl472324464/2025-11-14-twist-two-bottles",
    "lyl472324464/2025-11-18-twist-two-bottles",
    "lyl472324464/2025-11-26-twist-two-bottles",
    "lyl472324464/2025-12-10-twist-one-bottle",
    "lyl472324464/2025-12-23-twist-one-bottle",
    "lyl472324464/2026-01-20-twist-one-bottle",
    "lyl472324464/2026-01-28-twist-many-bottle",
    "lyl472324464/2026-02-03-no-cap-and-direction",
    "lyl472324464/2026-03-04-one-direction",
    "lyl472324464/2026-03-05-two-direction",
    "lyl472324464/2026-03-12-one-have-cap",
    "lyl472324464/2026-03-12-one-have-cap-direction",
    "lyl472324464/2026-03-12-one-havent-cap",
    "lyl472324464/2026-03-12-one-havent-cap-direction",
    "lyl472324464/2026-03-12-two-have-all-left",
    "lyl472324464/2026-03-12-two-have-cap-all-right",
    "lyl472324464/2026-03-12-two-have-cap-one-right",
    "lyl472324464/2026.03.16_twist_many-without-rinse",
    "lyl472324464/2026-04-21_direction-lerobot-without-rinse",
    "lyl472324464/2026-04-21_direction_2-lerobot-without-rinse",
    "lyl472324464/2026-04-21_direction_havent_cap-lerobot-without-rinse",
    "lyl472324464/2026-04-21_direction_havent_cap_water-lerobot-without-rinse",
    "lyl472324464/2026-04-23_direction_havent_cap_water-lerobot-without-rinse",
    "lyl472324464/2026-04-23_direction_have_cap_water-lerobot-without-rinse",
    "lyl472324464/2026-04-27direction_have_cap_water-lerobot-without-rinse",
    "lyl472324464/2026-04-27_direction_have_cap_water2-lerobot-without-rinse",
    "lyl472324464/2026-04-28_direction_have_cap_water-lerobot-without-rinse",
    "lyl472324464/2026-04-28_direction_have_cap_water2-lerobot-without-rinse",
    "lyl472324464/2026-05-01_turn_over-lerobot-without-rinse",
    "lyl472324464/2026-05-03_turn_over-lerobot-without-rinse",
    "lyl472324464/2026-05-04_direction-lerobot-without-rinse",
    "lyl472324464/2026-05-04_direction-twist-water-lerobot-without-rinse",
    "lyl472324464/2026-05-04_turn_over-lerobot-without-rinse",
    "lyl472324464/2026-05-05_direction-water-lerobot-without-rinse",
]

_EII_DATA_SYSTEM_WITHOUT_RINSE_41_REPO_IDS = [
    "lyl472324464/2025-09-15-twist-one-bottle-no-box-in-the-front-without-rinse",
    "lyl472324464/2025-11-06-twist-many-bottles-without-rinse",
    "lyl472324464/2025-11-14-twist-two-bottles",
    "lyl472324464/2025-11-18-twist-two-bottles",
    "lyl472324464/2025-11-26-twist-two-bottles",
    "lyl472324464/2025-12-10-twist-one-bottle",
    "lyl472324464/2025-12-23-twist-one-bottle",
    "lyl472324464/2026-01-20-twist-one-bottle",
    "lyl472324464/2026-01-28-twist-many-bottle",
    "lyl472324464/2026-02-03-no-cap-and-direction-without-rinse",
    "lyl472324464/2026-03-04-one-direction-lerobot-without-rinse",
    "lyl472324464/2026-03-05-two-direction-lerobot-without-rinse",
    "lyl472324464/2026-03-12-one-havent-cap",
    "lyl472324464/2026-03-12-one-havent-cap-direction",
    "lyl472324464/2026-04-21_direction-lerobot-without-rinse",
    "lyl472324464/2026-04-21_direction_2-lerobot-without-rinse",
    "lyl472324464/2026-04-21_direction_haven-t_cap-lerobot-without-rinse",
    "lyl472324464/2026-04-21_direction_havent_cap_water-lerobot-without-rinse",
    "lyl472324464/2026-04-23_direction_have_cap_water-lerobot-without-rinse",
    "lyl472324464/2026-04-23_direction_havent_cap_water-lerobot-without-rinse",
    "lyl472324464/2026-04-27_direction_have_cap_water2-lerobot-without-rinse",
    "lyl472324464/2026-04-27direction_have_cap_water-lerobot-without-rinse",
    "lyl472324464/2026-04-28_direction_have_cap_water-lerobot-without-rinse",
    "lyl472324464/2026-04-28_direction_have_cap_water2-lerobot-without-rinse",
    "lyl472324464/2026-05-01_turn_over-lerobot-without-rinse",
    "lyl472324464/2026-05-03_turn_over-lerobot-without-rinse",
    "lyl472324464/2026-05-04_direction-lerobot-without-rinse",
    "lyl472324464/2026-05-04_direction-twist-water-lerobot-without-rinse",
    "lyl472324464/2026-05-04_turn_over-lerobot-without-rinse",
    "lyl472324464/2026-05-05_direction-water-lerobot-without-rinse",
    "lyl472324464/2026-05-11_cap-lerobot-without-rinse",
    "lyl472324464/2026-05-11_free-spinning-lerobot-without-rinse",
    "lyl472324464/2026-05-11_twist-lerobot-without-rinse",
    "lyl472324464/2026-05-12_twist-lerobot-without-rinse",
    "lyl472324464/2026-05-12_twist2-lerobot-without-rinse",
    "lyl472324464/2026.03.12_one_have_cap-lerobot-without-rinse",
    "lyl472324464/2026.03.12_one_have_cap_direction-lerobot-without-rinse",
    "lyl472324464/2026.03.12_two_have_all_left-lerobot-without-rinse",
    "lyl472324464/2026.03.12_two_have_cap_all_right-lerobot-without-rinse",
    "lyl472324464/2026.03.12_two_have_cap_one_right-lerobot-without-rinse",
    "lyl472324464/2026.03.16_twist_many-lerobot-without-rinse",
]

_EII_DATA_SYSTEM_WITHOUT_RINSE_34_EXCLUDED_REPO_IDS = {
    "lyl472324464/2025-09-15-twist-one-bottle-no-box-in-the-front-without-rinse",
    "lyl472324464/2025-11-14-twist-two-bottles",
    "lyl472324464/2025-11-18-twist-two-bottles",
    "lyl472324464/2025-11-26-twist-two-bottles",
    "lyl472324464/2025-12-10-twist-one-bottle",
    "lyl472324464/2025-12-23-twist-one-bottle",
    "lyl472324464/2026-01-20-twist-one-bottle",
}

_EII_DATA_SYSTEM_WITHOUT_RINSE_34_REPO_IDS = [
    repo_id
    for repo_id in _EII_DATA_SYSTEM_WITHOUT_RINSE_41_REPO_IDS
    if repo_id not in _EII_DATA_SYSTEM_WITHOUT_RINSE_34_EXCLUDED_REPO_IDS
]

_EII_FREE_SPINNING_REPO_ID = "lyl472324464/2026-05-11_free-spinning-lerobot-without-rinse"

_EII_DATA_SYSTEM_WITHOUT_RINSE_34_REPO_IDS_FREE_SPIN_X6 = [
    *_EII_DATA_SYSTEM_WITHOUT_RINSE_34_REPO_IDS,
    *([_EII_FREE_SPINNING_REPO_ID] * 5),
]

_EII_DATA_SYSTEM_WITHOUT_RINSE_25_REPO_IDS = [
    "lyl472324464/2025-09-15-twist-one-bottle-no-box-in-the-front-without-rinse",
    "lyl472324464/2025-11-06-twist-many-bottles-without-rinse",
    "lyl472324464/2025-11-14-twist-two-bottles-without-rinse",
    "lyl472324464/2025-11-18-twist-two-bottles",
    "lyl472324464/2025-11-26-twist-two-bottles-without-rinse",
    "lyl472324464/2025-12-10-twist-one-bottle",
    "lyl472324464/2025-12-23-twist-one-bottle-without-rinse",
    "lyl472324464/2026-01-20-twist-one-bottle",
    "lyl472324464/2026-01-28-twist-many-bottle-without-rinse",
    "lyl472324464/2026-02-03-no-cap-and-direction-without-rinse",
    "lyl472324464/2026-03-04-one-direction-lerobot-without-rinse",
    "lyl472324464/2026-03-05-two-direction-lerobot-without-rinse",
    "lyl472324464/2026.03.12_one_have_cap-lerobot-without-rinse",
    "lyl472324464/2026.03.12_one_have_cap_direction-lerobot-without-rinse",
    "lyl472324464/2026-03-12-one-havent-cap",
    "lyl472324464/2026-03-12-one-havent-cap-direction",
    "lyl472324464/2026.03.12_two_have_all_left-lerobot-without-rinse",
    "lyl472324464/2026.03.12_two_have_cap_all_right-lerobot-without-rinse",
    "lyl472324464/2026.03.12_two_have_cap_one_right-lerobot-without-rinse",
    "lyl472324464/2026.03.16_twist_many-lerobot-without-rinse",
    "lyl472324464/2026-04-21_direction-lerobot-without-rinse",
    "lyl472324464/2026-04-21_direction_2-lerobot-without-rinse",
    "lyl472324464/2026-04-21_direction_haven-t_cap-lerobot-without-rinse",
    "lyl472324464/2026-04-21_direction_havent_cap_water-lerobot-without-rinse",
    "lyl472324464/2026-04-23_direction_havent_cap_water-lerobot-without-rinse",
]

_EII_DATA_SYSTEM_WITHOUT_RINSE_MERGED_ADJUST_PICKUP_36_REPO_IDS = [
    "lyl472324464/2025-09-15-twist-one-bottle-no-box-in-the-front-without-rinse-merged-adjust-pickup",
    "lyl472324464/2025-11-06-twist-many-bottles-without-rinse-merged-adjust-pickup",
    "lyl472324464/2025-11-14-twist-two-bottles-without-rinse-merged-adjust-pickup",
    "lyl472324464/2025-11-18-twist-two-bottles-merged-adjust-pickup",
    "lyl472324464/2025-11-26-twist-two-bottles-without-rinse-merged-adjust-pickup",
    "lyl472324464/2025-12-10-twist-one-bottle-merged-adjust-pickup",
    "lyl472324464/2025-12-23-twist-one-bottle-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-01-20-twist-one-bottle-merged-adjust-pickup",
    "lyl472324464/2026-01-28-twist-many-bottle-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-02-03-no-cap-and-direction-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-03-04-one-direction-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-03-05-two-direction-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026.03.12_one_have_cap-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026.03.12_one_have_cap_direction-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-03-12-one-havent-cap-merged-adjust-pickup",
    "lyl472324464/2026-03-12-one-havent-cap-direction-merged-adjust-pickup",
    "lyl472324464/2026.03.12_two_have_all_left-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026.03.12_two_have_cap_all_right-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026.03.12_two_have_cap_one_right-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026.03.16_twist_many-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-04-21_direction-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-04-21_direction_2-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-04-21_direction_haven-t_cap-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-04-21_direction_havent_cap_water-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-04-23_direction_havent_cap_water-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-04-28_direction_have_cap_water-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-04-28_direction_have_cap_water2-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-05-11_free-spinning-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-05-11_free-spinning-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-05-11_free-spinning-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-05-11_free-spinning-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-05-11_free-spinning-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-05-11_cap-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-05-11_twist-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-05-12_twist-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-05-12_twist2-lerobot-without-rinse-merged-adjust-pickup",
]

_EII_DATA_SYSTEM_WITHOUT_RINSE_RETURN_HOME_29_REPO_IDS = [
    "lyl472324464/2025-09-15-twist-one-bottle-no-box-in-the-front-without-rinse-merged-adjust-pickup",
    "lyl472324464/2025-12-10-twist-one-bottle-merged-adjust-pickup",
    "lyl472324464/2025-12-23-twist-one-bottle-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-01-20-twist-one-bottle-merged-adjust-pickup",
    "lyl472324464/2026-02-03-no-cap-and-direction-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-03-04-one-direction-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026.03.12_one_have_cap-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026.03.12_one_have_cap_direction-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-03-12-one-havent-cap-merged-adjust-pickup",
    "lyl472324464/2026-03-12-one-havent-cap-direction-merged-adjust-pickup",
    "lyl472324464/2026-04-21_direction-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-04-21_direction_2-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-04-21_direction_haven-t_cap-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-04-21_direction_havent_cap_water-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-04-23_direction_havent_cap_water-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-04-28_direction_have_cap_water-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-04-28_direction_have_cap_water2-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-05-11_free-spinning-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-05-11_free-spinning-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-05-11_free-spinning-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-05-11_free-spinning-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-05-11_free-spinning-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-05-11_cap-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-05-11_twist-lerobot-truncated-return-home-exp-truncated-return-home-20260520-095140",
    "lyl472324464/2026-05-12_twist-lerobot-truncated-return-home-exp-truncated-return-home-20260520-095140",
    "lyl472324464/2026-05-12_twist2-lerobot-truncated-return-home-exp-truncated-return-home-20260520-095140",
    "lyl472324464/2026-05-04_turn_over-lerobot-without-rinse",
    "lyl472324464/2026-05-03_turn_over-lerobot-without-rinse",
    "lyl472324464/2026-05-01_turn_over-lerobot-without-rinse",
]

_EII_TURN_OVER_WITHOUT_RINSE_REPO_IDS = [
    "lyl472324464/2026-05-04_turn_over-lerobot-without-rinse",
    "lyl472324464/2026-05-03_turn_over-lerobot-without-rinse",
    "lyl472324464/2026-05-01_turn_over-lerobot-without-rinse",
]

# Used by `eii_data_system_without_rinse_cam3_fullft_h200_return_home_29repo`:
# the base list above has 29 entries, then the three turn_over repos are repeated
# four extra times each, and free-spinning is repeated ten extra times.
_EII_DATA_SYSTEM_WITHOUT_RINSE_RETURN_HOME_TURN_OVER_X5_REPO_IDS = [
    *_EII_DATA_SYSTEM_WITHOUT_RINSE_RETURN_HOME_29_REPO_IDS,
    *[repo_id for repo_id in _EII_TURN_OVER_WITHOUT_RINSE_REPO_IDS for _ in range(4)],
]

_EII_FREE_SPINNING_MERGED_ADJUST_PICKUP_REPO_ID = (
    "lyl472324464/2026-05-11_free-spinning-lerobot-without-rinse-merged-adjust-pickup"
)

_EII_DATA_SYSTEM_WITHOUT_RINSE_RETURN_HOME_TURN_OVER_X5_FREE_SPIN_PLUS10_REPO_IDS = [
    *_EII_DATA_SYSTEM_WITHOUT_RINSE_RETURN_HOME_TURN_OVER_X5_REPO_IDS,
    *([_EII_FREE_SPINNING_MERGED_ADJUST_PICKUP_REPO_ID] * 10),
]

@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    assets_dir: str
    asset_id: str


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig:
    repo_id: str | None = None
    repo_ids: list[str] | None = None
    assets: AssetsConfig = tyro.MISSING
    transform_pipeline: _transforms.AlohaTransformPipeline | None = None
    use_quantile_norm: bool = True
    use_delta_joint_actions: bool = True
    adapt_to_pi: bool = True
    video_memory_num_frames: int = 1
    video_memory_stride_seconds: float = 1.0
    include_low: bool = True
    include_subtask: bool = True
    action_sequence_keys: Sequence[str] = ("action",)


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    name: tyro.conf.Suppress[str]
    project_name: str = "openpi"
    exp_name: str = tyro.MISSING
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)
    pytorch_weight_path: str | None = None
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"
    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)
    data: LeRobotAlohaDataConfig = tyro.MISSING
    checkpoint_base_dir: str = "./checkpoints"
    seed: int = 42
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    num_workers: int = 2
    num_train_steps: int = 30_000
    log_interval: int = 100
    save_interval: int = 1000
    keep_period: int | None = 5000
    overwrite: bool = False
    resume: bool = False
    wandb_enabled: bool = True
    policy_metadata: dict[str, Any] | None = None
    fsdp_devices: int = 1

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

def _local_assets(config_name: str, base_dir: str = "./assets") -> AssetsConfig:
    return AssetsConfig(assets_dir=str(pathlib.Path(base_dir) / config_name), asset_id="trossen")


def _make_twist_train_config(
    name: str,
    *,
    repo_ids: list[str],
    lora: bool,
    batch_size: int,
    num_workers: int,
    fsdp_devices: int = 1,
    include_low: bool = True,
    include_subtask: bool = True,
    gradient_accumulation_steps: int = 1,
    image_resolution: tuple[int, int] = (224, 224),
    max_token_len: int | None = None,
    video_memory_num_frames: int = 1,
    video_memory_stride_seconds: float = 1.0,
    assets: AssetsConfig,
    exp_name: str = tyro.MISSING,
    checkpoint_base_dir: str = "./checkpoints",
    wandb_enabled: bool = True,
    overwrite: bool = False,
    resume: bool = False,
    num_train_steps: int = 40_000,
) -> TrainConfig:
    if lora:
        model = pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            image_resolution=image_resolution,
            max_token_len=max_token_len,
        )
        freeze_filter = pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter()
        ema_decay = None
    else:
        model = pi0_config.Pi0Config(
            image_resolution=image_resolution,
            max_token_len=max_token_len,
        )
        freeze_filter = nnx.Nothing()
        ema_decay = 0.99

    return TrainConfig(
        name=name,
        exp_name=exp_name,
        model=model,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=2.5e-5,
            decay_steps=40_000,
            decay_lr=2.5e-6,
        ),
        log_interval=10,
        data=LeRobotAlohaDataConfig(
            adapt_to_pi=True,
            video_memory_num_frames=video_memory_num_frames,
            video_memory_stride_seconds=video_memory_stride_seconds,
            repo_ids=repo_ids,
            assets=assets,
            transform_pipeline=_transforms.AlohaTransformPipeline(
                include_low=include_low,
                include_subtask=include_subtask,
                image_resolution=model.image_resolution,
                max_token_len=model.max_token_len,
                discrete_state_input=model.discrete_state_input,
                assets_dir=assets.assets_dir,
                asset_id=assets.asset_id,
                adapt_to_pi=True,
                use_delta_joint_actions=True,
                action_dim=model.action_dim,
            ),
            include_low=include_low,
            include_subtask=include_subtask,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE_PARAMS),
        freeze_filter=freeze_filter,
        ema_decay=ema_decay,
        save_interval=1000,
        num_train_steps=num_train_steps,
        batch_size=batch_size,
        num_workers=num_workers,
        fsdp_devices=fsdp_devices,
        gradient_accumulation_steps=gradient_accumulation_steps,
        checkpoint_base_dir=checkpoint_base_dir,
        wandb_enabled=wandb_enabled,
        overwrite=overwrite,
        resume=resume,
    )


_CONFIGS = [
    _make_twist_train_config(
        "twist_off_the_bottle_cap",
        repo_ids=_TWIST_AND_STATIC_REPO_IDS,
        include_low=False,
        lora=False,
        batch_size=256,
        num_workers=16,
        assets=_local_assets("twist_off_the_bottle_cap"),
    ),
    _make_twist_train_config(
        "twist_off_the_bottle_cap_lora",
        repo_ids=_TWIST_ONLY_REPO_IDS,
        lora=True,
        batch_size=32,
        num_workers=4,
        assets=_local_assets("twist_off_the_bottle_cap_lora"),
    ),
    _make_twist_train_config(
        "twist_water_tear_promptfix_lora",
        repo_ids=_TWIST_WATER_TEAR_REPO_IDS,
        lora=True,
        batch_size=64,
        num_workers=4,
        assets=_local_assets("twist_water_tear_promptfix_lora"),
    ),
    _make_twist_train_config(
        "eii_data_system_no_tear_cam3_lora",
        repo_ids=_EII_DATA_SYSTEM_HUB_NO_TEAR_REPO_IDS,
        lora=True,
        # micro-batch 64 x 2 accum = 128 effective (scripts/train.py). H100 80GB + LoRA + 3x224 cams: 64 is used elsewhere (tear_lora); OOM则改 32x4。
        batch_size=64,
        num_workers=4,
        include_low=False,
        include_subtask=False,
        gradient_accumulation_steps=2,
        # Load norm stats from ./assets/<config.name>/trossen (compute_norm_stats), not gs:// pi05_base.
        assets=_local_assets("eii_data_system_no_tear_cam3_lora"),
    ),
    _make_twist_train_config(
        "eii_data_system_without_rinse_cam3_fullft_h200",
        repo_ids=_EII_DATA_SYSTEM_WITHOUT_RINSE_36_REPO_IDS,
        lora=False,
        batch_size=256,
        num_workers=128,
        fsdp_devices=8,
        include_low=False,
        include_subtask=False,
        gradient_accumulation_steps=1,
        exp_name="no_rinse_cam3_fullft_36repo_bs256_nw128_fsdp8_20260510",
        checkpoint_base_dir="/workspace/openpi0.5-rtc/checkpoints",
        wandb_enabled=True,
        overwrite=True,
        resume=False,
        # 2026-05-08 benchmarked on 8x H200:
        # This config for the 2026-05-10 36-repo production run:
        # fsdp=8, nw=128, bs=256, 3 cameras, full fine-tune, 40k steps,
        # video_memory_num_frames=1, video_memory_stride_seconds=1.0,
        # log_interval=10, save_interval=1000, wandb_enabled=True,
        # overwrite=True, resume=False.
        # Norm stats are loaded from:
        # /workspace/openpi0.5-rtc/assets/eii_data_system_without_rinse_cam3_fullft_h200/trossen/norm_stats.json
        # Nearby H200 benchmark:
        # - fsdp=2, nw=64, bs=256:
        #   train_step_time ~= 1.78s/step on a single-repo smoke benchmark,
        #   data_wait_time ~= 0.03s - 0.37s/step in steady state.
        # - fsdp=4, nw=32, bs=256:
        #   train_step_time ~= 1.58s/step,
        #   data_wait_time ~= 0.04s - 3.58s/step.
        # - fsdp=4, nw=64, bs=256:
        #   train_step_time ~= 1.57s - 1.84s/step,
        #   data_wait_time ~= 0.03s - 2.76s/step.
        # - fsdp=8, nw=64, bs=256:
        #   train_step_time ~= 1.43s - 1.45s/step after warmup,
        #   data_wait_time ~= 0.04s - 0.29s/step over a longer run.
        # - fsdp=8, nw=64, bs=512:
        #   train_step_time ~= 2.74s - 3.37s/step,
        #   data_wait_time can spike badly (observed up to ~= 24.95s).
        # - fsdp=8, nw=64, bs=1024:
        #   train_step_time ~= 5.29s - 5.50s/step,
        #   data_wait_time ~= 0.10s - 0.99s/step in early steady state,
        #   VRAM ~= 99.8 GiB / GPU in steady state.
        # Nearby 8x H100 80GB benchmarks:
        # - fsdp=8, nw=64, bs=256:
        #   train_step_time ~= 1.50s - 1.90s/step,
        #   data_wait_time settles to ~= 0.03s - 0.04s/step after a few steps,
        #   with early spikes up to ~= 6.54s.
        # - fsdp=8, nw=64, bs=512:
        #   train_step_time ~= 2.86s - 3.15s/step,
        #   data_wait_time is usually ~= 0.05s - 0.10s/step,
        #   with observed spikes up to ~= 19.79s and ~= 6.43s,
        #   VRAM ~= 62.4 GiB / GPU after step execution.
        assets=_local_assets("eii_data_system_without_rinse_cam3_fullft_h200"),
    ),
    _make_twist_train_config(
        "eii_data_system_without_rinse_cam3_fullft_h200_41repo",
        repo_ids=_EII_DATA_SYSTEM_WITHOUT_RINSE_41_REPO_IDS,
        lora=False,
        batch_size=256,
        num_workers=128,
        fsdp_devices=8,
        include_low=False,
        include_subtask=False,
        gradient_accumulation_steps=1,
        exp_name="no_rinse_cam3_fullft_41repo_bs256_nw128_fsdp8_20260513",
        checkpoint_base_dir="/workspace/openpi0.5-rtc/checkpoints",
        wandb_enabled=True,
        overwrite=True,
        resume=False,
        # 2026-05-13 setup for the 41-repo production run. Matches the
        # 36-repo H200 full fine-tune shape: 3 cameras, no temporal memory,
        # full fine-tune, fsdp=8, bs=256, nw=128, 40k steps.
        assets=_local_assets("eii_data_system_without_rinse_cam3_fullft_h200_41repo"),
    ),
    _make_twist_train_config(
        "eii_data_system_without_rinse_cam3_fullft_h200_34repo",
        repo_ids=_EII_DATA_SYSTEM_WITHOUT_RINSE_34_REPO_IDS_FREE_SPIN_X6,
        lora=False,
        batch_size=256,
        num_workers=64,
        fsdp_devices=4,
        include_low=False,
        include_subtask=False,
        gradient_accumulation_steps=1,
        exp_name="no_rinse_cam3_fullft_34repo_freespinx6_bs256_nw64_fsdp4_20260513",
        checkpoint_base_dir="/workspace/openpi0.5-rtc/checkpoints",
        wandb_enabled=True,
        overwrite=True,
        resume=False,
        # 2026-05-13 setup excluding the 7 early twist-only repos that
        # should not participate in this production run. This remote has
        # 4x B200, so use fsdp=4 while keeping the 34-repo norm path.
        # num_workers=128 failed during multiprocessing spawn on this host;
        # 64 is the current setting to reduce dataloader wait spikes.
        # Duplicate the free-spinning dataset 5 extra times to weight its
        # specialized task 6x in the sampler while keeping the 34-repo assets.
        assets=_local_assets("eii_data_system_without_rinse_cam3_fullft_h200_34repo"),
    ),
    _make_twist_train_config(
        "eii_data_system_without_rinse_cam3_fullft_h200_25repo",
        repo_ids=_EII_DATA_SYSTEM_WITHOUT_RINSE_25_REPO_IDS,
        lora=False,
        batch_size=256,
        num_workers=64,
        fsdp_devices=4,
        include_low=False,
        include_subtask=False,
        gradient_accumulation_steps=1,
        exp_name="no_rinse_cam3_fullft_25repo_bs256_nw64_fsdp4_20260514",
        checkpoint_base_dir="/workspace/openpi0.5-rtc/checkpoints",
        wandb_enabled=True,
        overwrite=True,
        resume=False,
        # 2026-05-14 setup for the user-specified 25-repo no-rinse run.
        # Same 3-camera full fine-tune shape as the 34-repo B200 config:
        # cam_high/cam_left_wrist/cam_right_wrist, no temporal memory,
        # bs=256, nw=64, fsdp=4, 40k steps.
        assets=_local_assets("eii_data_system_without_rinse_cam3_fullft_h200_25repo"),
    ),
    _make_twist_train_config(
        "eii_data_system_without_rinse_cam3_fullft_h200_merged_adjust_pickup_36repo",
        repo_ids=_EII_DATA_SYSTEM_WITHOUT_RINSE_MERGED_ADJUST_PICKUP_36_REPO_IDS,
        lora=False,
        batch_size=256,
        num_workers=64,
        fsdp_devices=4,
        include_low=False,
        include_subtask=False,
        gradient_accumulation_steps=1,
        exp_name="no_rinse_cam3_fullft_merged_adjust_pickup_36repo_bs256_nw64_fsdp4_20260516",
        checkpoint_base_dir="/workspace/openpi0.5-rtc/checkpoints",
        wandb_enabled=True,
        overwrite=True,
        resume=False,
        # 2026-05-16 setup for merged-adjust-pickup datasets. The
        # free-spinning repo is intentionally repeated five times exactly
        # as requested, so both sampler weight and norm stats reflect it.
        assets=_local_assets("eii_data_system_without_rinse_cam3_fullft_h200_merged_adjust_pickup_36repo"),
    ),
    _make_twist_train_config(
        "eii_data_system_without_rinse_cam3_fullft_h200_return_home_29repo",
        repo_ids=_EII_DATA_SYSTEM_WITHOUT_RINSE_RETURN_HOME_TURN_OVER_X5_FREE_SPIN_PLUS10_REPO_IDS,
        lora=False,
        batch_size=256,
        num_workers=64,
        fsdp_devices=4,
        include_low=False,
        include_subtask=False,
        gradient_accumulation_steps=1,
        exp_name="no_rinse_cam3_fullft_return_home_29repo_bs256_nw64_fsdp4_20260520",
        checkpoint_base_dir="/workspace/openpi0.5-rtc/checkpoints",
        wandb_enabled=True,
        overwrite=False,
        resume=True,
        # 2026-05-22 resume setup: keep the same checkpoint directory and
        # continue from the latest saved step. The three turn-over no-rinse
        # repos remain weighted to five total copies each, and free-spinning
        # merged-adjust-pickup has ten additional copies.
        num_train_steps=60_000,
        assets=_local_assets("eii_data_system_without_rinse_cam3_fullft_h200_return_home_29repo"),
    ),
    _make_twist_train_config(
        "eii_data_system_without_rinse_cam3_fullft_a100",
        repo_ids=_EII_DATA_SYSTEM_WITHOUT_RINSE_36_REPO_IDS,
        lora=False,
        batch_size=256,
        num_workers=16,
        fsdp_devices=4,
        include_low=False,
        include_subtask=False,
        gradient_accumulation_steps=1,
        # 2026-05-07 benchmarked on 8x A100 80GB:
        # This config: fsdp=4, nw=16, bs=256
        # train_step_time ~= 14.6s/step on a single-repo smoke benchmark.
        # data_wait_time ~= 0.04s - 1.90s/step in steady state.
        # Other A100 full-ft cam3 single-repo benchmarks:
        # - fsdp=4, nw=16, bs=128:
        #   train_step_time ~= 12.4s/step,
        #   data_wait_time ~= 0.02s - 0.49s/step.
        # - fsdp=8, nw=16, bs=128:
        #   train_step_time ~= 20.2s/step,
        #   data_wait_time ~= 0.04s - 0.24s/step.
        # - fsdp=2, nw=16, bs=128:
        #   train_step_time ~= 21.5s/step,
        #   data_wait_time ~= 0.02s - 1.10s/step.
        # - fsdp=1, nw=16, bs=128:
        #   OOM on step 0.
        assets=_local_assets("eii_data_system_without_rinse_cam3_fullft_a100"),
    ),
    _make_twist_train_config(
        "eii_rinse_cam4_lora",
        repo_ids=_EII_RINSE_REPO_IDS,
        lora=True,
        batch_size=32,
        num_workers=4,
        include_low=True,
        include_subtask=True,
        gradient_accumulation_steps=2,
        assets=_local_assets("eii_rinse_cam4_lora"),
    ),
    _make_twist_train_config(
        "eii_rinse_9repo_cam4_lora",
        repo_ids=_EII_RINSE_9REPO_REPO_IDS,
        lora=True,
        batch_size=32,
        num_workers=16,
        include_low=True,
        include_subtask=True,
        gradient_accumulation_steps=2,
        assets=_local_assets("eii_rinse_9repo_cam4_lora"),
    ),
    _make_twist_train_config(
        "eii_rinse_11repo_cam4_fullft",
        repo_ids=_EII_RINSE_11REPO_INSERT_X5_REPO_IDS,
        lora=False,
        batch_size=256,
        num_workers=128,
        include_low=True,
        include_subtask=True,
        gradient_accumulation_steps=1,
        assets=_local_assets("eii_rinse_11repo_cam4_fullft"),
    ),
    _make_twist_train_config(
        "eii_rinse_9repo_cam4_lora_6000",
        repo_ids=_EII_RINSE_9REPO_REPO_IDS,
        lora=True,
        batch_size=32,
        num_workers=16,
        include_low=True,
        include_subtask=True,
        gradient_accumulation_steps=2,
        exp_name="eii_rinse_9repo_cam4_lora_wandb_bs32_acc2_nw16_nomem_20260510_093635",
        assets=_local_assets("eii_rinse_9repo_cam4_lora_6000"),
    ),
    _make_twist_train_config(
        "eii_rinse_9repo_cam4_lora_15000",
        repo_ids=_EII_RINSE_9REPO_REPO_IDS,
        lora=True,
        batch_size=32,
        num_workers=16,
        include_low=True,
        include_subtask=True,
        gradient_accumulation_steps=2,
        exp_name="eii_rinse_9repo_cam4_lora_wandb_bs32_acc2_nw16_nomem_20260510_093635",
        assets=_local_assets("eii_rinse_9repo_cam4_lora_15000"),
    ),
    _make_twist_train_config(
        "eii_rinse_cam4_fullft",
        repo_ids=_EII_RINSE_REPO_IDS,
        lora=False,
        batch_size=128,
        num_workers=16,
        include_low=True,
        include_subtask=True,
        gradient_accumulation_steps=1,
        assets=_local_assets("eii_rinse_cam4_fullft"),
    ),
    TrainConfig(
        name="debug",
        data=LeRobotAlohaDataConfig(
            repo_id=_TWIST_ONLY_REPO_IDS[0],
            assets=_local_assets("debug"),
            transform_pipeline=_transforms.AlohaTransformPipeline(
                include_low=False,
                include_subtask=False,
                image_resolution=(224, 224),
                max_token_len=200,
                discrete_state_input=True,
                assets_dir=_local_assets("debug").assets_dir,
                asset_id=_local_assets("debug").asset_id,
                adapt_to_pi=True,
                use_delta_joint_actions=True,
                action_dim=32,
            ),
            include_low=False,
            include_subtask=False,
        ),
        batch_size=2,
        model=pi0_config.Pi0Config(
            paligemma_variant="dummy",
            action_expert_variant="dummy",
        ),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
]

_CONFIGS[0] = dataclasses.replace(_CONFIGS[0], policy_metadata=_TROSSEN_RESET_POSE)

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")

_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")
    return _CONFIGS_DICT[config_name]
