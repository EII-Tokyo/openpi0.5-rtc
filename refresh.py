from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import os
import time
import shutil

repo_ids=[
                "lyl472324464/twist",
                "lyl472324464/twist",
                "lyl472324464/twist",
                "lyl472324464/twist",
                "lyl472324464/twist",
                "lyl472324464/twist",
                "lyl472324464/twist",
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
                "lyl472324464/hook_cable_8pin",
                "lyl472324464/handover_clear_zip_bag_upright",
                "lyl472324464/hook_cable_narrow_8pin",
                "lyl472324464/handover_metallic_zip_bag_upright",
                "lyl472324464/hook_rubber_shaft",
                "lyl472324464/handover_towel",
                "lyl472324464/hit_mark_with_hammer",
                "lyl472324464/fold_big_towel",
                "lyl472324464/fold_towel_assist",
                "lyl472324464/fold_towel_in_random_places",
                "lyl472324464/fold_green_towel",
                "lyl472324464/fold_yellow_towel",
                "lyl472324464/fold_light_blue_towel",
                "lyl472324464/fold_bath_towel",
                "lyl472324464/fold_orange_towel",
                "lyl472324464/fit_small_gear_shaft",
                "lyl472324464/fit_large_gear_shaft",
                "lyl472324464/find_insert_small_gear_shaft",
                "lyl472324464/find_insert_large_gear_shaft",        
                "lyl472324464/find_hole_and_insert_into_gear",
                "lyl472324464/close_toolbox",
                "lyl472324464/close_cardboard_box",
                "lyl472324464/brush_screws_into_dustpan_left_brush_human_hold",
                "lyl472324464/brush_screws_into_dustpan_human_brush_left_hold",
            ]

datasets = [
                "remove-label-20251014"
]

for dataset in datasets:
    success = False
    while not success:
        try:
            print(f"Processing {dataset}...")
            dataset = LeRobotDataset(f"lyl472324464/{dataset}")
            dataset.push_to_hub()
            success = True
            # shutil.rmtree(f"/home/eii/.cache/huggingface/datasets/")
            # shutil.rmtree(f"/home/eii/.cache/huggingface/lerobot/lyl472324464/{dataset}")           
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(10)