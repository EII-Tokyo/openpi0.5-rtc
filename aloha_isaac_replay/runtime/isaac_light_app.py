from __future__ import annotations


LIGHTWEIGHT_SIMULATION_APP_CONFIG: dict[str, object] = {
    "headless": True,
    "create_new_stage": False,
    "disable_viewport_updates": True,
    "width": 320,
    "height": 240,
    "window_width": 320,
    "window_height": 240,
    "anti_aliasing": 0,
    "multi_gpu": False,
    "sync_loads": False,
    "samples_per_pixel_per_frame": 1,
    "denoiser": False,
    "max_bounces": 1,
    "max_specular_transmission_bounces": 1,
    "max_volume_bounces": 0,
    "fast_shutdown": True,
    "limit_cpu_threads": 8,
}
