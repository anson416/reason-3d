"""Offline unit tests for the render-filename + camera-convention helpers.

No bpy, no Blender, no API. Verifies the native->common pitch remap, the
transparent-master and background-composite filename scheme, and that the
OFAT baseline master maps onto the common 0==top-down key. Run with:

    python -m pytest test_render_filenames.py
or
    python test_render_filenames.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config as c  # noqa: E402


def test_native_to_common_pitch():
    assert c.native_to_common_pitch(90) == 0   # native top-down -> common 0
    assert c.native_to_common_pitch(0) == 90   # native eye-level -> common 90
    assert c.native_to_common_pitch(45) == 45  # mid unchanged
    for p in c.PITCHS:
        assert c.native_to_common_pitch(c.native_to_common_pitch(p)) == p


def test_master_filename():
    name = c.render_master_filename(512, 50, 0, 0, "city")
    assert name == "render_res-512_focal-50_pitch-0_yaw-0_env-city.png", name


def test_composite_filename():
    name = c.render_composite_filename(512, 50, 0, 0, "city", (255, 255, 255))
    assert (
        name
        == "render_res-512_focal-50_pitch-0_yaw-0_env-city_bg-255-255-255.png"
    ), name


def test_baseline_common_key_is_top_down():
    # baseline native pitch 90 (top-down) must remap to common 0 in the key
    assert c.baseline_common_key() == (512, 50, 0, 0, "city"), c.baseline_common_key()


def test_baseline_master_in_ofat_uses_common_zero():
    # The OFAT set's baseline master must have the common 0 pitch in its
    # filename (native 90 top-down -> common 0).
    b = c.baseline_common_key()
    found = False
    for cfg in c.ofat_camera_configs():
        common = c.native_to_common_pitch(cfg["pitch"])
        if (
            cfg["res"], cfg["focal"], common, cfg["yaw"], cfg["hdri"]
        ) == b:
            found = True
            break
    assert found, "baseline (common top-down) master missing from OFAT set"


def test_yaw_sweep_uses_oblique_pitch():
    # Yaw sweep runs at the oblique pitch; in the filename that pitch must be
    # the COMMON value (native 45 -> common 45) -- not native 90.
    yaw_cfgs = [cfg for cfg in c.ofat_camera_configs() if cfg["yaw"] != 0]
    assert yaw_cfgs, "no yaw-sweep configs"
    for cfg in yaw_cfgs:
        assert cfg["pitch"] == c.YAW_SWEEP_PITCH == 45, cfg


def test_backgrounds_have_ten_colors_including_118():
    bgs = c.ofat_backgrounds()
    assert (118, 118, 118) in bgs
    assert (255, 255, 255) in bgs  # white baseline
    assert len(bgs) == 10, len(bgs)


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
