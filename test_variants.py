"""Offline unit tests for the named content-perturbation variants.

These run WITHOUT any LLM, embedding API, or Blender: they build a small
synthetic scene + synthetic embeddings/object_data and verify each variant
transform produces the correct object set / sizes / guids. Run with:

    python -m pytest test_variants.py
or
    python test_variants.py
"""

import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "build_scene"))
sys.path.insert(0, HERE)

import config  # noqa: E402
import generate_variants as gv  # noqa: E402
from rendering import convert_for_blender as cf  # noqa: E402


# --- synthetic fixtures -----------------------------------------------------

PLACED_OBJECTS = [
    {"name": "Chair", "center": [1.0, 0.5, 2.0], "rotation": [0, 90, 0],
     "size": [0.5, 1.0, 0.5], "size_after_rotation": [1.0, 1.0, 0.5]},
    {"name": "Table", "center": [0.0, 0.5, 0.0], "rotation": [0, 0, 0],
     "size": [2.0, 1.0, 1.0], "size_after_rotation": [2.0, 1.0, 1.0]},
    {"name": "Cup", "center": [0.2, 1.0, 0.1], "rotation": [0, 0, 0],
     "size": [0.1, 0.1, 0.1], "size_after_rotation": [0.1, 0.1, 0.1]},
    {"name": "Lamp", "center": [3.0, 1.5, 3.0], "rotation": [0, 45, 0],
     "size": [0.3, 2.0, 0.3], "size_after_rotation": [0.42, 2.0, 0.42]},
]

PLACED_OBJECTS_DATA = [
    {"guid": 1, "boundsCenter": [0, 0, 0], "size": [0.5, 1.0, 0.5]},
    {"guid": 2, "boundsCenter": [0, 0, 0], "size": [2.0, 1.0, 1.0]},
    {"guid": 3, "boundsCenter": [0, 0, 0], "size": [0.1, 0.1, 0.1]},
    {"guid": 4, "boundsCenter": [0, 0, 0], "size": [0.3, 2.0, 0.3]},
]

# Non-degenerate embeddings: OvenAsset (guid 99) is anti-correlated to all
# furniture, so it is the globally least-similar prefab for every object.
EMBEDDINGS = {
    "ChairAsset": {"guid": 1, "embedding_phys": [0.9, 0.1, 0.0],
                   "embedding_func": [0.9, 0.1, 0.0], "embedding_cont": [0.9, 0.1, 0.0]},
    "TableAsset": {"guid": 2, "embedding_phys": [0.1, 0.9, 0.0],
                   "embedding_func": [0.1, 0.9, 0.0], "embedding_cont": [0.1, 0.9, 0.0]},
    "CupAsset": {"guid": 3, "embedding_phys": [0.0, 0.0, 0.95],
                 "embedding_func": [0.0, 0.0, 0.95], "embedding_cont": [0.0, 0.0, 0.95]},
    "LampAsset": {"guid": 4, "embedding_phys": [0.6, 0.6, 0.0],
                  "embedding_func": [0.6, 0.6, 0.0], "embedding_cont": [0.6, 0.6, 0.0]},
    "OvenAsset": {"guid": 99, "embedding_phys": [-0.95, -0.95, -0.95],
                  "embedding_func": [-0.95, -0.95, -0.95],
                  "embedding_cont": [-0.95, -0.95, -0.95]},
}

OBJECT_DATA = {"prefabs": [
    {"guid": 1, "prefabName": "ChairAsset", "boundsCenter": {"x": 0, "y": 0, "z": 0},
     "boundsSize": {"x": 0.5, "y": 1.0, "z": 0.5}},
    {"guid": 2, "prefabName": "TableAsset", "boundsCenter": {"x": 0, "y": 0, "z": 0},
     "boundsSize": {"x": 2.0, "y": 1.0, "z": 1.0}},
    {"guid": 3, "prefabName": "CupAsset", "boundsCenter": {"x": 0, "y": 0, "z": 0},
     "boundsSize": {"x": 0.1, "y": 0.1, "z": 0.1}},
    {"guid": 4, "prefabName": "LampAsset", "boundsCenter": {"x": 0, "y": 0, "z": 0},
     "boundsSize": {"x": 0.3, "y": 2.0, "z": 0.3}},
    {"guid": 99, "prefabName": "OvenAsset", "boundsCenter": {"x": 0, "y": 0, "z": 0},
     "boundsSize": {"x": 0.8, "y": 0.9, "z": 0.6}},
]}


def _setup(tmp):
    """Point config + modules at synthetic data and stub out Blender conversion."""
    emb = os.path.join(tmp, "embeddings.json")
    obj = os.path.join(tmp, "object_data.json")
    with open(emb, "w") as f:
        json.dump(EMBEDDINGS, f)
    with open(obj, "w") as f:
        json.dump(OBJECT_DATA, f)
    gv.EMBEDDINGS = emb
    gv.OBJ_DATA = obj
    cf.OBJ_DATA = obj
    cf.DESCRIPTIONS = os.path.join(tmp, "descriptions.json")
    cf.ROTATION_DATA = os.path.join(tmp, "rotation_data.json")
    with open(cf.DESCRIPTIONS, "w") as f:
        json.dump({"ChairAsset": {"guid": 1}, "TableAsset": {"guid": 2}}, f)
    with open(cf.ROTATION_DATA, "w") as f:
        json.dump([], f)
    cf.convert = lambda d: open(os.path.join(d, "raw_blender.json"), "w").write("[]")
    gv.convert = cf.convert


def test_half_variant_keeps_half():
    with tempfile.TemporaryDirectory() as tmp:
        _setup(tmp)
        scene = os.path.join(tmp, "scene")
        os.makedirs(scene)
        gv.generate_half_variant(scene, PLACED_OBJECTS, PLACED_OBJECTS_DATA, "p", 42)
        out = json.load(open(os.path.join(scene, "variant_01_half", "placed_objects.json")))
        assert len(out) == 2, out  # round(4 * 0.5) == 2


def test_biggest_only_keeps_table():
    with tempfile.TemporaryDirectory() as tmp:
        _setup(tmp)
        scene = os.path.join(tmp, "scene")
        os.makedirs(scene)
        gv.generate_biggest_only_variant(scene, PLACED_OBJECTS, PLACED_OBJECTS_DATA, "p")
        out = json.load(open(os.path.join(scene, "variant_02_biggest-only", "placed_objects.json")))
        assert len(out) == 1 and out[0]["name"] == "Table", out  # vol 2.0 is largest


def test_scrambled_preserves_count_and_moves():
    with tempfile.TemporaryDirectory() as tmp:
        _setup(tmp)
        scene = os.path.join(tmp, "scene")
        os.makedirs(scene)
        orig_centers = {o["name"]: tuple(o["center"]) for o in PLACED_OBJECTS}
        gv.generate_scrambled_variant(scene, PLACED_OBJECTS, PLACED_OBJECTS_DATA, "p", 42)
        out = json.load(open(os.path.join(scene, "variant_03_scrambled", "placed_objects.json")))
        assert len(out) == 4
        # every object moved (center changed) but is still in-scene
        for o in out:
            assert tuple(o["center"]) != orig_centers[o["name"]]
            assert o["size_after_rotation"] is not None


def test_worst_object_swaps_all_to_least_similar():
    with tempfile.TemporaryDirectory() as tmp:
        _setup(tmp)
        scene = os.path.join(tmp, "scene")
        os.makedirs(scene)
        gv.generate_worst_object_variant(scene, PLACED_OBJECTS, PLACED_OBJECTS_DATA, "p")
        data = json.load(open(os.path.join(scene, "variant_04_worst-object", "placed_objects_data.json")))
        assert all(d["guid"] == 99 for d in data), data  # all swapped to OvenAsset


def test_worst_match_for_guid():
    with tempfile.TemporaryDirectory() as tmp:
        _setup(tmp)
        for guid in [1, 2, 3, 4]:
            w = gv._worst_match_for_guid(guid, EMBEDDINGS)
            assert w is not None
            assert w[1] == 99, (guid, w)  # OvenAsset is globally least-similar


def test_named_variants_produce_all_four():
    with tempfile.TemporaryDirectory() as tmp:
        _setup(tmp)
        scene = os.path.join(tmp, "scene")
        os.makedirs(scene)
        gv.generate_named_variants(scene, PLACED_OBJECTS, PLACED_OBJECTS_DATA, "p", 42)
        for v in gv.NAMED_VARIANT_DIRS:
            d = os.path.join(scene, v)
            assert os.path.isdir(d), v
            for fn in ("placed_objects.json", "placed_objects_data.json",
                       "prompt.txt", "raw_blender.json"):
                assert os.path.isfile(os.path.join(d, fn)), (v, fn)
            # every variant folder also gets a meshes/ dir (export_meshes),
            # even when no source mesh files are present on this machine.
            assert os.path.isdir(os.path.join(d, "meshes")), (v, "meshes")


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
