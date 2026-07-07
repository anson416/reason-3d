import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BLENDER_FILE, DESCRIPTIONS, OBJ_DATA, ROTATION_DATA


def swap(vec):
    return [vec[0], vec[2], vec[1]]


def _build_name_index():
    """guid -> asset name, falling back to OBJ_DATA prefabName when a guid is
    absent from DESCRIPTIONS (e.g. a substituted/alt asset whose description
    was never generated). Without this fallback the list-comprehension below
    raises an unhelpful IndexError on the first missing guid."""
    name_by_guid = {}
    try:
        with open(OBJ_DATA, "r") as f:
            for prefab in json.load(f).get("prefabs", []):
                name_by_guid[prefab["guid"]] = prefab["prefabName"]
    except (FileNotFoundError, KeyError):
        pass
    return name_by_guid


def convert(scene_dir=None):
    if scene_dir is not None:
        json_path = os.path.join(scene_dir, "placed_objects.json")
        json_data_path = os.path.join(scene_dir, "placed_objects_data.json")
        output_path = os.path.join(scene_dir, "raw_blender.json")
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(
            script_dir, "../build_scene/placed_objects.json"
        )
        json_data_path = os.path.join(
            script_dir, "../build_scene/placed_objects_data.json"
        )
        output_path = BLENDER_FILE

    with open(json_data_path, "r") as f:
        object_data = json.load(f)
    with open(json_path, "r") as f:
        objects = json.load(f)
    with open(DESCRIPTIONS, "r") as file:
        id_data = json.load(file)
    with open(ROTATION_DATA, "r") as file:
        rotation_data = json.load(file)

    output = []
    fallback_name_by_guid = _build_name_index()
    for data, obj in zip(object_data, objects):
        guid = data["guid"]
        name = next(
            (k for k, v in id_data.items() if v["guid"] == guid), None
        )
        if name is None:
            # guid not described in DESCRIPTIONS; fall back to the asset's
            # prefabName (filename stem) from OBJ_DATA so rendering still works.
            name = fallback_name_by_guid.get(guid)
        if name is None:
            raise RuntimeError(
                f"guid {guid!r} not found in DESCRIPTIONS or OBJ_DATA; cannot "
                f"resolve asset file for object {obj.get('name')!r}."
            )
        output.append(
            {
                "position": swap(obj["center"]),
                "pre_rotation": next(
                    (
                        a["rotation"]
                        for a in rotation_data
                        if a["guid"] == data["guid"]
                    ),
                    [0, 0, 0],
                ),
                "rotation": swap(
                    [
                        -obj["rotation"][0],
                        -obj["rotation"][1],
                        -obj["rotation"][2],
                    ]
                ),
                "uid": name,
                # VLMUNR: intended object size (meters) so the renderer can
                # rescale native-unit GLBs to their real-world dimensions.
                "size": obj.get("size_after_rotation") or obj.get("size"),
            }
        )
    with open(output_path, "w") as outfile:
        json.dump(output, outfile, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene-dir", default=None, help="Timestamped scene directory"
    )
    args = parser.parse_args()
    convert(args.scene_dir)
