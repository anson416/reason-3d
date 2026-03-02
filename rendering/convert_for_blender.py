import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BLENDER_FILE, DESCRIPTIONS, ROTATION_DATA


def swap(vec):
    return [vec[0], vec[2], vec[1]]


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
    for data, obj in zip(object_data, objects):
        name = [k for k, v in id_data.items() if v["guid"] == data["guid"]][0]
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
