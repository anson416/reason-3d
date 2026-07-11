import argparse
import json
import os
import shutil
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import ASSETS, BLENDER_FILE, DESCRIPTIONS, OBJ_DATA, ROTATION_DATA

# Mesh-file extensions the renderer can import, in priority order (must match
# render_layout.py's candidate search so exported_meshes() copies the EXACT
# file the renderer would load for a given asset).
_MESH_EXTENSIONS = (".glb", ".gltf", ".fbx", ".obj", ".blend")


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


def _resolve_asset_file(uid, asset_base_path):
    """Find the source mesh file for an asset ``uid`` on disk.

    Mirrors render_layout.py's candidate search exactly (flat ``<uid>.glb`` and
    nested ``<uid>/<uid>.glb`` layouts, same extension priority), so the file
    copied into ``meshes/`` is the same one the renderer loads. Returns the path
    or ``None`` if no candidate exists (e.g. the asset folder is absent on this
    machine, or the guid resolved to a name with no mesh).
    """
    for ext in _MESH_EXTENSIONS:
        for candidate in (
            os.path.join(asset_base_path, f"{uid}{ext}"),
            os.path.join(asset_base_path, uid, f"{uid}{ext}"),
        ):
            if os.path.isfile(candidate):
                return candidate
    return None


def export_meshes(scene_dir):
    """Copy each placed object's source mesh into ``<scene_dir>/meshes/``.

    The renderer imports these same source GLBs/FBXs and (optionally) rescales
    them to fit the placed size; the GEOMETRY shown to a VLM is therefore the
    source asset file, so copying it makes each scene folder self-contained
    (mesh + layout + optional renderings) -- which is what the audit needs.
    No Blender required: pure file copy keyed by guid -> asset name -> file.

    Missing files are skipped silently (a warning is printed) rather than
    crashing the run, so a scene can still be generated/rendered even if some
    asset meshes are absent on this machine. Object-instance names (e.g.
    ``Sofa2``) are used as the destination filenames so duplicate placements of
    the same asset do not collide.
    """
    json_path = os.path.join(scene_dir, "placed_objects.json")
    json_data_path = os.path.join(scene_dir, "placed_objects_data.json")
    if not (os.path.isfile(json_path) and os.path.isfile(json_data_path)):
        return  # nothing placed (e.g. empty biggest-only scene) -> nothing to copy

    with open(json_path, "r") as f:
        objects = json.load(f)
    with open(json_data_path, "r") as f:
        object_data = json.load(f)

    name_by_guid = _build_name_index()
    try:
        with open(DESCRIPTIONS, "r") as file:
            id_data = json.load(file)
    except FileNotFoundError:
        id_data = {}

    meshes_dir = os.path.join(scene_dir, "meshes")
    os.makedirs(meshes_dir, exist_ok=True)

    for data, obj in zip(object_data, objects):
        guid = data["guid"]
        # guid -> asset name, same resolution as convert(): DESCRIPTIONS first,
        # OBJ_DATA prefabName fallback for substituted/alt assets.
        name = next(
            (k for k, v in id_data.items() if v.get("guid") == guid), None
        )
        if name is None:
            name = name_by_guid.get(guid)
        if name is None:
            print(f"[meshes] guid {guid!r} resolves to no asset name; skipping")
            continue
        src = _resolve_asset_file(name, ASSETS)
        if src is None:
            print(f"[meshes] no mesh file found for asset {name!r}; skipping")
            continue
        ext = os.path.splitext(src)[1]
        obj_name = obj.get("name") or name
        # Asset names use a ``category/name`` convention (e.g.
        # ``Office/Gaming Chair``), so flatten any path separators in the
        # destination filename -- otherwise os.path.join treats the category as
        # a (non-existent) subdirectory of meshes/ and copy2 fails.
        safe_name = obj_name.replace(os.sep, "_").replace("/", "_").replace("\\", "_")
        dst = os.path.join(meshes_dir, f"{safe_name}{ext}")
        shutil.copy2(src, dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene-dir", default=None, help="Timestamped scene directory"
    )
    args = parser.parse_args()
    convert(args.scene_dir)
