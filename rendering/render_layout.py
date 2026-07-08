import argparse
import json
import math
import os
import sys

import bpy
from mathutils import Vector

sys.path.append(os.path.abspath("."))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # rendering/ (for `import bpa`)
from config import (  # noqa: E402
    ASSETS,
    BASELINE_FOCAL,
    BASELINE_HDRI,
    BASELINE_PITCH,
    BASELINE_RES,
    BASELINE_YAW,
    HDRI_DIR,
    native_to_common_pitch,
    ofat_camera_configs,
    render_master_filename,
)
import bpa  # noqa: E402  (vendored from vlmunr -- bpy required)

# === PARSE ARGS ===
# Supports both `blender --background --python this.py -- --scene-dir X --mode M`
# (args after the `--` separator) and direct `python this.py --scene-dir X --mode M`
# (bpy as a module, no separator).
if "--" in sys.argv:
    argv = sys.argv[sys.argv.index("--") + 1 :]
else:
    argv = sys.argv[1:]
parser = argparse.ArgumentParser()
parser.add_argument(
    "--scene-dir", required=True, help="Path to timestamped scene dir"
)
parser.add_argument(
    "--mode",
    choices=["baseline", "ofat"],
    default="baseline",
    help="baseline = single baseline render (512/50/top-down/0/city); "
         "ofat = full one-factor-at-a-time sweep.",
)
args, _ = parser.parse_known_args(argv)

scene_dir = args.scene_dir
render_mode = args.mode


# === CONFIG ===
unity_layout_file = os.path.join(scene_dir, "raw_blender.json")
asset_base_path = ASSETS
material_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "materials"
)
floor_filepath = os.path.join(
    material_dir, "wood_floor_worn_1k.blend/wood_floor_worn_1k.blend"
)
wall_blend_filepath = os.path.join(
    material_dir, "beige_wall_001_1k.blend/beige_wall_001_1k.blend"
)
floor_path = floor_filepath
wall_path = wall_blend_filepath
material_name = os.path.basename(floor_path)[:-9]
wall_material_name = os.path.basename(wall_path)[:-9]

renderings_dir = os.path.join(scene_dir, "renderings")
os.makedirs(renderings_dir, exist_ok=True)


# === CLEAR SCENE (bpa: purges objects + all datablocks) ===
# NOTE: bpa.clear() also removes materials, so the .blend materials MUST be
# loaded AFTER clear() (the old code cleared only objects and could load first).
bpa.clear()

with bpy.data.libraries.load(floor_path, link=False) as (data_from, data_to):
    if material_name in data_from.materials:
        data_to.materials.append(material_name)
    else:
        print(f"Error: Material '{material_name}' not found in '{floor_path}'")
with bpy.data.libraries.load(wall_path, link=False) as (data_from, data_to):
    if wall_material_name in data_from.materials:
        data_to.materials.append(wall_material_name)
    else:
        print(
            f"Error: Material '{wall_material_name}' not found in '{wall_blend_filepath}'"
        )


# === IMPORT LAYOUT ===
with open(unity_layout_file, "r") as f:
    layout = json.load(f)

for info in layout:
    uid = info["uid"]
    position = info["position"]
    rotation_deg = info["rotation"]
    rotation_rad = [math.radians(r) for r in rotation_deg]
    pre_rotation_deg = info["pre_rotation"][1]
    pre_rot_rad = math.radians(pre_rotation_deg)

    # Supported extensions in priority order
    extensions = [".glb", ".fbx", ".obj", ".blend"]
    file_path = None

    # Look for the first existing file with any supported extension.
    # Support both flat layout (<base>/<uid>.glb) and nested objathor layout
    # (<base>/<uid>/<uid>.glb).
    for ext in extensions:
        for candidate in (
            os.path.join(asset_base_path, f"{uid}{ext}"),
            os.path.join(asset_base_path, uid, f"{uid}{ext}"),
        ):
            if os.path.isfile(candidate):
                file_path = candidate
                break
        if file_path:
            break

    if not file_path:
        print(f"No supported file found for UID: {uid}")
        continue
    else:
        print(f"Importing: {file_path}")

        # Track objects before import
        objects_before_import = set(bpy.context.scene.objects)

        # Determine which importer to use
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".glb" or ext == ".gltf":
            bpy.ops.import_scene.gltf(filepath=file_path)
        elif ext == ".fbx":
            bpy.ops.import_scene.fbx(filepath=file_path)
        elif ext == ".obj":
            bpy.ops.import_scene.obj(filepath=file_path)
        elif ext == ".blend":
            with bpy.data.libraries.load(file_path, link=False) as (
                data_from,
                data_to,
            ):
                data_to.objects = [name for name in data_from.objects]
            for obj in data_to.objects:
                if obj:
                    bpy.context.collection.objects.link(obj)

    objects_after_import = set(bpy.context.scene.objects)
    new_objects = objects_after_import - objects_before_import
    # 1. Filter out non-mesh objects and identify the target objects for join
    mesh_objects_to_join = [obj for obj in new_objects if obj.type == "MESH"]

    # 2. Deselect everything
    bpy.ops.object.select_all(action="DESELECT")

    # 3. Choose the active object for the join operation
    active_join_obj = None
    for obj in mesh_objects_to_join:
        if obj.parent is None:  # Prioritize unparented objects
            active_join_obj = obj
            break
    if active_join_obj is None:  # If all are parented, just pick the first one
        active_join_obj = mesh_objects_to_join[0]

    # Clear parent of the chosen active object if it has one (keeping its world transform)
    if active_join_obj.parent:
        print(
            f"Clearing parent of active object '{active_join_obj.name}' before join."
        )
        bpy.context.view_layer.objects.active = active_join_obj
        active_join_obj.select_set(True)
        bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
        active_join_obj.select_set(False)

    # Now select all objects to be joined
    bpy.context.view_layer.objects.active = active_join_obj
    active_join_obj.select_set(True)

    for obj in mesh_objects_to_join:
        if obj != active_join_obj:
            if obj.parent:
                print(f"Clearing parent of object '{obj.name}' before join.")
                current_active = bpy.context.view_layer.objects.active
                current_selected = [o for o in bpy.context.selected_objects]

                bpy.ops.object.select_all(action="DESELECT")
                bpy.context.view_layer.objects.active = obj
                obj.select_set(True)
                bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")

                bpy.ops.object.select_all(action="DESELECT")
                for sel_obj in current_selected:
                    sel_obj.select_set(True)
                bpy.context.view_layer.objects.active = current_active

            obj.select_set(True)

    try:
        if bpy.context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        bpy.ops.object.join()
        print("Objects joined successfully.")
        loaded = bpy.context.view_layer.objects.active
    except Exception as e:
        print(f"Error joining objects: {e}. Ignoring the join operation.")
        loaded = None

    # VLMUNR_PATCH fit-to-size
    # Uniformly rescale the joined object to its intended size (meters).
    try:
        import bpy as _b
        _b.context.view_layer.objects.active = loaded
        _b.context.view_layer.update()
        _dims = loaded.dimensions
        _cur_max = max(_dims.x, _dims.y, _dims.z)
        _tgt = info.get("size")
        if _tgt:
            _tgt_max = max(abs(float(t)) for t in _tgt)
        else:
            _tgt_max = min(_cur_max, 3.5)
        if _cur_max > 1e-6 and _tgt_max > 1e-6:
            _f = _tgt_max / _cur_max
            loaded.scale = (loaded.scale.x*_f, loaded.scale.y*_f, loaded.scale.z*_f)
            _b.ops.object.transform_apply(location=False, rotation=False, scale=True)
        # Hard clamp: never exceed 3.5 m on any axis.
        _b.context.view_layer.update()
        _dims = loaded.dimensions
        _cur_max = max(_dims.x, _dims.y, _dims.z)
        if _cur_max > 3.5:
            _f = 3.5 / _cur_max
            loaded.scale = (loaded.scale.x*_f, loaded.scale.y*_f, loaded.scale.z*_f)
            _b.ops.object.transform_apply(location=False, rotation=False, scale=True)
    except Exception as _e:
        print("VLMUNR fit-to-size failed:", _e)
    bpy.ops.object.select_all(action="DESELECT")
    loaded.select_set(True)
    bbox_corners_world = [
        loaded.matrix_world @ Vector(corner) for corner in loaded.bound_box
    ]
    bbox_center_world = sum(bbox_corners_world, Vector()) / 8

    bpy.context.scene.cursor.location = bbox_center_world
    bpy.ops.object.origin_set(type="ORIGIN_CURSOR")
    bpy.ops.transform.rotate(value=-math.radians(pre_rot_rad), orient_axis="Z")
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    bpy.context.object.rotation_mode = "YXZ"
    for i in range(3):
        loaded.rotation_euler[i] += rotation_rad[i]

    loaded.location = position
    print(loaded.location)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


# === ADD DYNAMIC FLOOR PLANE ===
all_objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]

if not all_objs:
    raise RuntimeError("No mesh objects found in scene.")

bbox_min = Vector((float("inf"), float("inf"), float("inf")))
bbox_max = Vector((float("-inf"), float("-inf"), float("-inf")))

for obj in all_objs:
    bpy.context.view_layer.update()
    for corner in obj.bound_box:
        world_corner = obj.matrix_world @ Vector(corner)
        for i in range(3):
            bbox_min[i] = min(bbox_min[i], world_corner[i])
            bbox_max[i] = max(bbox_max[i], world_corner[i])

min_coord = bbox_min
max_coord = bbox_max

floor_center_x = (min_coord.x + max_coord.x) / 2
floor_center_y = (min_coord.y + max_coord.y) / 2

x_extent = max_coord.x - min_coord.x
y_extent = max_coord.y - min_coord.y
# VLMUNR: pad the shell so floor/walls extend beyond the furniture -> room-like
_VLMUNR_SHELL_MARGIN = 1.0
x_extent = x_extent + 2*_VLMUNR_SHELL_MARGIN
y_extent = y_extent + 2*_VLMUNR_SHELL_MARGIN

bpy.ops.mesh.primitive_plane_add(
    size=1, location=(floor_center_x, floor_center_y, 0)
)
floor = bpy.context.active_object
floor.name = "Floor"
floor.scale = (x_extent, y_extent, 1)

floor = bpy.context.object
if floor.data.materials:
    floor.data.materials[0] = bpy.data.materials[material_name]
else:
    floor.data.materials.append(bpy.data.materials[material_name])


# === ADD FOUR WALLS (dollhouse: normal-driven back-face culling) ===

wall_height = 2.7
_wx = x_extent
_wy = y_extent
_wcx = floor_center_x
_wcy = floor_center_y

polyhaven_material_name = wall_material_name
polyhaven_mat = bpy.data.materials.get(polyhaven_material_name)

if not polyhaven_mat:
    raise NameError(
        f"Error: Material '{polyhaven_material_name}' not found. "
        "Please ensure it's loaded in the blend file before running this script."
    )


def create_wall(name, location, rotation, length_scale):
    bpy.ops.mesh.primitive_plane_add(
        size=1, location=location, rotation=rotation
    )
    wall = bpy.context.active_object
    wall.name = name
    wall.scale = (length_scale, wall_height, 1)
    print("wall instantiated")
    if wall.data.materials:
        wall.data.materials[0] = polyhaven_mat
    else:
        wall.data.materials.append(polyhaven_mat)
    return wall


def apply_backface_culling(material):
    """Dollhouse convention: make camera-facing (front) faces transparent and
    keep back-facing faces visible, driven by the surface normal. Mirrors
    bpa.Builder.add_material(..., backface_culling=True) node wiring, applied to
    an existing material so the loaded .blend wall texture is preserved.

    ShaderNodeNewGeometry.Backfacing is 1 for back-facing faces, 0 for front.
    MixShader(Fac=Backfacing): Fac=0 -> input[1]=Transparent (front->see-through),
    Fac=1 -> input[2]=BSDF (back->visible). So near walls' exteriors become
    transparent (camera sees into the room) while far walls remain visible.
    """
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    output = nodes.get("Material Output") or next(
        (n for n in nodes if n.type == "OUTPUT_MATERIAL"), None
    )
    if output is None:
        output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.get("Principled BSDF") or next(
        (n for n in nodes if n.type == "BSDF_PRINCIPLED"), None
    )
    if bsdf is None:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    # Drop any existing link into the output's Surface socket.
    for link in list(links):
        if link.to_node == output and link.to_socket == output.inputs["Surface"]:
            links.remove(link)
    mix = nodes.new("ShaderNodeMixShader")
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    geometry = nodes.new("ShaderNodeNewGeometry")
    links.new(bsdf.outputs["BSDF"], mix.inputs[2])
    links.new(mix.outputs["Shader"], output.inputs["Surface"])
    links.new(geometry.outputs["Backfacing"], mix.inputs["Fac"])
    links.new(transparent.outputs["BSDF"], mix.inputs[1])


# Apply the dollhouse back-face culling once to the shared wall material.
apply_backface_culling(polyhaven_mat)

# Back wall (Y+), Front wall (Y-), Left wall (X-), Right wall (X+). Each is a
# plane standing on edge; the culling material decides visibility per camera.
create_wall("BackWall",  (_wcx, _wcy + _wy/2, wall_height/2), (math.radians(90), 0, 0), _wx)
create_wall("FrontWall", (_wcx, _wcy - _wy/2, wall_height/2), (math.radians(90), 0, math.radians(180)), _wx)
create_wall("LeftWall",  (_wcx - _wx/2, _wcy, wall_height/2), (math.radians(90), 0, math.radians(90)), _wy)
create_wall("RightWall", (_wcx + _wx/2, _wcy, wall_height/2), (math.radians(90), 0, math.radians(-90)), _wy)


# === RENDER (bpa: tight-fit fit_ratio=1, transparent masters) ===
# bpa convention: pitch 0 == top-down. This renderer stores NATIVE pitch
# (90 == top-down), so remap via native_to_common_pitch before passing to bpa
# AND in the filename (cross-method consistency: 0 == top-down in filenames).

renderer = bpa.Renderer()
renderer.compute_world_vertices()
center, radius = renderer.compute_bounding_sphere()
if radius <= 0:
    raise RuntimeError("Degenerate bounding sphere (no visible geometry).")

if render_mode == "baseline":
    configs = [dict(res=BASELINE_RES, focal=BASELINE_FOCAL,
                    pitch=BASELINE_PITCH, yaw=BASELINE_YAW, hdri=BASELINE_HDRI)]
else:
    configs = ofat_camera_configs()

# Group by HDRI so the (expensive) environment-map swap happens once per HDRI.
by_hdri: dict[str, list[dict]] = {}
for c in configs:
    by_hdri.setdefault(c["hdri"], []).append(c)

for hdri, cfgs in by_hdri.items():
    hdri_path = os.path.join(HDRI_DIR, f"{hdri}.exr")
    # bpa.initialize: cycles/GPU + world + env map (lighting) + transparent film.
    bpa.initialize(transparent=True, environment_map=(hdri_path, 1.0))
    for c in cfgs:
        res, focal, native_pitch, yaw = c["res"], c["focal"], c["pitch"], c["yaw"]
        common_pitch = native_to_common_pitch(native_pitch)
        master_name = render_master_filename(res, focal, common_pitch, yaw, hdri)
        master_path = os.path.join(renderings_dir, master_name)
        print(f"Rendering: {master_name}")
        renderer.render_perspective(
            master_path,
            center,
            radius,
            rotation=(common_pitch, 0, yaw),
            resolution=res,
            focal_length=focal,
            fit_ratio=1.0,  # tight-fit (bpa logic)
            background=None,  # transparent master; bg composited in phase 2
        )

print(f"All transparent masters saved to: {renderings_dir}")
