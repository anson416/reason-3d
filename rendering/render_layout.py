import argparse
import json
import math
import os
import sys
from itertools import product

import bpy
from mathutils import Euler, Vector

# === PARSE ARGS (after Blender's -- separator) ===
argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
parser = argparse.ArgumentParser()
parser.add_argument(
    "--scene-dir", required=True, help="Path to timestamped scene dir"
)
args = parser.parse_args(argv)

scene_dir = args.scene_dir

sys.path.append(os.path.abspath("."))
from config import (
    ASSETS,
    FOCAL_LENGTHS,
    HDRI_DIR,
    HDRIS,
    PITCHS,
    RESOLUTIONS,
    YAWS,
)

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

# === CLEAR DEFAULT SCENE ===
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

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

    # Look for the first existing file with any supported extension
    for ext in extensions:
        candidate = os.path.join(asset_base_path, f"{uid}{ext}")
        if os.path.isfile(candidate):
            file_path = candidate
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
    # 1. Filter out non-mesh objects and identify the target objects for joining
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


# === ADD FOUR WALLS ===

wall_height = 2.7
scene_width = max_coord.x - min_coord.x
scene_depth = max_coord.y - min_coord.y

polyhaven_material_name = wall_material_name
polyhaven_mat = bpy.data.materials.get(polyhaven_material_name)

if not polyhaven_mat:
    raise NameError(
        f"Error: Material '{polyhaven_material_name}' not found. "
        "Please ensure it's loaded in the blend file before running the script."
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


# UNCOMMENT THIS BLOCK IF YOU WANT WALLS
# =========================
# # Back wall (Y+), facing -Y (inward)
# create_wall("BackWall", (offset_x,offset_y + y / 2, wall_height / 2), (math.radians(90), 0, 0), x)
#
# # Front wall (Y-), facing +Y (inward)
# create_wall("FrontWall", (offset_x, -y / 2 + offset_y, wall_height / 2), (math.radians(90), 0, math.radians(180)), x)
#
# # Left wall (X-), facing +X (inward)
# create_wall("LeftWall", (-x / 2 + offset_x, offset_y, wall_height / 2), (math.radians(90), 0, math.radians(90)), y)
#
# # Right wall (X+), facing -X (inward)
# create_wall("RightWall", (x / 2 + offset_x, offset_y, wall_height / 2), (math.radians(90), 0, math.radians(-90)), y)
# ========================


# === COMPUTE BOUNDING SPHERE FOR CAMERA POSITIONING ===
all_objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
if not all_objs:
    raise RuntimeError("No mesh objects found in scene.")

bs_min = Vector((float("inf"), float("inf"), float("inf")))
bs_max = Vector((float("-inf"), float("-inf"), float("-inf")))

for obj in all_objs:
    bpy.context.view_layer.update()
    for corner in obj.bound_box:
        world_corner = obj.matrix_world @ Vector(corner)
        for i in range(3):
            bs_min[i] = min(bs_min[i], world_corner[i])
            bs_max[i] = max(bs_max[i], world_corner[i])

scene_center = (bs_min + bs_max) / 2
scene_radius = (bs_max - scene_center).length


# === DELETE ALL EXISTING LIGHTS ===
for obj in [o for o in bpy.data.objects if o.type == "LIGHT"]:
    bpy.data.objects.remove(obj, do_unlink=True)

# === SET UP AREA LIGHT ===
scene = bpy.context.scene

area_light_name = "RoomAreaLight"
light_data = bpy.data.lights.new(name=area_light_name, type="AREA")
light_data.energy = 1000
light_data.size = 10

light = bpy.data.objects.new(name=area_light_name, object_data=light_data)
scene.collection.objects.link(light)
light.location = (floor_center_x, floor_center_y, 4)


# === GPU / CYCLES SETUP ===
def setup_cycles():
    cycles_prefs = bpy.context.preferences.addons["cycles"].preferences
    for device_type in ["OPTIX", "CUDA", "HIP", "METAL", "NONE"]:
        try:
            cycles_prefs.compute_device_type = device_type
            break
        except TypeError:
            continue
    cycles_prefs.get_devices()
    gpu_available = False
    for device in cycles_prefs.devices:
        if device.type == "CPU":
            device.use = False
        else:
            device.use = True
            gpu_available = True
    if not gpu_available:
        for device in cycles_prefs.devices:
            if device.type == "CPU":
                device.use = True
                break
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU" if gpu_available else "CPU"
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.02
    scene.cycles.samples = 128
    scene.cycles.use_denoising = True
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"


setup_cycles()


# === HDRI SETUP HELPER ===
def set_hdri(hdri_name, strength=0.5):
    world = scene.world
    if world is None:
        bpy.ops.world.new()
        world = bpy.context.scene.world = bpy.data.worlds[-1]
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    # Remove existing environment texture nodes
    for node in list(nodes):
        if node.type == "TEX_ENVIRONMENT":
            nodes.remove(node)

    bg = nodes["Background"]
    bg.inputs[1].default_value = strength

    env_path = os.path.join(HDRI_DIR, f"{hdri_name}.exr")
    env_node = nodes.new("ShaderNodeTexEnvironment")
    env_node.image = bpy.data.images.load(env_path, check_existing=True)
    links.new(env_node.outputs[0], bg.inputs[0])


# === CAMERA SETUP HELPER ===
def set_camera(pitch_deg, yaw_deg, focal_length):
    # Remove existing cameras
    for obj in [o for o in bpy.data.objects if o.type == "CAMERA"]:
        bpy.data.objects.remove(obj, do_unlink=True)

    cam_data = bpy.data.cameras.new("RenderCamera")
    cam_obj = bpy.data.objects.new("RenderCamera", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    cam_data.type = "PERSP"
    cam_data.lens_unit = "MILLIMETERS"
    cam_data.lens = focal_length

    # Camera distance from bounding sphere
    half_fov = cam_data.angle / 2
    distance = (
        scene_radius / math.sin(half_fov) if half_fov > 0 else scene_radius * 3
    )

    # Position camera using pitch/yaw (spherical coordinates)
    pitch_rad = math.radians(pitch_deg)
    yaw_rad = math.radians(yaw_deg)

    # Spherical to Cartesian: pitch=0 is horizontal, pitch=90 is top-down
    cam_x = scene_center.x + distance * math.cos(pitch_rad) * math.sin(yaw_rad)
    cam_y = scene_center.y - distance * math.cos(pitch_rad) * math.cos(yaw_rad)
    cam_z = scene_center.z + distance * math.sin(pitch_rad)

    cam_obj.location = (cam_x, cam_y, cam_z)

    # Point camera toward scene center
    direction = scene_center - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    cam_data.clip_start = 0.01
    cam_data.clip_end = distance * 3


# === RENDER LOOP ===
for hdri in HDRIS:
    set_hdri(hdri)
    for res, focal, pitch, yaw in product(
        RESOLUTIONS, FOCAL_LENGTHS, PITCHS, YAWS
    ):
        set_camera(pitch, yaw, focal)
        scene.render.resolution_x = res
        scene.render.resolution_y = res
        filename = f"render_{res}_{focal}_{pitch}_{yaw}_{hdri}.png"
        scene.render.filepath = os.path.join(renderings_dir, filename)
        print(f"Rendering: {filename}")
        bpy.ops.render.render(write_still=True)

print(f"All renders saved to: {renderings_dir}")
