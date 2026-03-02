import os

# Global variables
API_KEY = os.environ.get("CA_API_KEY3")
BASE_URL = "https://api.chatanywhere.tech/v1"

# Paths
git_root = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.expanduser(
    "~/reason_assets/"
)  # PATH to your folder with 3d objects (.fbx, .obj, .glb, .blend)
IMAGES = os.path.expanduser(
    "~/reason_images/"
)  # PATH where to save all images from the preprocessing
DESCRIPTIONS = os.path.join(
    git_root, "data/descriptions.json"
)  # PATH to object descriptions
EMBEDDINGS = os.path.join(
    git_root, "data/embeddings.json"
)  # PATH to description embeddings
OBJ_DATA = os.path.join(
    git_root, "data/object_data.json"
)  # PATH to object metadata
ROTATION_DATA = os.path.join(
    git_root, "data/rotation_data.json"
)  # PATH to fixed rotation data
RESULTS = os.path.join(git_root, "results")  # PATH to the results folder
OUTPUT = os.path.join(RESULTS, "raw_outputs")  # Raw pipeline output
BLENDER_FILE = os.path.join(
    RESULTS, "raw_blender.json"
)  # Converted for blender
RENDERS = os.path.join(RESULTS, "final_renders")  # Final renders (legacy)

# === Rendering Constants ===
ColorRGB = tuple[int, int, int]

HDRI_DIR = os.path.join(git_root, "rendering", "hdri")
RESOLUTIONS: list[int] = [
    # 224,
    # 256,
    # 384,
    # 448,
    512,
    # 640,
    # 768,
    # 1024,
]
BACKGROUND_COLORS: list[ColorRGB] = [
    # (0, 0, 0),
    # (18, 18, 18),  # Dark mode
    # (65, 65, 65),  # 5% physically
    # (117, 117, 117),  # 18% physically
    # (128, 128, 128),
    # (186, 186, 186),  # 50% physically
    # (204, 204, 204),  # 80% visually
    (255, 255, 255),
]
FOCAL_LENGTHS: list[int] = [
    # 24,
    # 35,
    50,
    # 85,
    # 100,
    # 200,
]
PITCHS: list[int] = [
    # 60,
    90,
]
YAWS: list[int] = [
    0,
    # 30,
    # 60,
    # 90,
    # 120,
    # 150,
    # 180,
    # 210,
    # 240,
    # 270,
    # 300,
    # 330,
]
HDRIS: list[str] = [
    "city",
    # "courtyard",
    # "forest",
    # "interior",
    # "night",
    # "studio",
    # "sunrise",
    # "sunset",
]
