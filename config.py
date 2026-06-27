import os

# Global variables
API_KEY = os.environ.get("CA_API_KEY3")
BASE_URL = "https://api.chatanywhere.tech/v1"
# Text-reasoning backbone for placement (text-agnostic role), standardised on a
# single pinned snapshot for the audit.
MODEL = "gpt-5.1-2025-11-13"

# Paths
git_root = os.path.dirname(os.path.abspath(__file__))
# Asset/image dirs default to large scratch storage (home quota is tiny);
# override with VLMUNR_REASON_ASSETS / VLMUNR_REASON_IMAGES if needed.
_ASSETS_DEFAULT = "/research/d2/fyp24/yflam1/reason_assets/"
_IMAGES_DEFAULT = "/research/d2/fyp24/yflam1/reason_images/"
ASSETS = os.environ.get(
    "VLMUNR_REASON_ASSETS",
    _ASSETS_DEFAULT if os.path.isdir(_ASSETS_DEFAULT) else os.path.expanduser("~/reason_assets/"),
)  # PATH to your folder with 3d objects (.fbx, .obj, .glb, .blend)
IMAGES = os.environ.get(
    "VLMUNR_REASON_IMAGES",
    _IMAGES_DEFAULT if os.path.isdir(_IMAGES_DEFAULT) else os.path.expanduser("~/reason_images/"),
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
# Factor levels for the VLM-unreliability audit (paper Table 1).
# CAMERA CONVENTION: this renderer uses pitch == 90 deg for the top-down view
# (camera directly above; cam_z = distance*sin(pitch)) and pitch == 0 for a
# horizontal eye-level view -- the OPPOSITE of the bpa-based methods
# (genxr / HSM / LayoutVLM / IDesign), where pitch == 0 is top-down. The
# cross-method analysis maps each method onto a common "tilt-from-top-down"
# axis via the per-method camera manifest; see INTEGRATION.md.
RESOLUTIONS: list[int] = [196, 224, 256, 336, 384, 448, 512, 768, 1024]
BACKGROUND_COLORS: list[ColorRGB] = [
    (0, 0, 0),
    (65, 65, 65),  # 5% physically
    (128, 128, 128),
    (186, 186, 186),  # 50% physically
    (204, 204, 204),  # 80% visually
    (255, 255, 255),
    (255, 0, 0),  # chromatic (paper Table 1)
    (0, 255, 0),
    (0, 0, 255),
]
FOCAL_LENGTHS: list[int] = [16, 24, 35, 50, 85, 100, 200]
# top-down (90) -> eye-level (0) in 15-deg steps == paper's 7 pitch levels.
PITCHS: list[int] = [90, 75, 60, 45, 30, 15, 0]
# 8 azimuths at 45-deg steps; yaw sweep run at the oblique pitch below.
YAWS: list[int] = [0, 45, 90, 135, 180, 225, 270, 315]
YAW_SWEEP_PITCH: int = 45
HDRIS: list[str] = [
    "city",
    "courtyard",
    "forest",
    "interior",
    "night",
    "studio",
    "sunrise",
    "sunset",
]

# Baselines (held constant for any factor not being swept).
BASELINE_RES: int = 512
BASELINE_FOCAL: int = 50
BASELINE_BG: ColorRGB = (128, 128, 128)
BASELINE_HDRI: str = "city"
BASELINE_PITCH: int = 90  # top-down in THIS renderer's convention
BASELINE_YAW: int = 0


def ofat_camera_configs() -> list[dict]:
    """One-factor-at-a-time (res, focal, pitch, yaw, hdri) camera configs.

    Matches the paper's OFAT design (NOT a full Cartesian product, which would
    be infeasible). Background is composited separately over each master, so it
    is not part of these camera keys. Returns de-duplicated config dicts.
    """
    base = dict(
        res=BASELINE_RES,
        focal=BASELINE_FOCAL,
        pitch=BASELINE_PITCH,
        yaw=BASELINE_YAW,
        hdri=BASELINE_HDRI,
    )
    seen: set[tuple] = set()
    out: list[dict] = []

    def add(**ov):
        c = {**base, **ov}
        k = (c["res"], c["focal"], c["pitch"], c["yaw"], c["hdri"])
        if k not in seen:
            seen.add(k)
            out.append(c)

    for r in RESOLUTIONS:  # 1a resolution
        add(res=r)
    for f in FOCAL_LENGTHS:  # 1d focal
        add(focal=f)
    for h in HDRIS:  # 1c lighting
        add(hdri=h)
    for p in PITCHS:  # 2 pitch (at baseline yaw)
        add(pitch=p)
    for y in YAWS:  # 2 yaw (at oblique pitch)
        add(pitch=YAW_SWEEP_PITCH, yaw=y)
    return out


def ofat_backgrounds() -> list[ColorRGB]:
    """Background colours composited over the baseline-camera master (1b)."""
    return list(BACKGROUND_COLORS)
