import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from os.path import abspath, dirname, exists, join

from PIL import Image

sys.path.append(abspath(join(dirname(__file__), "..")))
from config import API_KEY, BASE_URL, DESCRIPTIONS, OBJ_DATA
from utils.llm import JsonResponseModel, Llm

# Maximum number of prefabs described concurrently.
MAX_WORKERS = 16

# One Llm client per worker thread. ``Llm`` mutates per-instance message
# history and drives a rich live spinner, so a single instance is not safe to
# share across threads. A thread-local gives each worker its own instance,
# reused across every prefab that thread handles — bounding the number of live
# OpenAI/httpx clients (and thus file descriptors / sockets) to MAX_WORKERS
# instead of leaking a fresh, never-closed client per call.
_llm_local = threading.local()


def _get_llm() -> Llm:
    """Return this thread's ``Llm`` client, creating it on first use."""
    llm = getattr(_llm_local, "llm", None)
    if llm is None:
        llm = Llm(
            "gemini-2.5-flash-lite",
            timeout=600,
            max_retries=5,
            api_key=API_KEY,
            base_url=BASE_URL,
        )
        _llm_local.llm = llm
    return llm


def get_structured_description(image_path1, image_path2, name):
    """
    Get a structured description of an object from two images using Gemini.

    Args:
        image_path1: Path to the first image file
        image_path2: Path to the second image file

    Returns:
        Dictionary containing structured description with physical, functional,
        and contextual properties
    """
    # Load and fully decode each image so the underlying file descriptor is
    # released immediately. ``Image.open`` is lazy: it keeps the file fd open
    # until ``load()``/``close()``. Calling ``load()`` inside the ``with``
    # guarantees the fd is closed before the (slow) LLM call, rather than
    # being held for its duration. The converted RGB copies are closed after
    # use so their pixel buffers are freed promptly instead of waiting for GC
    # under concurrent load.
    image1 = _load_rgb_image(image_path1)
    image2 = _load_rgb_image(image_path2)
    try:
        class Schema(JsonResponseModel):
            Physical_properties: str
            Functional_properties: str
            Contextual_properties: str
            name: str

        # Prepare the prompt for structured description
        prompt = f"""\
The two images show the same object from different angles/perspectives.
Please provide a detailed structured description of this object divided into these three categories:

1. Physical properties (size, shape, color, material, parts/components)
2. Functional properties (purpose, how it's used, what it does)
3. Contextual properties (where it might be found, what settings it belongs in)

Also name what the object is.

Output (JSON):
{Schema.to_str()}"""

        llm = _get_llm()
        # The client persists across calls on this thread; reset its
        # conversation state so prior prefabs don't bleed into this one and
        # so token usage / memory don't grow across the run.
        llm.clear_context()
        llm.clear_history()
        # verbose=False keeps rich panels/spinners from interleaving across
        # concurrent workers (rich live status is not thread-safe).
        llm_output = llm(
            prompt, Schema, images=[image1, image2], image_detail="low", verbose=False
        )
        response = llm_output.response.model_dump()
        return {k.replace("_", " "): v for k, v in response.items()}
    finally:
        image1.close()
        image2.close()


def _load_rgb_image(path):
    """Open an image from ``path``, fully decode it, and return an in-memory
    RGB copy. The source file descriptor is closed before this returns."""
    with Image.open(path) as img:
        img.load()  # force full decode; closes the file fp (opened from path)
        return img.convert("RGB")


def process_prefabs(json_file_path):
    """
    Process all prefabs in the JSON file and generate structured descriptions for each

    Args:
        json_file_path: Path to the prefab_data.json file

    Returns:
        Dictionary with prefab structured descriptions
    """

    # Set default output path if not provided
    output_file_path = DESCRIPTIONS

    # Load prefab data
    with open(json_file_path, "r") as file:
        prefab_data = json.load(file)

    # Check if description file already exists and load it
    existing_descriptions = {}
    if exists(output_file_path):
        print(f"Found existing descriptions file: {output_file_path}")
        with open(output_file_path, "r") as file:
            try:
                existing_descriptions = json.load(file)
                print(f"Loaded {len(existing_descriptions)} existing descriptions")
            except json.JSONDecodeError:
                print("Error loading existing descriptions. Starting fresh.")

    guids = [a["guid"] for a in existing_descriptions.values()]

    # Collect the prefabs that still need processing (order preserved).
    pending = []
    new_count = 0
    for prefab in prefab_data["prefabs"]:
        prefab_name = prefab["prefabName"]
        guid = prefab["guid"]

        # Skip if already processed
        if guid in guids:
            continue

        print(f"Processing new prefab: {prefab_name}")
        new_count += 1
        pending.append(prefab)

    def _describe(prefab):
        """Describe a single prefab. Returns (prefab, description_or_None)."""
        prefab_name = prefab["prefabName"]
        try:
            description = get_structured_description(
                prefab["imagePaths"][0], prefab["imagePaths"][1], prefab_name
            )
            return prefab, description
        except Exception as e:
            print(f"✗ Error processing {prefab_name}: {str(e)}")
            return prefab, None

    # Describe prefabs concurrently. Results are stored as they complete, so the
    # output dict/JSON is not guaranteed to match input order — that's fine
    # since downstream lookups are by key. Each completed result is flushed to
    # disk immediately, so a crash loses at most one in-flight item.
    total = len(pending)
    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_describe, prefab): prefab for prefab in pending}
        for future in as_completed(futures):
            completed += 1
            prefab, description = future.result()
            prefab_name = prefab["prefabName"]
            guid = prefab["guid"]

            if description is None:
                continue

            # Store result
            existing_descriptions[prefab_name] = {
                "guid": guid,
                "physical_properties": description.get("Physical properties"),
                "functional_properties": description.get("Functional properties"),
                "contextual_properties": description.get("Contextual properties"),
                "name": description.get("name"),
            }

            with open(output_file_path, "w") as file:
                json.dump(existing_descriptions, file, indent=4)
            print(f"✓ [{completed}/{total}] Successfully processed {prefab_name}")

    print(f"\nStructured descriptions saved to {output_file_path}")
    return existing_descriptions


# Run the script
if __name__ == "__main__":
    json_file_path = OBJ_DATA
    structured_descriptions = process_prefabs(json_file_path)
