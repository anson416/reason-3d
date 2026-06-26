#!/usr/bin/env python3
"""Generate ablation variants from an existing Reason-3D scene.

Creates 6 variants:
  - 3 reduced-object variants (1/2, 1/4, 1/8 of objects)
  - 3 alternative-asset variants (worst similarity matches at indices 0, 2, 4)

Usage:
    python generate_variants.py --scene-dir results/2026-03-03_01-56-08/
    python generate_variants.py --scene-dir results/2026-03-03_01-56-08/ --seed 123
"""

import argparse
import copy
import json
import os
import random
import re
import sys

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from build_scene.utilities import get_rotated_bounding_box
from config import EMBEDDINGS, OBJ_DATA
from preprocessing.CreateEmbeddings import get_embedding
from rendering.convert_for_blender import convert


def load_scene(scene_dir):
    with open(os.path.join(scene_dir, "placed_objects.json")) as f:
        placed_objects = json.load(f)
    with open(os.path.join(scene_dir, "placed_objects_data.json")) as f:
        placed_objects_data = json.load(f)
    prompt_path = os.path.join(scene_dir, "prompt.txt")
    prompt_text = ""
    if os.path.exists(prompt_path):
        with open(prompt_path) as f:
            prompt_text = f.read()
    return placed_objects, placed_objects_data, prompt_text


def strip_trailing_digits(name):
    """'Armchair4' -> 'Armchair', 'Cushion/Pillow5' -> 'Cushion/Pillow'."""
    return re.sub(r"\d+$", "", name)


def write_variant(
    variant_dir, placed_objects, placed_objects_data, prompt_text
):
    os.makedirs(variant_dir, exist_ok=True)
    with open(os.path.join(variant_dir, "placed_objects.json"), "w") as f:
        json.dump(placed_objects, f, indent=4)
    with open(os.path.join(variant_dir, "placed_objects_data.json"), "w") as f:
        json.dump(placed_objects_data, f, indent=4)
    with open(os.path.join(variant_dir, "prompt.txt"), "w") as f:
        f.write(prompt_text)
    convert(variant_dir)


def generate_reduced_variants(
    scene_dir, placed_objects, placed_objects_data, prompt_text, seed
):
    rng = random.Random(seed)
    total = len(placed_objects)

    fractions = [
        (1 / 2, "variant_half"),
        (1 / 4, "variant_quarter"),
        (1 / 8, "variant_eighth"),
    ]

    for fraction, dirname in fractions:
        n = max(1, round(total * fraction))
        indices = sorted(rng.sample(range(total), n))

        variant_objs = [placed_objects[i] for i in indices]
        variant_data = [placed_objects_data[i] for i in indices]

        variant_dir = os.path.join(scene_dir, dirname)
        write_variant(variant_dir, variant_objs, variant_data, prompt_text)
        print(f"  {dirname}: {n}/{total} objects")


def compute_similarities(query_embedding, embeddings_data, exclude_guid):
    """Compute similarity against all prefab embeddings, sorted ascending (worst first)."""
    results = []
    for prefab_name, data in embeddings_data.items():
        if data["guid"] == exclude_guid:
            continue
        phys = np.array(data["embedding_phys"])
        func = np.array(data["embedding_func"])
        cont = np.array(data["embedding_cont"])

        sim = (
            cosine_similarity([query_embedding], [phys])[0][0]
            + cosine_similarity([query_embedding], [func])[0][0]
            + cosine_similarity([query_embedding], [cont])[0][0]
        ) / 3

        results.append((prefab_name, data["guid"], sim))

    results.sort(key=lambda x: x[2])  # ascending = worst first
    return results


def generate_alt_variants(
    scene_dir, placed_objects, placed_objects_data, prompt_text
):
    with open(EMBEDDINGS) as f:
        embeddings_data = json.load(f)
    with open(OBJ_DATA) as f:
        obj_data = json.load(f)

    obj_data_by_guid = {p["guid"]: p for p in obj_data["prefabs"]}

    # Find unique GUIDs and their base names
    unique_guids = {}
    for i, data in enumerate(placed_objects_data):
        guid = data["guid"]
        if guid not in unique_guids:
            base_name = strip_trailing_digits(placed_objects[i]["name"])
            unique_guids[guid] = base_name

    print(f"  Found {len(unique_guids)} unique asset types")

    # Compute embedding for each unique asset name (one API call per asset)
    guid_embeddings = {}
    for guid, base_name in unique_guids.items():
        print(f"    Embedding: {base_name}")
        guid_embeddings[guid] = get_embedding(base_name)

    # Compute similarity rankings for each unique GUID
    guid_rankings = {}
    for guid, embedding in guid_embeddings.items():
        guid_rankings[guid] = compute_similarities(
            embedding, embeddings_data, guid
        )

    # Generate variants at pick indices 0, 2, 4
    pick_indices = [0, 2, 4]
    for pick_idx in pick_indices:
        variant_objs = copy.deepcopy(placed_objects)
        variant_data = copy.deepcopy(placed_objects_data)

        for i, data in enumerate(variant_data):
            orig_guid = data["guid"]
            rankings = guid_rankings[orig_guid]

            # Fall back to last available if list is shorter than pick_idx + 1
            actual_idx = min(pick_idx, len(rankings) - 1)
            _, new_guid, sim = rankings[actual_idx]

            new_obj_data = obj_data_by_guid[new_guid]

            # Convert {x,y,z} dict to [x,y,z] array
            new_bounds_center = [
                new_obj_data["boundsCenter"]["x"],
                new_obj_data["boundsCenter"]["y"],
                new_obj_data["boundsCenter"]["z"],
            ]
            new_size = [
                new_obj_data["boundsSize"]["x"],
                new_obj_data["boundsSize"]["y"],
                new_obj_data["boundsSize"]["z"],
            ]

            # Update placed_objects_data
            variant_data[i]["guid"] = new_guid
            variant_data[i]["boundsCenter"] = new_bounds_center
            variant_data[i]["size"] = new_size

            # Update placed_objects: size and recompute size_after_rotation
            variant_objs[i]["size"] = new_size
            variant_objs[i]["size_after_rotation"] = get_rotated_bounding_box(
                new_size, variant_objs[i]["rotation"]
            )

        dirname = f"variant_alt_{pick_idx}"
        variant_dir = os.path.join(scene_dir, dirname)
        write_variant(variant_dir, variant_objs, variant_data, prompt_text)
        print(f"  {dirname}: worst-match index {pick_idx} (sim={sim:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description="Generate ablation variants from an existing scene"
    )
    parser.add_argument(
        "--scene-dir", required=True, help="Path to scene directory"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reduced variants"
    )
    args = parser.parse_args()

    scene_dir = args.scene_dir
    if not os.path.isdir(scene_dir):
        print(f"Error: {scene_dir} is not a directory")
        sys.exit(1)

    placed_objects, placed_objects_data, prompt_text = load_scene(scene_dir)
    print(f"Loaded scene: {len(placed_objects)} objects")

    print("\nGenerating reduced-object variants...")
    generate_reduced_variants(
        scene_dir, placed_objects, placed_objects_data, prompt_text, args.seed
    )

    print("\nGenerating alternative-asset variants...")
    generate_alt_variants(
        scene_dir, placed_objects, placed_objects_data, prompt_text
    )

    print("\nDone! Generated 6 variants.")


if __name__ == "__main__":
    main()
