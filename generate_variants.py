#!/usr/bin/env python3
"""Generate ablation variants from an existing Reason-3D scene.

Creates content-perturbation variants for the VLM-unreliability audit:
  - 3 reduced-object variants (1/2, 1/4, 1/8 of objects)
  - 1 layout-scramble variant (relocate every object within the scene
    footprint; object set and rotation preserved, arrangement destroyed)
  - 3 alternative-asset variants (worst similarity matches at indices 0, 2, 4)
  - 2 substitution variants (within-category and cross-category swaps)

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
from rendering.convert_for_blender import convert, export_meshes


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
    # Copy each placed object's source mesh into <variant_dir>/meshes/ so every
    # variant folder is self-contained (variant_04 swaps guids, so its meshes
    # differ from the base's -- the copy is keyed by the new guid).
    export_meshes(variant_dir)


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


def scramble_positions(placed_objects, seed):
    """Relocate every object's center to a random point within the scene's XZ
    footprint, preserving the object set, rotation, and size. Returns a new
    list of placed-object dicts (deep-copied).

    Pure function (seeded): the object count, names, rotations and sizes are
    unchanged; only the x and z of ``center`` move, within the bounding box of
    the original object centers (y/height preserved).
    """
    rng = random.Random(seed)
    objs = copy.deepcopy(placed_objects)
    if not objs:
        return objs
    xs = [o["center"][0] for o in objs]
    zs = [o["center"][2] for o in objs]
    xmin, xmax = min(xs), max(xs)
    zmin, zmax = min(zs), max(zs)
    # Degenerate (single object / colinear): fall back to a small jitter box.
    if xmax <= xmin:
        xmin, xmax = xmin - 0.5, xmax + 0.5
    if zmax <= zmin:
        zmin, zmax = zmin - 0.5, zmax + 0.5
    for o in objs:
        o["center"] = [
            rng.uniform(xmin, xmax),
            o["center"][1],
            rng.uniform(zmin, zmax),
        ]
    return objs


def generate_scramble_variant(
    scene_dir, placed_objects, placed_objects_data, prompt_text, seed
):
    variant_objs = scramble_positions(placed_objects, seed)
    variant_dir = os.path.join(scene_dir, "variant_scramble")
    write_variant(
        variant_dir, variant_objs, copy.deepcopy(placed_objects_data), prompt_text
    )
    print(f"  variant_scramble: {len(variant_objs)} objects relocated")


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


def _apply_swap(variant_objs, variant_data, i, new_obj_data, new_guid):
    """Swap object i's asset to ``new_guid`` while PRESERVING the original
    object's bounding box.

    Per the audit methodology, a substituted mesh must be rescaled to the
    ORIGINAL bounding box so the probe isolates object identity from geometry.
    We therefore only change the asset ``guid`` (identity) and keep the
    original ``size`` / ``size_after_rotation`` / ``boundsCenter``; the renderer
    then scales the new mesh to fit that original size.
    """
    variant_data[i]["guid"] = new_guid
    # size, size_after_rotation, boundsCenter, center, rotation all UNCHANGED.


def generate_substitution_variants(
    scene_dir, placed_objects, placed_objects_data, prompt_text
):
    """Two substitution variants that hold position/rotation fixed and swap the
    asset only:

      variant_subst_within -- swap each object for the MOST similar different
        instance (same semantic category, different asset).
      variant_subst_cross  -- swap each object for a dissimilar asset whose
        base name differs (a different category).

    Both reuse the embedding ranking from ``compute_similarities`` (ascending =
    worst first), so within = last entry (highest similarity) and cross = first
    entry (lowest similarity, which for distinct base names is a different
    category).
    """
    with open(EMBEDDINGS) as f:
        embeddings_data = json.load(f)
    with open(OBJ_DATA) as f:
        obj_data = json.load(f)
    obj_data_by_guid = {p["guid"]: p for p in obj_data["prefabs"]}

    unique_guids = {}
    for i, data in enumerate(placed_objects_data):
        guid = data["guid"]
        if guid not in unique_guids:
            unique_guids[guid] = strip_trailing_digits(placed_objects[i]["name"])

    guid_rankings = {}
    for guid, base_name in unique_guids.items():
        emb = get_embedding(base_name)
        guid_rankings[guid] = compute_similarities(emb, embeddings_data, guid)

    for mode in ("within", "cross"):
        variant_objs = copy.deepcopy(placed_objects)
        variant_data = copy.deepcopy(placed_objects_data)
        for i, data in enumerate(variant_data):
            rankings = guid_rankings[data["guid"]]  # ascending: worst..best
            if not rankings:
                continue
            # within: most similar (end); cross: least similar (start)
            _, new_guid, _ = rankings[-1] if mode == "within" else rankings[0]
            _apply_swap(variant_objs, variant_data, i, obj_data_by_guid[new_guid], new_guid)
        dirname = f"variant_subst_{mode}"
        variant_dir = os.path.join(scene_dir, dirname)
        write_variant(variant_dir, variant_objs, variant_data, prompt_text)
        print(f"  {dirname}: {mode}-category substitution")


# ---------------------------------------------------------------------------
# Named content-perturbation variants (variant_01 .. variant_04).
#
# These are pure local edits of an already-generated base scene: NO LLM calls,
# NO embedding API calls, NO regeneration. They fork the in-progress scene's
# placed_objects + placed_objects_data and rewrite them. Each writes to its own
# subfolder containing placed_objects.json, placed_objects_data.json,
# prompt.txt and raw_blender.json.
# ---------------------------------------------------------------------------

NAMED_VARIANT_DIRS = [
    "variant_01_half",
    "variant_02_biggest-only",
    "variant_03_scrambled",
    "variant_04_worst-object",
]


def _object_volume(obj):
    """Bounding-box volume (w*h*d) of a placed object, used to rank by size."""
    size = obj.get("size_after_rotation") or obj.get("size")
    if not size or len(size) < 3:
        return 0.0
    return float(size[0]) * float(size[1]) * float(size[2])


def generate_half_variant(
    scene_dir, placed_objects, placed_objects_data, prompt_text, seed
):
    """Keep a random ~50% subset of the objects (positions/rotations intact)."""
    rng = random.Random(seed)
    total = len(placed_objects)
    n = max(1, round(total * 0.5)) if total else 0
    indices = sorted(rng.sample(range(total), n)) if total else []
    variant_objs = [placed_objects[i] for i in indices]
    variant_data = [placed_objects_data[i] for i in indices]
    variant_dir = os.path.join(scene_dir, "variant_01_half")
    write_variant(variant_dir, variant_objs, variant_data, prompt_text)
    print(f"  variant_01_half: kept {n}/{total} objects")


def generate_biggest_only_variant(
    scene_dir, placed_objects, placed_objects_data, prompt_text
):
    """Keep only the single largest object (by bounding-box volume)."""
    if not placed_objects:
        # Nothing to keep; write an empty scene for downstream symmetry.
        variant_dir = os.path.join(scene_dir, "variant_02_biggest-only")
        write_variant(variant_dir, [], [], prompt_text)
        print("  variant_02_biggest-only: empty scene (no objects)")
        return
    best_i = max(range(len(placed_objects)), key=lambda i: _object_volume(placed_objects[i]))
    variant_objs = [placed_objects[best_i]]
    variant_data = [placed_objects_data[best_i]]
    variant_dir = os.path.join(scene_dir, "variant_02_biggest-only")
    write_variant(variant_dir, variant_objs, variant_data, prompt_text)
    print(
        f"  variant_02_biggest-only: kept '{variant_objs[0]['name']}' "
        f"(volume={_object_volume(variant_objs[0]):.3f})"
    )


def _scene_footprint(placed_objects):
    """(xmin,xmax,zmin,zmax) over object centers; degenerate-safe."""
    if not placed_objects:
        return 0.0, 0.0, 0.0, 0.0
    xs = [o["center"][0] for o in placed_objects]
    zs = [o["center"][2] for o in placed_objects]
    xmin, xmax = min(xs), max(xs)
    zmin, zmax = min(zs), max(zs)
    if xmax <= xmin:
        xmin, xmax = xmin - 0.5, xmax + 0.5
    if zmax <= zmin:
        zmin, zmax = zmin - 0.5, zmax + 0.5
    return xmin, xmax, zmin, zmax


def generate_scrambled_variant(
    scene_dir, placed_objects, placed_objects_data, prompt_text, seed
):
    """Re-position AND re-rotate every object anywhere inside the whole-scene
    XZ footprint (object set, size and height preserved). Destroys the
    arrangement while keeping every object in the scene."""
    rng = random.Random(seed + 1)
    objs = copy.deepcopy(placed_objects)
    if objs:
        xmin, xmax, zmin, zmax = _scene_footprint(objs)
        for o in objs:
            o["center"] = [
                rng.uniform(xmin, xmax),
                o["center"][1],
                rng.uniform(zmin, zmax),
            ]
            # Re-rotate: random Y-axis heading; keep the floor-plane convention
            # (no X/Z tilt that would sink objects into the floor).
            o["rotation"] = [
                0.0,
                float(rng.choice([0, 90, 180, 270])),
                0.0,
            ]
            o["size_after_rotation"] = get_rotated_bounding_box(
                o.get("size", o.get("size_after_rotation")), o["rotation"]
            )
    variant_dir = os.path.join(scene_dir, "variant_03_scrambled")
    write_variant(variant_dir, objs, copy.deepcopy(placed_objects_data), prompt_text)
    print(f"  variant_03_scrambled: {len(objs)} objects re-positioned/rotated")


# A prefab is treated as usable for the worst-object swap only if its bounding
# box is physically plausible on every axis. Outside this band the asset is a
# dataset/scaling artefact (e.g. the 165,000 m or 0.02 m outliers) whose
# near-orthogonal embedding would otherwise win as the "least similar" prefab
# for almost every furniture query and collapse the whole variant onto one
# broken object. The normal selection algorithm never picks these (they are
# never top-similar to anything), so mirroring that here keeps the worst-match
# swap meaningful.
_WORST_MIN_AXIS_M = 0.02
_WORST_MAX_AXIS_M = 50.0


def _plausible_prefab_guids(obj_data):
    """Set of prefab guids with every ``boundsSize`` axis in the plausible band."""
    valid = set()
    for p in obj_data.get("prefabs", []):
        b = p.get("boundsSize") or {}
        try:
            x, y, z = abs(b["x"]), abs(b["y"]), abs(b["z"])
        except (KeyError, TypeError):
            continue
        if (
            _WORST_MIN_AXIS_M <= x <= _WORST_MAX_AXIS_M
            and _WORST_MIN_AXIS_M <= y <= _WORST_MAX_AXIS_M
            and _WORST_MIN_AXIS_M <= z <= _WORST_MAX_AXIS_M
        ):
            valid.add(p["guid"])
    return valid


def _worst_match_similarities(query_guid, embeddings_data, valid_guids):
    """Prefabs ranked LEAST-similar-first to the asset ``query_guid`` (zero API
    calls: the chosen asset's own phys/func/cont embeddings live in
    embeddings.json), restricted to plausible prefabs.

    Comparison mirrors ``Object_retriever.find_assets_for_scene`` (mean of
    cosine(phys, phys) + cosine(func, func) + cosine(cont, cont)), but ranked
    ASCENDING so the first entry is the worst match -- the selection algorithm
    picks the best; this is the post-hoc "hack the sort to descending and pick
    the first" applied to the already-chosen asset.
    """
    q_name = next(
        (n for n, d in embeddings_data.items() if d["guid"] == query_guid), None
    )
    q = embeddings_data.get(q_name)
    if q is None:
        return []
    q_phys = np.array(q["embedding_phys"])
    q_func = np.array(q["embedding_func"])
    q_cont = np.array(q["embedding_cont"])
    ranked = []  # (sim, guid) ascending
    for data in embeddings_data.values():
        guid = data["guid"]
        if guid == query_guid or guid not in valid_guids:
            continue
        sim = (
            cosine_similarity([q_phys], [np.array(data["embedding_phys"])])[0][0]
            + cosine_similarity([q_func], [np.array(data["embedding_func"])])[0][0]
            + cosine_similarity([q_cont], [np.array(data["embedding_cont"])])[0][0]
        ) / 3
        ranked.append((sim, guid))
    ranked.sort(key=lambda x: x[0])
    return ranked


def generate_worst_object_variant(
    scene_dir, placed_objects, placed_objects_data, prompt_text
):
    """Swap each object's asset for a distinct least-similar prefab.

    Mirrors the object-selection algorithm (rank prefabs by mean cosine
    similarity, pick the best) but inverts the pick: each object gets the
    WORST-matching prefab, computed from the stored embeddings of its already
    chosen asset (so zero API calls). Greedy distinct assignment -- each object
    takes its worst prefab not already claimed by an earlier object -- stops the
    variant collapsing onto the single globally-most-dissimilar prefab (a
    near-orthogonal outlier that wins for most furniture queries), so every
    object becomes its OWN worst match instead of all becoming the same one.

    Only the asset identity (``guid``) changes; ``size``, ``size_after_rotation``,
    ``boundsCenter``, ``center`` and ``rotation`` are preserved so the renderer
    rescales each worst-matching mesh to the object's ORIGINAL intended
    dimensions (isolating object identity from geometry, like the substitution
    variants' ``_apply_swap``).
    """
    with open(EMBEDDINGS) as f:
        embeddings_data = json.load(f)
    with open(OBJ_DATA) as f:
        obj_data = json.load(f)
    valid_guids = _plausible_prefab_guids(obj_data)

    variant_objs = copy.deepcopy(placed_objects)
    variant_data = copy.deepcopy(placed_objects_data)

    # Cache each unique chosen-asset's worst-match ranking (the ranking is
    # identical for duplicate placements of the same asset, e.g. two Nightstands).
    rankings_cache = {}

    def ranking_for(guid):
        if guid not in rankings_cache:
            rankings_cache[guid] = _worst_match_similarities(
                guid, embeddings_data, valid_guids
            )
        return rankings_cache[guid]

    taken = set()
    swapped = 0
    for i, data in enumerate(variant_data):
        for _sim, new_guid in ranking_for(data["guid"]):
            if new_guid not in taken:
                taken.add(new_guid)
                # Identity-only swap: only the guid changes; size /
                # size_after_rotation / boundsCenter / center / rotation are
                # kept, so the renderer fits the new mesh to the ORIGINAL size.
                variant_data[i]["guid"] = new_guid
                swapped += 1
                break

    variant_dir = os.path.join(scene_dir, "variant_04_worst-object")
    write_variant(variant_dir, variant_objs, variant_data, prompt_text)
    print(
        f"  variant_04_worst-object: {swapped}/{len(variant_objs)} objects "
        f"swapped to distinct worst matches"
    )


def generate_named_variants(
    scene_dir, placed_objects, placed_objects_data, prompt_text, seed
):
    """Produce the four named variants (01_half, 02_biggest-only, 03_scrambled,
    04_worst-object) as sibling subfolders of ``scene_dir``."""
    generate_half_variant(
        scene_dir, placed_objects, placed_objects_data, prompt_text, seed
    )
    generate_biggest_only_variant(
        scene_dir, placed_objects, placed_objects_data, prompt_text
    )
    generate_scrambled_variant(
        scene_dir, placed_objects, placed_objects_data, prompt_text, seed
    )
    generate_worst_object_variant(
        scene_dir, placed_objects, placed_objects_data, prompt_text
    )


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

    print("\nGenerating layout-scramble variant...")
    generate_scramble_variant(
        scene_dir, placed_objects, placed_objects_data, prompt_text, args.seed
    )

    # VLMUNR: alt_* (worst-match) variants dropped per methodology.
    # generate_alt_variants(scene_dir, placed_objects, placed_objects_data, prompt_text)

    print("\nGenerating substitution variants (within/cross category)...")
    generate_substitution_variants(
        scene_dir, placed_objects, placed_objects_data, prompt_text
    )

    print("\nDone! Generated 9 variants.")


if __name__ == "__main__":
    main()
