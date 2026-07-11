import sys
from os.path import abspath, dirname, join

import numpy as np
from utilities import Attributes, get_attr_from_guid

sys.path.append(abspath(join(dirname(__file__), "..")))
from llm_config import GenConfig, build_llm
from preprocessing.CreateEmbeddings import get_embedding
from utils.llm import JsonResponseModel


# Legacy chat model for object-extraction / best-choice roles.
_EXTRACT_MODEL = "gemini-2.5-pro"


def get_scene_objects(scene_description, num_obj, gen=None):
    class Item(JsonResponseModel):
        name: str
        Physical_properties: str
        Functional_properties: str
        Contextual_properties: str
        quantity: int

    class Schema(JsonResponseModel):
        objects: list[Item]

    prompt = f"""\
Given the following scene description: {scene_description}

Please follow these steps:
Step 1: Identify """
    if num_obj != "":
        num_obj = f"{num_obj} "
    prompt += f"{num_obj}objects in the scene. "
    if num_obj != "":
        prompt += "If you think there are more objects focus on the most important ones. "
    prompt += f"""\
Try to be a bit broad and don't pick objects that are part of any other objects that you picked. E.g. Don't pick a bed and then pillows and blanket.
The objects can't be entire rooms or places. The objects also cannot be walls, floors, ceilings, doors, curtains, windows or anything else that has to do with the frame of the room. It also can't be the ground, water, sky or the sun.
Step 2: Without the context of the scene, describe each object, focusing on its physical properties (color, shape, material, texture, etc.), functional properties (what are its purpose, how it's used, and what it does?), and contextual properties (in what settings is it used, in which context might it be found, and what settings it belongs in?). Be detailed.
Step 3: Tell me how many duplicates of the object are required.

Output (JSON):
{Schema.to_str()}"""

    llm, temperature = build_llm(gen, _EXTRACT_MODEL)
    llm_output = llm(prompt, Schema, temperature=temperature)
    objects = [
        {k.replace("_", " "): v for k, v in item.model_dump().items()}
        for item in llm_output.response.objects
    ]
    return {"objects": objects}


# When the LLM rejects every candidate (index 0), fall back to the top-1
# similarity match only if it clears this threshold -- a high-similarity match
# that the LLM spuriously rejected (common under a slow/flaky proxy) is far
# more useful to placement than dropping the object entirely, which can nuke
# the whole scene when it happens to every object. Below the threshold the
# match is genuinely weak, so the object is dropped as before.
_TOP1_FALLBACK_SIMILARITY = 0.6


def find_assets_for_scene(scene_description, embeddings_data, top_n, gen=None):
    """
    Generate descriptions of assets required for the scene and return best matches.

    Args:
        scene_description: Description of the scene
        embeddings_data: Dictionary containing prefab embeddings
        top_n: Number of objects to return
        gen: Optional GenConfig override for the chat-LLM roles.

    Returns:
        List of top matching prefabs
    """
    # Get objects needed for scene
    scene_objects = get_scene_objects(scene_description, top_n, gen)

    # Build the prefab embedding matrices ONCE (vectorized cosine similarity).
    # The previous implementation called sklearn.cosine_similarity once per
    # prefab per property field (~150k individual calls per object on a 50k
    # asset DB), which made each scene take 10-15 min through a flaky proxy and
    # increased the chance of degenerate LLM responses downstream. The math
    # here is identical: cosine similarity is the dot product of L2-normalized
    # vectors, computed as a single matrix multiply per field.
    prefab_names = list(embeddings_data.keys())
    prefab_phys = np.array(
        [embeddings_data[n]["embedding_phys"] for n in prefab_names], dtype=np.float32
    )
    prefab_func = np.array(
        [embeddings_data[n]["embedding_func"] for n in prefab_names], dtype=np.float32
    )
    prefab_cont = np.array(
        [embeddings_data[n]["embedding_cont"] for n in prefab_names], dtype=np.float32
    )
    prefab_guids = [embeddings_data[n]["guid"] for n in prefab_names]
    # L2-normalize rows; +eps guards against a zero vector.
    prefab_phys /= np.linalg.norm(prefab_phys, axis=1, keepdims=True) + 1e-9
    prefab_func /= np.linalg.norm(prefab_func, axis=1, keepdims=True) + 1e-9
    prefab_cont /= np.linalg.norm(prefab_cont, axis=1, keepdims=True) + 1e-9

    chosen_assets = []
    for asset in scene_objects.get("objects", scene_objects):
        name = asset.get("name")
        phys_desc, func_desc, cont_desc = (
            asset.get("Physical properties"),
            asset.get("Functional properties"),
            asset.get("Contextual properties"),
        )
        object_phys_embedding, object_func_embedding, object_cont_embedding = (
            get_embedding(f"{name}: {phys_desc}"),
            get_embedding(f"{name}: {func_desc}"),
            get_embedding(f"{name}: {cont_desc}"),
        )
        # Normalize the query vectors, then similarity = mean of three cosine
        # sims (one per property field) -- same weighting as before.
        object_phys_embedding /= np.linalg.norm(object_phys_embedding) + 1e-9
        object_func_embedding /= np.linalg.norm(object_func_embedding) + 1e-9
        object_cont_embedding /= np.linalg.norm(object_cont_embedding) + 1e-9
        similarities = (
            prefab_phys @ object_phys_embedding
            + prefab_func @ object_func_embedding
            + prefab_cont @ object_cont_embedding
        ) / 3.0

        # Top candidates (fewer than 5 if the asset database is small), highest
        # similarity first.
        order = np.argsort(-similarities)[:5]
        candidates = [
            {
                "prefab_name": prefab_names[i],
                "guid": prefab_guids[i],
                "quantity": asset.get("quantity"),
                "similarity": float(similarities[i]),
            }
            for i in order
        ]
        #############
        # Let LLM decide from the candidates
        #############
        get_attr_from_guid(Attributes.FULL_DESCRIPTION, candidates, [])
        get_attr_from_guid(Attributes.NAME, candidates, [])
        index = pick_best_choice(
            f"{name}: {phys_desc} {func_desc} {cont_desc}",
            [
                {
                    k: v
                    for k, v in a.items()
                    if k == "Full description" or k == "name"
                }
                for a in candidates
            ],
            gen,
        )
        if 0 < index <= len(candidates):
            chosen_assets.append(candidates[index - 1])
        else:
            # LLM rejected every candidate (index 0 / out of range). Fall back
            # to the top-1 match when it is strong enough that a spurious
            # rejection (slow proxy, degenerate output) is the more likely
            # explanation than a genuinely absent object.
            top1 = candidates[0] if candidates else None
            if top1 is not None and top1["similarity"] >= _TOP1_FALLBACK_SIMILARITY:
                print(
                    f"[retrieve] LLM rejected all candidates for '{name}'; "
                    f"falling back to top-1 match (similarity="
                    f"{top1['similarity']:.3f}, guid={top1['guid']})."
                )
                chosen_assets.append(top1)
            else:
                chosen_assets.append(
                    {
                        "guid": 0,
                        "reason": f"Object not found. No matches exist for '{name}' in your database.",
                    }
                )
                print(
                    f"Object not found. No matches exist for '{name}' in your database."
                )
    return chosen_assets


def pick_best_choice(target_object, top5, gen=None):
    class Schema(JsonResponseModel):
        index: int

    prompt = f"""\
You are given a description of a target object: {target_object}

You are also given a list of five objects with their descriptions: {top5}

Your task is to tell me which one of the five objects in the list matches the description of the target object best. 
Do this by giving me a number between 1 and 5, which serves as an index of the list.
If you think that no object in the list can be can be used as a substitution for the target object, please output a 0.

Output (JSON):
{Schema.to_str()}"""

    llm, temperature = build_llm(gen, _EXTRACT_MODEL)
    llm_output = llm(prompt, Schema, temperature=temperature)
    return llm_output.response.index
