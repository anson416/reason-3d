import sys
from os.path import abspath, dirname, join

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utilities import Attributes, get_attr_from_guid

sys.path.append(abspath(join(dirname(__file__), "..")))
from config import API_KEY, BASE_URL
from preprocessing.CreateEmbeddings import get_embedding
from utils.llm import JsonResponseModel, Llm


def get_scene_objects(scene_description, num_obj):
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

    llm = Llm(
        "gemini-2.5-pro",
        max_tokens=32768,
        timeout=600,
        max_retries=5,
        api_key=API_KEY,
        base_url=BASE_URL,
    )
    llm_output = llm(prompt, Schema)
    objects = [
        {k.replace("_", " "): v for k, v in item.model_dump().items()}
        for item in llm_output.response.objects
    ]
    return {"objects": objects}


def find_assets_for_scene(scene_description, embeddings_data, top_n):
    """
    Generate 10 descriptions of assets required for the scene and return best matches.

    Args:
        scene_description: Description of the scene
        embeddings_data: Dictionary containing prefab embeddings
        top_n: Number of objects to return

    Returns:
        List of top matching prefabs
    """
    # Get objects needed for scene
    scene_objects = get_scene_objects(scene_description, top_n)

    # Calculate similarity with each prefab

    chosen_assets = []
    top5 = []
    for asset in scene_objects.get("objects", scene_objects):
        similarities = []
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
        for prefab_name, data in embeddings_data.items():
            prefab_embedding_phys = np.array(data["embedding_phys"])
            prefab_embedding_func = np.array(data["embedding_func"])
            prefab_embedding_cont = np.array(data["embedding_cont"])

            # Calculate cosine similarity
            similarity = (
                cosine_similarity(
                    [object_phys_embedding], [prefab_embedding_phys]
                )[0][0]
                / 3
                + cosine_similarity(
                    [object_func_embedding], [prefab_embedding_func]
                )[0][0]
                / 3
                + cosine_similarity(
                    [object_cont_embedding], [prefab_embedding_cont]
                )[0][0]
                / 3
            )

            similarities.append(
                {
                    "prefab_name": prefab_name,
                    "guid": data["guid"],
                    "quantity": asset.get("quantity"),
                    "similarity": similarity,
                }
            )

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x["similarity"], reverse=True)

        top5.append(similarities[:5])
        #############
        # Let LLM decide from top5
        #############
        get_attr_from_guid(Attributes.FULL_DESCRIPTION, top5[-1], [])
        get_attr_from_guid(Attributes.NAME, top5[-1], [])
        index = pick_best_choice(
            f"{name}: {phys_desc} {func_desc} {cont_desc}",
            [
                {
                    k: v
                    for k, v in a.items()
                    if k == "Full description" or k == "name"
                }
                for a in top5[-1]
            ],
        )
        if 0 < index < 6:
            chosen_assets.append(top5[-1][index - 1])
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


def pick_best_choice(target_object, top5):
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

    llm = Llm(
        "gemini-2.5-pro",
        max_tokens=32768,
        timeout=600,
        max_retries=5,
        api_key=API_KEY,
        base_url=BASE_URL,
    )
    llm_output = llm(prompt, Schema)
    return llm_output.response.index
