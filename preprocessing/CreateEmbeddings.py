import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from os.path import abspath, dirname, exists, join

import numpy as np

sys.path.append(abspath(join(dirname(__file__), "..")))
from config import API_KEY, BASE_URL, DESCRIPTIONS, EMBEDDINGS
from utils.llm import TextEmbedder

# Maximum number of embeddings generated concurrently. The shared
# TextEmbedder only talks to the (thread-safe) OpenAI client, so calls can
# run in parallel safely.
MAX_WORKERS = 16


def get_embedding(text):
    """
    Get embedding for text using the configured embedding model.

    Reuses a single module-level ``TextEmbedder`` client instead of building a
    new one per call (the previous version spawned a fresh HTTP client for
    every embedding — 3x per object during generation, which is wasteful and
    slow under rate limits). The client itself is initialized eagerly so the
    lazy-init ``global`` assignment doesn't race under concurrent workers.

    Args:
        text: The text to embed

    Returns:
        Embedding array
    """
    return _embedder(text, np.ndarray)


def get_embeddings(texts):
    """
    Get embeddings for multiple texts in a single API request.

    One HTTP round-trip through the proxy replaces one-per-text, so embedding
    a prefab's three property fields issues 1 request instead of 3. Order is
    preserved: result[i] corresponds to texts[i].

    Args:
        texts: The texts to embed

    Returns:
        List of embedding vectors (one per input text)
    """
    return _embedder.embed_batch(list(texts), list)


_embedder = TextEmbedder(
    model="text-embedding-3-small", api_key=API_KEY, base_url=BASE_URL
)


def embed_descriptions(descriptions_file, embedding_file_path):
    """
    Embed all prefab descriptions and save the embeddings

    Args:
        descriptions_file: Path to the JSON file with prefab descriptions

    Returns:
        Dictionary with prefab embeddings
    """
    # Load the descriptions
    with open(descriptions_file, "r") as file:
        prefab_desc = json.load(file)

    embeddings_data = {}
    if exists(embedding_file_path):
        print(f"Found existing descriptions file: {embedding_file_path}")
        with open(embedding_file_path, "r") as file:
            try:
                embeddings_data = json.load(file)
                print(f"Loaded {len(embeddings_data)} existing descriptions")
            except json.JSONDecodeError:
                print("Error loading existing descriptions. Starting fresh.")
    guids = [a["guid"] for a in embeddings_data.values()]

    # Collect prefabs that still need embeddings (order preserved).
    pending = []
    for prefab_name, data in prefab_desc.items():
        if data["guid"] in guids:
            continue
        pending.append((prefab_name, data))

    def _embed_prefab(prefab_name, data):
        """Embed one prefab's three property fields.

        Returns (prefab_name, result_dict_or_None) so callers can store results
        in submission order.
        """
        try:
            phys = data["physical_properties"]
            func = data["functional_properties"]
            cont = data["contextual_properties"]
            guid = data["guid"]

            # Embed all three fields in a single request. The embeddings
            # endpoint accepts a list of inputs, so one round-trip through
            # the proxy replaces three (plus the per-prefab thread pool the
            # old version spun up). Order is preserved: phys -> func -> cont.
            embedding_phys, embedding_func, embedding_cont = get_embeddings(
                [phys, func, cont]
            )

            return prefab_name, {
                "guid": guid,
                "embedding_phys": embedding_phys,
                "embedding_func": embedding_func,
                "embedding_cont": embedding_cont,
            }
        except Exception as e:
            print(f"Error processing {prefab_name}: {str(e)}")
            return prefab_name, None

    print(f"Generating embeddings for {len(pending)} prefabs...")
    # Process prefabs concurrently. Results are stored as they complete, so the
    # output dict/JSON is not guaranteed to match input order — that's fine
    # since downstream lookups are by key. Each completed result is flushed to
    # disk immediately, so a crash loses at most one in-flight item.
    total = len(pending)
    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_embed_prefab, prefab_name, data): prefab_name
            for prefab_name, data in pending
        }
        for future in as_completed(futures):
            completed += 1
            prefab_name, result = future.result()
            if result is None:
                continue

            embeddings_data[prefab_name] = result
            with open(embedding_file_path, "w") as file:
                json.dump(embeddings_data, file, indent=4)
            print(f"✓ [{completed}/{total}] Generated embedding for: {prefab_name}")

    # Save embeddings to a file
    with open(embedding_file_path, "w") as file:
        json.dump(embeddings_data, file, indent=4)

    print(f"Embeddings saved to {embedding_file_path}")


def main():
    descriptions_file = DESCRIPTIONS
    embedding_file_path = EMBEDDINGS
    embed_descriptions(descriptions_file, embedding_file_path)


if __name__ == "__main__":
    main()
