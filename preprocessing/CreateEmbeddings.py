import json
import os
import sys
import tempfile
import threading
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

# Flush the accumulated results to disk at most every this many new items.
# Writing the (large) embedding dict is O(n) per flush, so flushing after
# every single completion makes the run O(n^2) — near the end each per-item
# write took ~27s and starved all workers, making the run look sequential.
# A background flusher drains a dirty set on this cadence instead, so the
# worker pool is never blocked on disk I/O.
FLUSH_EVERY = 1000


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


def _salvage_json(s):
    """Recover complete top-level key/value entries from a truncated JSON
    object string. The embeddings file is written non-atomically; if the run
    is interrupted mid-write, ``json.load`` throws and (previously) silently
    dropped every embedding already paid for. This scans the truncated text
    and keeps every value that decoded in full, stopping at the first
    incomplete/truncated value. Returns a dict of the salvaged entries."""
    dec = json.JSONDecoder()
    i = s.find("{", 0)
    if i == -1:
        return {}
    i += 1
    salvaged = {}
    while i < len(s):
        while i < len(s) and s[i] in " ,\n\t\r":
            i += 1
        if i >= len(s) or s[i] == "}":
            break
        try:
            key, end = dec.raw_decode(s, i)
        except json.JSONDecodeError:
            break
        j = end
        while j < len(s) and s[j] in " \n\t\r":
            j += 1
        if j >= len(s) or s[j] != ":":
            break
        j += 1
        while j < len(s) and s[j] in " \n\t\r":
            j += 1
        try:
            val, j2 = dec.raw_decode(s, j)
        except json.JSONDecodeError:
            break  # truncated value -> stop, don't include partial
        salvaged[key] = val
        i = j2
    return salvaged


def _load_existing(embedding_file_path):
    """Load existing embeddings, tolerating a file truncated by an
    interrupted write. Returns a dict of complete entries."""
    if not exists(embedding_file_path):
        return {}
    with open(embedding_file_path, "r") as file:
        s = file.read()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # The previous run was interrupted mid-write, leaving a truncated
        # file. Salvage the complete entries instead of silently starting
        # fresh and throwing away already-paid-for embeddings.
        salvaged = _salvage_json(s)
        print(
            f"Existing file was truncated/incomplete; salvaged "
            f"{len(salvaged)} complete embeddings."
        )
        return salvaged


def _save_atomic(embeddings_data, embedding_file_path):
    """Write embeddings atomically: serialize to a temp file in the same
    directory, then os.replace onto the target path. os.replace is atomic on
    POSIX, so a crash mid-write can never leave a half-written file (the
    cause of the truncated-file bug). Uses separators with no indent for a
    ~2-3x smaller file and faster serialize/parse."""
    d = os.path.dirname(embedding_file_path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as file:
            json.dump(embeddings_data, file, separators=(",", ":"))
        os.replace(tmp, embedding_file_path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


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

    # Load existing embeddings, tolerating a truncated file from an
    # interrupted prior write (see _load_existing).
    embeddings_data = _load_existing(embedding_file_path)
    if embeddings_data:
        print(f"Loaded {len(embeddings_data)} existing embeddings")
    existing_guids = {a["guid"] for a in embeddings_data.values()}

    # Collect prefabs that still need embeddings (order preserved).
    pending = []
    for prefab_name, data in prefab_desc.items():
        if data["guid"] in existing_guids:
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

    # Results accumulate here, guarded by a lock (the worker pool completes
    # them concurrently). The expensive disk write happens on a separate
    # background thread on a throttled cadence so the main loop (and thus the
    # workers) is never blocked on serialization/disk — the cause of the
    # apparent "sequential" behavior, since each full-dict re-dump grew O(n).
    dirty = threading.Event()
    stop_flusher = threading.Event()
    lock = threading.Lock()

    def _flush():
        snapshot = dict(embeddings_data)
        _save_atomic(snapshot, embedding_file_path)

    def _flusher():
        while True:
            if dirty.wait(timeout=5.0):
                dirty.clear()
                with lock:
                    _flush()
            if stop_flusher.is_set():
                return

    flusher = threading.Thread(target=_flusher, daemon=True)
    flusher.start()

    total = len(pending)
    completed = 0
    try:
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

                with lock:
                    embeddings_data[prefab_name] = result
                    # Mark dirty; signal at most every FLUSH_EVERY items so
                    # the flusher wakes on a bounded cadence, not per-item.
                    if completed % FLUSH_EVERY == 0:
                        dirty.set()
                print(f"✓ [{completed}/{total}] Generated embedding for: {prefab_name}")
    finally:
        # Final flush of everything, then stop the flusher thread.
        with lock:
            dirty.set()
        stop_flusher.set()
        dirty.set()
        flusher.join(timeout=60.0)
        with lock:
            _flush()

    print(f"Embeddings saved to {embedding_file_path}")


def main():
    descriptions_file = DESCRIPTIONS
    embedding_file_path = EMBEDDINGS
    embed_descriptions(descriptions_file, embedding_file_path)


if __name__ == "__main__":
    main()
