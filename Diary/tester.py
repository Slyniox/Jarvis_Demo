#!/usr/bin/env python3

import json
import sqlite3

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


###############################################################################
# Configuration
###############################################################################

DATABASE = "LJDB_FR.db"
INDEX_FILE = "memory.index"
MAPPING_FILE = "memory_mapping.json"

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"


###############################################################################
# Retrieval
###############################################################################

def debug_retrieval(
    query,
    model,
    index,
    mapping,
    cursor,
    k=5
):

    print()
    print(f"Query: {query}")
    print("=" * 100)

    vectors = mapping["vectors"]

    # ------------------------------------------------------------------
    # Embed query
    # ------------------------------------------------------------------

    query_embedding = model.encode(
        [query],
        prompt_name="query",
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    scores, indices = index.search(
        query_embedding,
        k
    )

    # ------------------------------------------------------------------
    # Display results
    # ------------------------------------------------------------------

    for rank, (score, faiss_id) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):

        if faiss_id < 0:
            continue

        result = vectors[faiss_id]

        vector_type = result["type"]
        entry_id = result["entry_id"]
        date = result["date"]
        chunk_index = result["chunk_index"]

        # --------------------------------------------------------------
        # Retrieve parent diary entry
        # --------------------------------------------------------------

        cursor.execute(
            """
            SELECT text
            FROM entries
            WHERE id = ?
            """,
            (entry_id,)
        )

        entry_row = cursor.fetchone()

        entry_text = (
            entry_row[0]
            if entry_row
            else "[ENTRY NOT FOUND]"
        )

        # --------------------------------------------------------------
        # Print result
        # --------------------------------------------------------------

        print()
        print(f"#{rank}")
        print("-" * 100)

        print(f"FAISS ID     : {faiss_id}")
        print(f"Similarity   : {score:.4f}")
        print(f"Type         : {vector_type}")
        print(f"Entry ID     : {entry_id}")
        print(f"Date         : {date}")

        if vector_type == "chunk":
            print(f"Chunk index  : {chunk_index}")

        # --------------------------------------------------------------
        # Metadata
        # --------------------------------------------------------------

        if vector_type == "metadata":

            cursor.execute(
                """
                SELECT json
                FROM metadata
                WHERE entry_id = ?
                """,
                (entry_id,)
            )

            metadata_row = cursor.fetchone()

            print()
            print("Metadata used for embedding:")
            print("-" * 60)

            if metadata_row:
                print(metadata_row[0])
            else:
                print("[METADATA NOT FOUND]")

        # --------------------------------------------------------------
        # Parent entry
        # --------------------------------------------------------------

        print()
        print("Full parent entry:")
        print("-" * 60)
        print(entry_text)

    print()
    print("=" * 100)


###############################################################################
# Main
###############################################################################

def main():

    print("Loading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME,
        model_kwargs={"dtype": "float16"})

    print("Loading FAISS index...")

    index = faiss.read_index(INDEX_FILE)

    print("Loading mapping...")

    with open(
        MAPPING_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        mapping = json.load(f)

    print("Opening database...")

    connection = sqlite3.connect(DATABASE)

    cursor = connection.cursor()

    print()
    print("Ready.")
    print("Enter a query, or type 'exit' / 'quit' to stop.")
    print()

    # ------------------------------------------------------------------
    # Interactive loop
    # ------------------------------------------------------------------

    while True:

        try:
            query = input("Query > ").strip()

        except (KeyboardInterrupt, EOFError):

            print()
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit", "q"):

            break

        debug_retrieval(
            query=query,
            model=model,
            index=index,
            mapping=mapping,
            cursor=cursor,
            k=5
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    connection.close()

    print("Goodbye.")


###############################################################################

if __name__ == "__main__":
    main()