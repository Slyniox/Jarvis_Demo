#!/usr/bin/env python3

"""
build_faiss_index.py

Builds a multi-resolution FAISS memory index for Jarvis.

For each diary entry:

    1. One "summary" embedding is created from the metadata.summary field.

    2. The original diary entry is split into 250-token chunks with
       40-token overlap. Each chunk gets its own embedding.

Both vector types point back to the parent entries.id.

FAISS mapping:

    FAISS vector ID -> {
        "type": "summary" | "chunk",
        "entry_id": <entries.id>,
        "date": <entry_date>,
        "chunk_index": <int | null>
    }

The same FAISS index contains both summary and chunk vectors.
"""

import sqlite3
import json
from datetime import datetime

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

# Chunking
CHUNK_SIZE = 250
CHUNK_OVERLAP = 40

# Embedding
BATCH_SIZE = 4


###############################################################################
# Text construction
###############################################################################

def build_summary_embedding_text(date, summary, events):
    """
    Build the document used for the day-level embedding.

    The embedding contains:
        - date
        - metadata.summary
        - metadata.events

    Other metadata fields are intentionally excluded.
    """

    date = date.split("T")[0]

    dt = datetime.strptime(date, "%Y-%m-%d")

    # Convert events to readable text.
    if isinstance(events, list):
        events_text = "\n".join(
            f"- {event}" for event in events
        )
    elif events:
        events_text = str(events)
    else:
        events_text = "None"

    return (
        f"Date: {date}\n"
        f"Year: {dt.year}\n"
        f"Month: {dt.strftime('%B')}\n"
        f"\n"
        f"Daily summary:\n"
        f"{summary}\n"
        f"\n"
        f"Events:\n"
        f"{events_text}"
    )


def build_chunk_embedding_text(date, chunk):
    """
    Build the document used for a diary chunk embedding.
    """

    date = date.split("T")[0]

    return (
        f"Date: {date}\n"
        f"\n"
        f"Diary entry:\n"
        f"{chunk}"
    )


###############################################################################
# Database
###############################################################################

def load_entries(connection):
    """
    Load diary entries together with the summary and events
    extracted from the metadata JSON.
    """

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            e.id,
            e.entry_date,
            e.text,
            m.json
        FROM entries e
        LEFT JOIN metadata m
            ON e.id = m.entry_id
        ORDER BY e.entry_date ASC, e.id ASC;
        """
    )

    rows = cursor.fetchall()

    results = []

    for entry_id, date, diary_text, metadata_json in rows:

        summary = None
        events = None

        if metadata_json:

            try:
                metadata = json.loads(metadata_json)

                summary = metadata.get("summary")
                events = metadata.get("events")

            except (json.JSONDecodeError, TypeError):

                print(
                    f"Warning: invalid metadata JSON "
                    f"for entry {entry_id}"
                )

        results.append(
            (
                entry_id,
                date,
                diary_text,
                summary,
                events
            )
        )

    return results


###############################################################################
# Token-based chunking
###############################################################################

def chunk_text(
    text,
    tokenizer,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):
    """
    Split diary text according to the embedding model tokenizer.

    Example with 250-token chunks and 40-token overlap:

        chunk 0: tokens   0 -> 250
        chunk 1: tokens 210 -> 460
        chunk 2: tokens 420 -> 670
        ...

    The tokenizer used here is the tokenizer belonging to the embedding
    model, so the chunk size corresponds to actual model tokens.
    """

    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError(
            "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
        )

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=False
    )

    if not token_ids:
        return []

    step = chunk_size - overlap

    chunks = []

    for start in range(0, len(token_ids), step):

        end = min(
            start + chunk_size,
            len(token_ids)
        )

        chunk_ids = token_ids[start:end]

        chunk = tokenizer.decode(
            chunk_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip()

        if chunk:
            chunks.append(chunk)

        # Stop once the final token has been included.
        if end >= len(token_ids):
            break

    return chunks


###############################################################################
# Main
###############################################################################

def main():

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    # Use the tokenizer belonging to the embedding model.
    tokenizer = model.tokenizer

    print("Opening database...")

    connection = sqlite3.connect(DATABASE)

    rows = load_entries(connection)

    print(f"{len(rows)} diary entries found.")

    ###########################################################################
    # Prepare embedding documents
    ###########################################################################

    texts = []
    mapping = []

    summary_count = 0
    chunk_count = 0

    for row in rows:

        entry_id = row[0]
        date = row[1]
        diary_text = row[2]
        summary = row[3]
        events = row[4]

        #######################################################################
        # 1. SUMMARY VECTOR
        #######################################################################

        if summary and summary.strip():

            summary_text = build_summary_embedding_text(
                date,
                summary,
                events
            )

            texts.append(summary_text)

            mapping.append({
                "type": "summary",
                "entry_id": entry_id,
                "date": date,
                "chunk_index": None
            })

            summary_count += 1

        #######################################################################
        # 2. DIARY CHUNK VECTORS
        #######################################################################

        chunks = chunk_text(
            diary_text,
            tokenizer,
            CHUNK_SIZE,
            CHUNK_OVERLAP
        )

        for chunk_index, chunk in enumerate(chunks):

            chunk_text_for_embedding = build_chunk_embedding_text(
                date,
                chunk
            )

            texts.append(chunk_text_for_embedding)

            mapping.append({
                "type": "chunk",
                "entry_id": entry_id,
                "date": date,
                "chunk_index": chunk_index
            })

            chunk_count += 1

    connection.close()

    ###########################################################################
    # Statistics
    ###########################################################################

    print()
    print("Documents prepared:")
    print(f"  Summary vectors : {summary_count}")
    print(f"  Chunk vectors   : {chunk_count}")
    print(f"  Total vectors   : {len(texts)}")
    print()

    ###########################################################################
    # Generate embeddings
    ###########################################################################

    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        prompt_name="document",
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype(np.float32)

    dimension = embeddings.shape[1]

    print()
    print(f"Embedding dimension: {dimension}")

    ###########################################################################
    # Build FAISS index
    ###########################################################################

    print("Building FAISS index...")

    # Normalized embeddings + inner product = cosine similarity.
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    ###########################################################################
    # Save FAISS index
    ###########################################################################

    print("Saving FAISS index...")

    faiss.write_index(
        index,
        INDEX_FILE
    )

    ###########################################################################
    # Save mapping
    ###########################################################################

    print("Saving mapping...")

    mapping_data = {
        "model": MODEL_NAME,

        "dimension": dimension,

        "chunk_size": CHUNK_SIZE,

        "chunk_overlap": CHUNK_OVERLAP,

        "vectors": mapping
    }

    with open(
        MAPPING_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            mapping_data,
            f,
            indent=4,
            ensure_ascii=False
        )

    ###########################################################################
    # Final statistics
    ###########################################################################

    print()
    print("Done.")
    print()
    print(f"Vectors : {index.ntotal}")
    print(f"  Summary : {summary_count}")
    print(f"  Chunks  : {chunk_count}")
    print()
    print(f"Index   : {INDEX_FILE}")
    print(f"Mapping : {MAPPING_FILE}")


###############################################################################

if __name__ == "__main__":
    main()