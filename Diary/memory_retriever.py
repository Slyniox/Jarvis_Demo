#!/usr/bin/env python3
"""
memory_retriever_v2.py

Multi-resolution diary retrieval:

1. FAISS retrieves both summary vectors and diary-entry chunks.
2. Results are deduplicated by parent entry_id.
3. Each planner query reranks candidates using ONLY the full diary entry text.
4. Results from multiple planner queries are merged.
5. Only after reranking do we decide which entries are returned in full
   and which are represented by their metadata summaries.
"""

import json
import sqlite3
from pathlib import Path

import faiss
import numpy as np


from sentence_transformers import SentenceTransformer
from memory_reranker import rerank_memory


BASE_DIR = Path(__file__).resolve().parent

DATABASE = BASE_DIR / "LJDB_FR.db"
INDEX_FILE = BASE_DIR / "memory.index"
MAPPING_FILE = BASE_DIR / "memory_mapping.json"
PROFILE_DIR = BASE_DIR / "profiles"

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

# Retrieval
TOP_K = 30
MIN_SIMILARITY = 0.32
MEMORY_WINDOW = 0.175

# Final context
MAX_FULL_ENTRIES = 5
MAX_SUMMARIES = 3


class MemoryRetriever:

    def __init__(self):

        print("Loading Qwen 0.6B embedder...")

        self.model = SentenceTransformer(
            MODEL_NAME,
            local_files_only=True,
            device="cpu",
            model_kwargs={"dtype": "float16"}
        )

        print("Qwen 0.6B embedder loaded.")

        self.index = faiss.read_index(str(INDEX_FILE))

        with open(MAPPING_FILE, "r", encoding="utf-8") as f:
            mapping = json.load(f)

        # New multi-resolution mapping
        self.mapping = mapping["vectors"]

        self.profile_cache = {}

        self.people = {
            profile.stem.lower(): profile.stem
            for profile in PROFILE_DIR.glob("*.md")
        }

    # ------------------------------------------------------------------
    # DATABASE
    # ------------------------------------------------------------------

    def get_db(self):

        db = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row

        return db

    def fetch_full_entry(self, entry_id):

        db = self.get_db()

        try:
            row = db.execute(
                """
                SELECT id, entry_date, text
                FROM entries
                WHERE id = ?
                """,
                (entry_id,)
            ).fetchone()

            if row is None:
                return None

            return {
                "id": row["id"],
                "date": row["entry_date"],
                "entry": row["text"],
            }

        finally:
            db.close()

    def fetch_metadata(self, entry_id):

        db = self.get_db()

        try:
            row = db.execute(
                """
                SELECT
                    e.id,
                    e.entry_date,
                    m.json
                FROM entries e
                LEFT JOIN metadata m
                    ON m.entry_id = e.id
                WHERE e.id = ?
                """,
                (entry_id,)
            ).fetchone()

            if row is None:
                return None

            metadata = {}

            if row["json"]:
                try:
                    metadata = json.loads(row["json"])
                except json.JSONDecodeError:
                    metadata = {}

            metadata["id"] = row["id"]
            metadata["date"] = row["entry_date"]

            return metadata

        finally:
            db.close()

    # ------------------------------------------------------------------
    # EMBEDDING
    # ------------------------------------------------------------------

    def embed_query(self, query):

        embedding = self.model.encode(
            query,
            prompt_name="query",
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        return embedding.astype(np.float32)

    # ------------------------------------------------------------------
    # FAISS CANDIDATE RETRIEVAL
    # ------------------------------------------------------------------

    def retrieve_candidates(self, query, top_k: int):

        embedding = self.embed_query(query)

        scores, indices = self.index.search(
            embedding.reshape(1, -1),
            top_k
        )

        scores = scores[0]
        indices = indices[0]

        candidates = {}

        for score, faiss_id in zip(scores, indices):

            if faiss_id < 0:
                continue

            # Stop once similarity becomes too low
            if score < MIN_SIMILARITY:
                continue

            vector = self.mapping[faiss_id]

            entry_id = vector["entry_id"]

            # Keep only the strongest vector belonging to each
            # parent diary entry.
            if entry_id in candidates:
                if score <= candidates[entry_id]["embedding_score"]:
                    continue

            entry = self.fetch_full_entry(entry_id)

            if entry is None:
                continue

            metadata = self.fetch_metadata(entry_id)

            candidates[entry_id] = {
                "id": entry["id"],
                "date": entry["date"],

                # IMPORTANT:
                # This is the complete original diary entry.
                "entry": entry["entry"],

                # Metadata is kept for later, but is NOT sent
                # to the reranker.
                "metadata": metadata,

                "summary": (
                    metadata.get("summary")
                    if metadata
                    else None
                ),

                "embedding_score": float(score),

                "retrieval_type": vector.get("type"),

                "retrieval_chunk": vector.get("chunk_index"),

                "faiss_id": int(faiss_id),

                "retrieval_query": query,
            }

        return list(candidates.values())

    # ------------------------------------------------------------------
    # RERANKING
    # ------------------------------------------------------------------

    def rerank_candidates(self, query, candidates):

        if not candidates:
            return []

        # Build the retrieval structure expected by rerank_memory().
        #
        # IMPORTANT:
        # We intentionally put ONLY the complete diary entries here.
        # Metadata and summaries are NOT sent to the reranker.
        rerank_retrieval = {
            "full_entries": [
                {
                    "id": candidate["id"],
                    "date": candidate["date"],
                    "entry": candidate["entry"],
                }
                for candidate in candidates
            ],
            "summaries": [],
        }

        # Existing reranker expects:
        #
        #     rerank_memory(query, retrieval)
        #
        reranked_retrieval = rerank_memory(
            query,
            rerank_retrieval,
            top_k=len(candidates)
        )

        # The reranker returns the entries in reranked order.
        reranked_entries = reranked_retrieval["full_entries"]

        # Map our original candidates by ID so we can restore all
        # the retrieval information and metadata.
        candidate_by_id = {
            candidate["id"]: candidate
            for candidate in candidates
        }

        results = []

        for rank, entry in enumerate(reranked_entries):

            entry_id = entry["id"]

            candidate = candidate_by_id.get(entry_id)

            if candidate is None:
                continue

            result = candidate.copy()

            result["rerank_rank"] = rank
            result["retrieval_query"] = query

            # The reranker attaches this to the entry.
            if "score" in entry:
                result["score"] = entry["score"]

            results.append(result)

        return results

    # ------------------------------------------------------------------
    # SINGLE QUERY
    # ------------------------------------------------------------------

    def retrieve_for_query(self, query, top_k: int):

        candidates = self.retrieve_candidates(query, top_k)

        if not candidates:
            return []

        return self.rerank_candidates(
            query,
            candidates
        )

    # ------------------------------------------------------------------
    # MULTI QUERY
    # ------------------------------------------------------------------

    def retrieve_multiple(self, queries, top_k: int, max_full_entries: int, max_summaries: int):

        if not queries:
            return {
                "profiles": [],
                "full_entries": [],
                "summaries": [],
                "best_similarity": None,
                "all_results": [],
            }

        merged = {}

        # --------------------------------------------------------------
        # Each planner query gets its own:
        #
        #     FAISS retrieval
        #     ↓
        #     reranking
        #
        # This preserves the specific intent of each planner query.
        # --------------------------------------------------------------

        for query in queries:

            query = query.strip()

            if not query:
                continue

            results = self.retrieve_for_query(query, top_k)

            for result in results:

                entry_id = result["id"]

                if entry_id not in merged:

                    merged[entry_id] = {
                        **result,
                        "query_matches": []
                    }

                merged[entry_id]["query_matches"].append({
                    "query": query,
                    "rerank_rank": result["rerank_rank"],
                    "score": result.get("score"),
                    "embedding_score": result["embedding_score"],
                })

        # --------------------------------------------------------------
        # Merge scores across planner queries.
        #
        # The best reranker result is dominant.
        # Additional independent query matches provide a small bonus.
        # --------------------------------------------------------------

        results = []

        for result in merged.values():

            matches = result["query_matches"]

            rerank_scores = [
                match["score"]
                for match in matches
                if match["score"] is not None
            ]

            if rerank_scores:

                rerank_scores.sort(reverse=True)

                final_score = rerank_scores[0]

                # Small bonus when multiple planner queries
                # independently identify the same entry.
                if len(rerank_scores) > 1:
                    final_score += 0.05 * sum(
                        rerank_scores[1:3]
                    )

            else:

                # Fallback if rerank_memory doesn't expose scores.
                best_rank = min(
                    match["rerank_rank"]
                    for match in matches
                )

                final_score = -float(best_rank)

            result["final_score"] = final_score

            results.append(result)

        # Highest final score first
        results.sort(
            key=lambda x: x["final_score"],
            reverse=True
        )

        # --------------------------------------------------------------
        # ONLY NOW decide full entry vs summary.
        #
        # Reranking has already happened.
        # --------------------------------------------------------------

        full_entries = results[:max_full_entries]

        summaries = results[
            max_full_entries:
            max_full_entries + max_summaries
        ]

        # --------------------------------------------------------------
        # Profiles
        # --------------------------------------------------------------

        profiles = {}

        for query in queries:

            for person in self.extract_people_from_query(query):

                profile = self.load_profile(person)

                if profile is not None:
                    profiles[person] = profile

        return {
            "profiles": list(profiles.values()),
            "full_entries": full_entries,
            "summaries": summaries,
            "best_similarity": (
                results[0]["embedding_score"]
                if results
                else None
            ),
            "all_results": results,
        }

    # ------------------------------------------------------------------
    # PEOPLE / PROFILES
    # ------------------------------------------------------------------

    def extract_people_from_query(self, query):

        query_lower = query.lower()

        found = []

        for normalized_name, actual_name in self.people.items():

            if normalized_name in query_lower:
                found.append(actual_name)

        return found

    def load_profile(self, name):

        if name in self.profile_cache:
            return self.profile_cache[name]

        path = PROFILE_DIR / f"{name}.md"

        if not path.exists():
            return None

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        self.profile_cache[name] = {
            "name": name,
            "content": content,
        }

        return self.profile_cache[name]

    # ------------------------------------------------------------------
    # FORMATTING
    # ------------------------------------------------------------------

    def format_summary(self, memory):

        metadata = memory.get("metadata") or {}

        lines = []

        lines.append(
            f"Date: {memory.get('date', 'Unknown')}"
        )

        summary = metadata.get("summary")

        if summary:
            lines.append(
                f"Summary:\n{summary}"
            )

        for key in [
            "people",
            "places",
            "projects",
            "companies",
            "goals",
            "emotions",
            "events",
        ]:

            value = metadata.get(key)

            if not value:
                continue

            lines.append(
                f"{key.capitalize()}:\n{value}"
            )

        return "\n\n".join(lines)




if __name__ == "__main__":
    r = MemoryRetriever()

    while True:
        q = input("Query> ").strip()

        if not q:
            break

        res = r.retrieve_multiple([q])

        print("Profiles:", [
            profile["name"]
            for profile in res["profiles"]
        ])

        print("\n==============================")
        print("FULL ENTRIES")
        print("==============================")

        for entry in res["full_entries"]:

            print("=" * 80)
            print(entry["date"])
            print()
            print(entry["entry"])

            print(
                "Embedding similarity:",
                round(entry["embedding_score"], 3)
            )

            if entry.get("score") is not None:
                print(
                    "Rerank score:",
                    round(entry["score"], 3)
                )


        print("\n==============================")
        print("SUMMARIES")
        print("==============================")

        for memory in res["summaries"]:

            print("=" * 80)
            print(memory["date"])
            print()

            print(
                memory.get("summary")
                or memory.get("metadata", {}).get("summary", "")
            )

            print(
                "Embedding similarity:",
                round(memory["embedding_score"], 3)
            )

            if memory.get("score") is not None:
                print(
                    "Rerank score:",
                    round(memory["score"], 3)
                )