# memory_reranker.py

import torch
import time

from init_reranker import model, tokenizer


###############################################################################
# CONFIG
###############################################################################

DEVICE = "cuda"
MAX_LENGTH = 4096
BATCH_SIZE = 1


###############################################################################
# RERANK
###############################################################################

def rerank_memory(query, retrieval, top_k=12):

    full_entries = retrieval.get("full_entries", [])

    if not full_entries:
        return retrieval

    candidates = []

    for entry in full_entries:

        candidates.append({
            "type": "entry",
            "object": entry,
            "text": entry["entry"],
        })

    ###########################################################################
    # Build query/document pairs
    ###########################################################################

    pairs = [
        (query, candidate["text"])
        for candidate in candidates
    ]

    ###########################################################################
    # Reranking inference
    ###########################################################################

    scores = []

    t0 = time.perf_counter()

    for i in range(0, len(pairs), BATCH_SIZE):

        batch = pairs[i:i + BATCH_SIZE]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        inputs = {
            k: v.to(DEVICE)
            for k, v in inputs.items()
        }

        with torch.inference_mode():

            outputs = model(**inputs)

        batch_scores = outputs.logits.squeeze(-1)

        # squeeze(-1) becomes a scalar when BATCH_SIZE == 1
        if batch_scores.ndim == 0:
            batch_scores = batch_scores.unsqueeze(0)

        scores.extend(
            batch_scores.cpu().tolist()
        )

    ###########################################################################
    # Timing
    ###########################################################################

    t1 = time.perf_counter()

    print(
        f"Reranker inference: {t1 - t0:.2f}s"
    )

    ###########################################################################
    # Attach scores
    ###########################################################################
    for candidate, score in zip(candidates, scores):

        candidate["score"] = float(score)
        candidate["object"]["score"] = float(score)

    ###########################################################################
    # Sort
    ###########################################################################

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    ###########################################################################
    # Keep top K
    ###########################################################################

    candidates = candidates[:top_k]

    ###########################################################################
    # Rebuild retrieval
    #
    # Keep the original entry objects so the rest of Jarvis can still access:
    #
    #     entry["id"]
    #     entry["date"]
    #     entry["entry"]
    #
    ###########################################################################

    retrieval["full_entries"] = [
        candidate["object"]
        for candidate in candidates
    ]

    return retrieval