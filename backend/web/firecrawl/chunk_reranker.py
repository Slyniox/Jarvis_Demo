#!/usr/bin/env python3

"""
reranker.py

Web reranker for Jarvis.

Input:
    query (str)
    pages (list[str])

Output:
    Top-k reranked pages.
"""

#!/usr/bin/env python3

"""
reranker.py

Web reranker for Jarvis.

Input:
    query (str)
    pages (list[str])

Output:
    Top-k reranked pages.
"""

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3] / "Diary"))

from init_reranker import model, tokenizer


###############################################################################

DEVICE = "cuda"

MAX_LENGTH = 400

BATCH_SIZE = 1

MAX_CHUNKS = 4

MIN_SCORE = 1.3

###############################################################################


def rerank(query, documents, top_k=MAX_CHUNKS, min_score=MIN_SCORE):

    if not documents:
        return []

    ###########################################################################

    pairs = [
        (query, doc["text"])
        for doc in documents
    ]

    scores = []

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

        scores.extend(batch_scores.cpu().tolist())

    ###########################################################################

    for doc, score in zip(documents, scores):
        doc["score"] = float(score)

    documents.sort(
        key=lambda doc: doc["score"],
        reverse=True,
    )

    ###########################################################################

    relevant = [
        doc for doc in documents
        if doc["score"] >= min_score
    ]

    return relevant[:top_k]


# main.py
# Script de test du reranker : on fournit une requête au terminal,
# le modèle évalue 8 fiches pays et affiche la plus pertinente.

# ─── Corpus de test : 8 petites fiches sur des pays ────────────────────────────
# Chaque document est un dict avec au minimum une clé "text".
DOCUMENTS = [
    {
        "text": (
            "Le Japon est un archipel de l'océan Pacifique composé de "
            "6 852 îles. Sa capitale est Tokyo. Le pays est réputé pour "
            "sa cuisine (sushi, ramen), ses temples shintoïstes et son "
            "industrie technologique de pointe."
        )
    },
    {
        "text": (
            "Le Brésil est le plus grand pays d'Amérique du Sud. Il abrite "
            "la majeure partie de la forêt amazonienne et possède une "
            "biodiversité exceptionnelle. Sa capitale est Brasília et sa "
            "langue officielle est le portugais."
        )
    },
    {
        "text": (
            "L'Islande est une île volcanique située dans l'Atlantique Nord. "
            "Elle est connue pour ses geysers, ses glaciers, ses aurores "
            "boréales et ses sources d'eau chaude naturelles. Sa capitale "
            "est Reykjavik."
        )
    },
    {
        "text": (
            "L'Égypte, située au nord-est de l'Afrique, est traversée par "
            "le Nil. Elle abrite les pyramides de Gizeh, le Sphinx et de "
            "nombreux temples pharaoniques. Sa capitale est Le Caire."
        )
    },
    {
        "text": (
            "Le Canada est le deuxième plus grand pays au monde par sa "
            "superficie. Il est officiellement bilingue (français et "
            "anglais). On y trouve les Rocheuses, de vastes forêts "
            "boréales et de nombreux lacs. Sa capitale est Ottawa."
        )
    },
    {
        "text": (
            "La Norvège est un pays scandinave célèbre pour ses fjords, "
            "ses montagnes et son climat rigoureux. Elle est l'un des "
            "premiers producteurs de pétrole en Europe et sa capitale "
            "est Oslo."
        )
    },
    {
        "text": (
            "Le Pérou, en Amérique du Sud, est le berceau de l'Empire "
            "inca. Le Machu Picchu, perché dans les Andes, est son site "
            "archéologique le plus emblématique. Sa capitale est Lima et "
            "sa gastronomie est reconnue mondialement."
        )
    },
    {
        "text": (
            "L'Australie est à la fois un pays et un continent. Elle est "
            "connue pour la Grande Barrière de corail, l'opéra de Sydney, "
            "ses marsupiaux (kangourous, koalas) et son vaste outback "
            "désertique. Sa capitale est Canberra."
        )
    },
]


def main():
    print("=" * 60)
    print("  TEST DU RERANKER — 8 fiches pays")
    print("=" * 60)
    print("Tapez votre requête puis Entrée.")
    print("(tapez 'quit' ou laissez vide pour quitter)\n")

    while True:
        # ── Lecture de la requête utilisateur ───────────────────────────────
        query = input("Requête > ").strip()

        if not query or query.lower() == "quit":
            print("Au revoir !")
            break

        # ── Appel au reranker ───────────────────────────────────────────────
        # top_k=1 : on ne veut que LE document le plus pertinent.
        resultats = rerank(query, DOCUMENTS, top_k=1)

        # ── Affichage du résultat ───────────────────────────────────────────
        if resultats:
            meilleur = resultats[0]
            print(f"\n  📄 Document le plus pertinent (score : {meilleur['score']:.4f})")
            print(f"  {'─' * 50}")
            print(f"  {meilleur['text']}\n")
        else:
            print("\n  Aucun document trouvé.\n")


if __name__ == "__main__":
    main()
