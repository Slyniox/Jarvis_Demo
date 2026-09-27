#!/usr/bin/env python3

import json
from itertools import combinations
from difflib import SequenceMatcher
from pathlib import Path


INPUT_FILE = "memory_candidates.json"
OUTPUT_FILE = "merge_candidates.json"

CATEGORIES = [
    "people",
    "projects",
    "companies",
]


# ------------------------------------------------------------

def string_similarity(a, b):

    return SequenceMatcher(
        None,
        a.lower(),
        b.lower()
    ).ratio()


# ------------------------------------------------------------

def overlap_score(list1, list2):

    s1 = set(list1)
    s2 = set(list2)

    if not s1 or not s2:
        return 0.0

    return len(s1 & s2) / min(len(s1), len(s2))


# ------------------------------------------------------------

def year(date):

    return int(date[:4])


# ------------------------------------------------------------

def time_similarity(a, b):

    first_diff = abs(
        year(a["first_date"]) -
        year(b["first_date"])
    )

    last_diff = abs(
        year(a["last_date"]) -
        year(b["last_date"])
    )

    score = 1 - min(
        (first_diff + last_diff) / 10,
        1
    )

    return score


# ------------------------------------------------------------

def overall_score(a, b):

    s = string_similarity(
        a["name"],
        b["name"]
    )

    o = overlap_score(
        a["entry_ids"],
        b["entry_ids"]
    )

    t = time_similarity(
        a,
        b
    )

    return {

        "string": s,
        "overlap": o,
        "time": t,

        "overall":
            0.50*s +
            0.30*o +
            0.20*t
    }


# ------------------------------------------------------------

def process_category(category, items):

    candidates = []

    for a, b in combinations(items, 2):

        score = overall_score(a, b)

        if score["overall"] < 0.60:
            continue

        candidates.append({

            "entity_a": a["name"],

            "entity_b": b["name"],

            "score": round(
                score["overall"],
                3
            ),

            "string_similarity":
                round(score["string"], 3),

            "entry_overlap":
                round(score["overlap"], 3),

            "time_similarity":
                round(score["time"], 3),

            "mentions":

            {
                "a": a["mentions"],
                "b": b["mentions"]
            }

        })

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return candidates


# ------------------------------------------------------------

def main():

    data = json.loads(
        Path(INPUT_FILE).read_text(
            encoding="utf-8"
        )
    )

    output = {}

    for category in CATEGORIES:

        print(
            f"Processing {category}..."
        )

        output[category] = process_category(
            category,
            data[category]
        )

    Path(OUTPUT_FILE).write_text(

        json.dumps(
            output,
            indent=4,
            ensure_ascii=False
        ),

        encoding="utf-8"

    )

    print()
    print(
        f"Written {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()