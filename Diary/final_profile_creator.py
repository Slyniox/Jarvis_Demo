#!/usr/bin/env python3
"""
final_profile_creator.py

Builds one long-term memory profile per person in ONE model call.

The people and their relevant diary entries are obtained from:

    memory_candidates.json

The actual diary entries are loaded from:

    LJDB_FR.db

Expected memory_candidates.json structure:

{
    "people": [
        {
            "name": "Adam",
            "mentions": 42,
            "first_date": "...",
            "last_date": "...",
            "entry_ids": [1, 23, 45, ...]
        }
    ]
}

For each person, ALL associated diary entries are passed to the
model at once. No intermediate semantic fragments are created.
"""

from pathlib import Path
import json
import sqlite3
import requests
import time


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

DB_PATH = Path("LJDB_FR.db")
MEMORY_FILE = Path("memory_candidates.json")
OUTPUT_DIR = Path("profiles")

LLAMA_ENDPOINT = "http://127.0.0.1:8081/v1/chat/completions"

MODEL = "local"

TEMPERATURE = 0.6
MAX_TOKENS = 30000
RETRIES = 3


# ---------------------------------------------------------------------
# PROMPT
# ---------------------------------------------------------------------

PROMPT = r"""
Tu reconstruis la mémoire autobiographique à long terme de l'auteur.

Les éléments suivants sont des entrées complètes de son journal intime
dans lesquelles cette personne est mentionnée.

Considère les informations présentes dans les entrées comme fiables.

Ta tâche n'est PAS de résumer les entrées individuellement.

Tu dois plutôt reconstruire la manière dont l'auteur se souvient de cette
personne à travers l'ensemble de son journal.

Tu dois identifier les éléments récurrents, l'évolution de la relation,
les événements importants et les informations permettant de comprendre
qui est cette personne pour l'auteur.

Utilise EXACTEMENT les sections suivantes.

# Résumé de la relation

Écris entre 200 et 300 mots en décrivant :

- qui est cette personne pour l'auteur
- les principaux thèmes récurrents de la relation
- l'évolution de la relation lorsqu'elle est explicitement visible
- la relation actuelle

Adresse le narrateur à la SECONDE PERSONNE.

Reste NEUTRE ET FACTUEL.

N'interprète pas les sentiments, intentions ou motivations à moins qu'ils
ne soient explicitement mentionnés dans les entrées.

# Personnalité

Écris entre 50 et 150 mots en décrivant :

- la personnalité
- le style de communication
- les centres d'intérêt
- les comportements récurrents

N'infère que des traits qui sont étayés de manière répétée dans les entrées.

# Souvenirs communs les plus importants

Sélectionne entre CINQ et DIX souvenirs particulièrement importants.

Pour chacun :

## Titre

Un paragraphe décrivant ce qui s'est passé et pourquoi ce souvenir
fait partie des éléments importants de la mémoire de l'auteur concernant
cette personne.

Fusionne les souvenirs redondants.

Ne mentionne pas les entrées du journal, les identifiants d'entrées,
les numéros d'entrée ou le processus de reconstruction.

PERSONNE

{name}

====================================================

ENTRÉES DU JOURNAL

{entries}

"""


# ---------------------------------------------------------------------
# DATABASE
# ---------------------------------------------------------------------

def load_entries(entry_ids):

    if not entry_ids:
        return []

    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row

    try:

        placeholders = ",".join(
            "?" for _ in entry_ids
        )

        cursor = db.execute(
            f"""
            SELECT
                id,
                entry_date,
                text
            FROM entries
            WHERE id IN ({placeholders})
            """,
            entry_ids
        )

        rows = cursor.fetchall()

    finally:

        db.close()

    # Restore the order from entry_ids.
    by_id = {
        row["id"]: row
        for row in rows
    }

    entries = []

    for entry_id in entry_ids:

        row = by_id.get(entry_id)

        if row is None:
            continue

        entries.append({
            "id": row["id"],
            "date": row["entry_date"],
            "text": row["text"],
        })

    # Make sure the entries are chronologically ordered.
    entries.sort(
        key=lambda x: x["date"]
    )

    return entries


# ---------------------------------------------------------------------
# LOAD PEOPLE
# ---------------------------------------------------------------------

def load_people():

    if not MEMORY_FILE.exists():
        raise FileNotFoundError(MEMORY_FILE)

    with open(
        MEMORY_FILE,
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    return data.get("people", [])


# ---------------------------------------------------------------------
# BUILD ENTRY TEXT
# ---------------------------------------------------------------------

def build_entry_text(entries):

    out = []

    for entry in entries:

        txt = []

        txt.append("=" * 70)
        txt.append(
            f"Date: {entry['date']}"
        )
        txt.append("")
        txt.append(
            entry["text"]
        )

        out.append(
            "\n".join(txt)
        )

    return "\n\n".join(out)


# ---------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------

def call_model(prompt):

    body = {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,

        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    for attempt in range(RETRIES):

        try:

            r = requests.post(
                LLAMA_ENDPOINT,
                json=body,
                timeout=6000
            )

            r.raise_for_status()

            data = r.json()

            return data["choices"][0]["message"]["content"]

        except Exception as e:

            print(
                f"  Model call failed "
                f"(attempt {attempt + 1}/{RETRIES}): {e}"
            )

            if attempt == RETRIES - 1:
                raise

            time.sleep(5)


# ---------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------

def save_profile(person, text):

    OUTPUT_DIR.mkdir(
        exist_ok=True
    )

    filename = OUTPUT_DIR / f"{person}.md"

    filename.write_text(
        text,
        encoding="utf-8"
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    if not DB_PATH.exists():
        raise FileNotFoundError(DB_PATH)

    people = load_people()

    print(
        f"Found {len(people)} people.\n"
    )

    total = len(people)

    for i, person in enumerate(
        people,
        start=1
    ):

        name = person["name"]

        output_file = (
            OUTPUT_DIR / f"{name}.md"
        )

        if output_file.exists():

            print(
                f"[{i}/{total}] "
                f"Skipping {name} "
                f"(already exists)"
            )

            continue

        entry_ids = person.get(
            "entry_ids",
            []
        )

        print(
            f"[{i}/{total}] "
            f"Building {name}"
        )

        print(
            f"  Mentions: "
            f"{person.get('mentions', 0)}"
        )

        print(
            f"  Entries: "
            f"{len(entry_ids)}"
        )

        if not entry_ids:

            print(
                "  No entries found, skipping.\n"
            )

            continue

        # -------------------------------------------------------------
        # Load every complete diary entry associated with this person.
        # -------------------------------------------------------------

        entries = load_entries(
            entry_ids
        )

        if not entries:

            print(
                "  No matching database entries, skipping.\n"
            )

            continue

        print(
            f"  Loaded {len(entries)} "
            f"complete diary entries."
        )

        # -------------------------------------------------------------
        # Build ONE large prompt.
        # -------------------------------------------------------------

        entry_text = build_entry_text(
            entries
        )

        prompt = PROMPT.format(
            name=name,
            entries=entry_text
        )

        print(
            f"  Prompt size: "
            f"{len(prompt):,} characters"
        )

        # -------------------------------------------------------------
        # ONE model call.
        # -------------------------------------------------------------

        profile = call_model(
            prompt
        )

        save_profile(
            name,
            profile
        )

        print(
            "  ✓ done\n"
        )

    print("Finished.")


# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()
