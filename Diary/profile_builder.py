#!/usr/bin/env python3

import json
import re
import sqlite3
from pathlib import Path

import requests

DB_PATH = "LJDB_FR.db"
MEMORY_FILE = "memory_candidates.json"

API_URL = "http://localhost:8081/v1/chat/completions"
MODEL = "gpt-oss-20b"

MIN_MENTIONS = 10
CHUNK_SIZE = 15

FRAGMENT_DIR = "profile_fragments"

PROFILE_FRAGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string"
        },
        "important_events": {
            "type": "array",
            "items": {
                "type": "string"
            }
        }
    },
    "required": [
        "summary",
        "important_events"
    ],
    "additionalProperties": False
}


class ProfileBuilder:

    def __init__(self):

        self.db = sqlite3.connect(DB_PATH)
        self.db.row_factory = sqlite3.Row

        self.fragment_dir = Path(FRAGMENT_DIR)
        self.fragment_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------

    def get_fragment_path(
        self,
        category,
        entity,
        chunk_index,
        total_chunks,
        chunk
    ):

        directory = self.fragment_dir / category
        directory.mkdir(exist_ok=True)

        safe_name = self.sanitize_filename(entity)

        start = chunk[0]["entry_date"][:10]
        end = chunk[-1]["entry_date"][:10]

        return (
            directory /
            f"{safe_name}_{chunk_index:03d}-of-{total_chunks:03d}_{start}_to_{end}.json"
        )


    def load_candidates(self):

        with open(MEMORY_FILE, encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------

    def load_entries(self, ids):

        if not ids:
            return []

        cursor = self.db.cursor()

        placeholders = ",".join("?" * len(ids))

        cursor.execute(f"""
            SELECT
                id,
                entry_date,
                text
            FROM entries
            WHERE id IN ({placeholders})
            ORDER BY entry_date
        """, ids)

        return cursor.fetchall()

    # ------------------------------------------------------

    def chunk(self, entries):

        for i in range(0, len(entries), CHUNK_SIZE):
            yield entries[i:i + CHUNK_SIZE]

    # ------------------------------------------------------

    def sanitize_filename(self, text):

        text = re.sub(r'[<>:"/\\\\|?*]', "_", text)
        text = re.sub(r"\s+", "_", text)
        return text[:120]

    # ------------------------------------------------------

    def build_prompt(
        self,
        category,
        name,
        chunk,
        chunk_index,
        total_chunks
    ):

        diary = ""

        for row in chunk:

            diary += (
                f"\n===== {row['entry_date']} =====\n"
                f"{row['text']}\n"
            )

        system = f"""

Construis un fragment de mémoire à propos de {name}.

Des entrées de journal intime mentionnant {name} te sont fournies.

Remplis le schéma JSON fourni par l'API.

Règles :

Le résumé doit contenir 5 à 6 phrases et décrire ce qui se passe entre le narrateur et {name}.
Ignore les événements du journal qui n'impliquent pas {name}.
La liste important_events doit contenir uniquement les événements importants impliquant {name}.
N'invente jamais d'informations.

"""

        return system, diary

    # ------------------------------------------------------

    def ask_model(self, system, user):

        payload = {
            "model": MODEL,
            "temperature": 0.4,
            "chat_template_kwargs": {
                "enable_thinking": False
            },
            "messages": [
                {
                    "role": "system",
                    "content": system
                },
                {
                    "role": "user",
                    "content": user
                }
            ],
            "response_format": {
                "type": "json_object",
                "schema": PROFILE_FRAGMENT_SCHEMA
            }
        }

        response = requests.post(
            API_URL,
            json=payload,
            timeout=600
        )

        response.raise_for_status()

        text = response.json()["choices"][0]["message"]["content"]

        return json.loads(text)

    # ------------------------------------------------------

    def save_fragment(
        self,
        category,
        entity,
        chunk_index,
        total_chunks,
        chunk,
        fragment
    ):

        directory = self.fragment_dir / category
        directory.mkdir(exist_ok=True)

        safe_name = self.sanitize_filename(entity)

        start = chunk[0]["entry_date"][:10]
        end = chunk[-1]["entry_date"][:10]

        filename = self.get_fragment_path(
            category,
            entity,
            chunk_index,
            total_chunks,
            chunk
        )

        payload = {
            "entity": entity,
            "category": category,
            "chunk": chunk_index,
            "total_chunks": total_chunks,
            "start_date": start,
            "end_date": end,
            "entry_ids": [r["id"] for r in chunk],
            "fragment": fragment
        }

        filename.write_text(
            json.dumps(
                payload,
                indent=4,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

    # ------------------------------------------------------

    def process(self):

        candidates = self.load_candidates()

        for category, entities in candidates.items():

            print(f"\n===== {category.upper()} =====")

            for entity in entities:

                if entity["mentions"] < MIN_MENTIONS:
                    continue

                print(
                    f"\nBuilding fragments for "
                    f"{entity['name']} "
                    f"({entity['mentions']} mentions)"
                )

                rows = self.load_entries(
                    entity["entry_ids"]
                )

                chunks = list(self.chunk(rows))

                for index, chunk in enumerate(chunks, start=1):

                    print(
                        f"  Chunk {index}/{len(chunks)}"
                    )

                    try:
                        filename = self.get_fragment_path(
                            category,
                            entity["name"],
                            index,
                            len(chunks),
                            chunk
                        )

                        if filename.exists():

                            print("     ✓ Already exists")

                            continue

                        system, diary = self.build_prompt(
                            category,
                            entity["name"],
                            chunk,
                            index,
                            len(chunks)
                        )

                        fragment = self.ask_model(
                            system,
                            diary
                        )

                        self.save_fragment(
                            category,
                            entity["name"],
                            index,
                            len(chunks),
                            chunk,
                            fragment
                        )

                        print("     ✓ Saved")

                    except Exception as e:

                        print(f"     ✗ {e}")

    # ------------------------------------------------------

    def close(self):

        self.db.close()


if __name__ == "__main__":

    builder = ProfileBuilder()

    builder.process()

    builder.close()