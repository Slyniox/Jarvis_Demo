#!/usr/bin/env python3

import json
import sqlite3
from urllib import response
import requests
from sympy import python


class MetadataExtractor:

    def __init__(
        self,
        db_path="LJDB_FR.db",
        api_url="http://localhost:8081/v1/chat/completions",
        model="gpt-oss-20b"
    ):

        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row

        self.api_url = api_url
        self.model = model

        self.create_table()

    # --------------------------------------------------------

    def create_table(self):

        cursor = self.db.cursor()

        cursor.execute("""

        CREATE TABLE IF NOT EXISTS metadata(

            entry_id INTEGER PRIMARY KEY,

            json TEXT NOT NULL,

            processed_at TIMESTAMP
                DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY(entry_id)
                REFERENCES entries(id)

        )

        """)

        self.db.commit()

    # --------------------------------------------------------

    def get_unprocessed_entries(self):

        cursor = self.db.cursor()

        cursor.execute("""

        SELECT
            entries.id,
            entries.entry_date,
            entries.text

        FROM entries

        LEFT JOIN metadata

        ON entries.id = metadata.entry_id

        WHERE metadata.entry_id IS NULL

        ORDER BY entries.entry_date ASC

        """)

        return cursor.fetchall()

    # --------------------------------------------------------

    def ask_model(self, text):

        system_prompt = """Tu es un moteur d'extraction d'informations.

Extrais les informations structurées de l'entrée du journal.

Retourne UNIQUEMENT du JSON valide.

Le JSON DOIT contenir exactement les champs suivants :

{
"summary": "",
"importance": 1,
"people": [],
"places": [],
"projects": [],
"companies": [],
"goals": [],
"emotions": [],
"events": []
}

Règles :

N'invente jamais d'informations.
Les listes vides sont acceptées.
L'importance est mesurée selon la pertinence de cette entrée pour pour la mémoire à long terme de l'écrivain.
1 = Journée routinière, sans élément susceptible d'avoir de l'importance dans un an.
3 = Événements ou réflexions mineurs.
5 = Intéressant, mais sans changement majeur dans la vie de l'écrivain.
7 = Événement, réussite ou revers important.
9 = Étape majeure dans la vie de l'écrivain ou événement émotionnellement important.
10 = Événement exceptionnel qui restera presque certainement important pendant des années.
Ne produis pas de Markdown.
N'explique rien.
Retourne uniquement du JSON."""

        system_prompt_en = """
You are an information extraction engine.

Extract structured information from the diary entry.

Return ONLY valid JSON.

The JSON MUST have exactly these fields:

{
    "summary": "",
    "importance": 1,
    "people": [],
    "places": [],
    "projects": [],
    "companies": [],
    "goals": [],
    "emotions": [],
    "events": []
}

Rules:

- Never invent information.
- Empty lists are acceptable.
- Importance is measured by how relevant this entry is to the user's long-term memory.
 1 = Routine day with nothing likely to matter in a year.
 3 = Minor events or thoughts.
 5 = Interesting but not life-changing.
 7 = Significant event, achievement, or setback.
 9 = Major life milestone or emotionally important event.
 10 = Exceptional event that will almost certainly remain important for years.
- Do not output markdown.
- Do not think for more than 300 words.
- Do not explain.
- Output JSON only.
"""

        payload = {
            "model": self.model,
            "chat_template_kwargs": {
                "enable_thinking": False
            },
            "temperature": 0.2,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": text
                }
            ]
        }

        response = requests.post(
            self.api_url,
            json=payload,
            timeout=300
        )

        response.raise_for_status()

        content = response.json()["choices"][0]["message"]["content"]

        content = content.strip()

        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            content = content.rsplit("```", 1)[0]

        return json.loads(content)

    # --------------------------------------------------------

    def save_metadata(self, entry_id, metadata):

        cursor = self.db.cursor()

        cursor.execute("""

        INSERT OR REPLACE INTO metadata

        (
            entry_id,
            json
        )

        VALUES (?,?)

        """,
        (
            entry_id,
            json.dumps(metadata, ensure_ascii=False)
        ))

        self.db.commit()

    # --------------------------------------------------------

    def process_all(self):

        entries = self.get_unprocessed_entries()

        print(f"{len(entries)} entries remaining.\n")

        for entry in entries:

            print(
                f"[{entry['id']}] "
                f"{entry['entry_date']}"
            )

            try:

                metadata = self.ask_model(
                    entry["text"]
                )

                self.save_metadata(
                    entry["id"],
                    metadata
                )

                print("   ✓ Done")

            except Exception as e:

                print(f"   ✗ {e}")

    # --------------------------------------------------------

    def close(self):

        self.db.close()


if __name__ == "__main__":

    extractor = MetadataExtractor()

    extractor.process_all()

    extractor.close()