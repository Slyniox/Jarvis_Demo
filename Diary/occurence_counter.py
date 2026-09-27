#!/usr/bin/env python3

import json
import sqlite3
from collections import defaultdict
from pathlib import Path


DB_PATH = "LJDB_FR.db"


CATEGORIES = [
    "people",
    "projects",
    "companies",
    "places",
    "goals",
    "events",
    "emotions"
]


class MemoryDetector:

    def __init__(self, db_path):

        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row

        self.memory = {
            category: defaultdict(lambda: {
                "count": 0,
                "first_date": None,
                "last_date": None,
                "entry_ids": []
            })
            for category in CATEGORIES
        }

    # ----------------------------------------------------------

    def load_entries(self):

        cursor = self.db.cursor()

        cursor.execute("""

        SELECT
            entries.id,
            entries.entry_date,
            metadata.json

        FROM entries

        JOIN metadata

        ON entries.id = metadata.entry_id

        ORDER BY entries.entry_date

        """)

        return cursor.fetchall()

    # ----------------------------------------------------------

    def process(self):

        rows = self.load_entries()

        for row in rows:

            metadata = json.loads(row["json"])

            entry_id = row["id"]
            date = row["entry_date"]

            for category in CATEGORIES:

                if category not in metadata:
                    continue

                for value in metadata[category]:

                    if not value:
                        continue

                    # Handle both strings and dictionaries

                    if isinstance(value, dict):
                        name = value.get("name", "").strip()
                    else:
                        name = str(value).strip()

                    if not name:
                        continue

                    item = self.memory[category][name]

                    item["count"] += 1

                    item["entry_ids"].append(entry_id)

                    if item["first_date"] is None:
                        item["first_date"] = date

                    item["last_date"] = date

    # ----------------------------------------------------------

    def export(self, filename="memory_candidates.json"):

        output = {}

        for category in CATEGORIES:

            output[category] = []

            for name, data in sorted(
                self.memory[category].items(),
                key=lambda x: x[1]["count"],
                reverse=True
            ):

                output[category].append({

                    "name": name,

                    "mentions": data["count"],

                    "first_date": data["first_date"],

                    "last_date": data["last_date"],

                    "entry_ids": data["entry_ids"]

                })

        with open(filename, "w", encoding="utf-8") as f:

            json.dump(
                output,
                f,
                indent=4,
                ensure_ascii=False
            )

        print(f"\nExported to {filename}")

    # ----------------------------------------------------------

    def print_summary(self):

        print()

        for category in CATEGORIES:

            print("=" * 60)
            print(category.upper())
            print("=" * 60)

            top = sorted(
                self.memory[category].items(),
                key=lambda x: x[1]["count"],
                reverse=True
            )[:5]

            for name, data in top:

                print(
                    f"{name:35}"
                    f"{data['count']:5} mentions"
                )

            print()

    # ----------------------------------------------------------

    def close(self):

        self.db.close()


def main():

    detector = MemoryDetector(DB_PATH)

    detector.process()

    detector.print_summary()

    detector.export()

    detector.close()


if __name__ == "__main__":
    main()