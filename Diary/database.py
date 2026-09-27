#!/usr/bin/env python3

import sqlite3
import hashlib
from pathlib import Path


class DiaryDatabase:

    def __init__(self, db_path="LJDB_FR.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.create_tables()

    # --------------------------------------------------

    def create_tables(self):

        cursor = self.conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS entries (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            entry_date TEXT NOT NULL,

            text TEXT NOT NULL,

            hash TEXT NOT NULL UNIQUE,

            word_count INTEGER,

            character_count INTEGER,

            imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        self.conn.commit()

    # --------------------------------------------------

    @staticmethod
    def compute_hash(text: str):

        return hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()

    # --------------------------------------------------

    def insert_entry(self, entry):

        cursor = self.conn.cursor()

        hash_value = self.compute_hash(entry.text)

        cursor.execute("""
            INSERT OR IGNORE INTO entries
            (
                entry_date,
                text,
                hash,
                word_count,
                character_count
            )
            VALUES (?, ?, ?, ?, ?)
        """,
        (
            entry.date.isoformat(),
            entry.text,
            hash_value,
            entry.word_count,
            entry.character_count
        ))

        self.conn.commit()

    # --------------------------------------------------

    def insert_entries(self, entries):

        inserted = 0

        for entry in entries:

            before = self.conn.total_changes

            self.insert_entry(entry)

            if self.conn.total_changes > before:
                inserted += 1

        print(f"Inserted {inserted} new entries.")

    # --------------------------------------------------

    def number_of_entries(self):

        cursor = self.conn.cursor()

        cursor.execute(
            "SELECT COUNT(*) FROM entries"
        )

        return cursor.fetchone()[0]

    # --------------------------------------------------

    def get_entry(self, entry_id):

        cursor = self.conn.cursor()

        cursor.execute("""

            SELECT *

            FROM entries

            WHERE id=?

        """, (entry_id,))

        return cursor.fetchone()

    # --------------------------------------------------

    def close(self):

        self.conn.close()