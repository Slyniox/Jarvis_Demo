#!/usr/bin/env python3

"""
Diary Parser

Parses a markdown diary where every entry starts with:

DD/MM/YYYY :

Example:

01/01/2025 :

Today was a good day.

02/01/2025 :

Worked on my AI project.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List
import re


# ==========================================================
# Configuration
# ==========================================================

DATE_REGEX = re.compile(
    r"^(\d{2}/\d{2}/\d{4})\s*:\s*$",
    re.MULTILINE
)


# ==========================================================
# Data Structure
# ==========================================================

@dataclass
class DiaryEntry:
    id: int
    date: datetime
    text: str
    word_count: int
    character_count: int
    def to_dict(self):
        return {
            "id": self.id,
            "date": self.date.isoformat(),
            "text": self.text,
            "word_count": self.word_count,
            "character_count": self.character_count,
    	}


# ==========================================================
# Parser
# ==========================================================

class DiaryParser:

    def __init__(self, filename: str):
        self.filename = Path(filename)

        if not self.filename.exists():
            raise FileNotFoundError(
                f"Diary file not found: {self.filename}"
            )

    def parse(self) -> List[DiaryEntry]:

        content = self.filename.read_text(
            encoding="utf-8"
        )

        matches = list(DATE_REGEX.finditer(content))

        if len(matches) == 0:
            raise RuntimeError(
                "No diary entries were found.\n"
                "Expected lines formatted as:\n"
                "DD/MM/YYYY :"
            )

        entries = []

        previous_date = None

        for i, match in enumerate(matches):

            start = match.end()

            end = (
                matches[i + 1].start()
                if i + 1 < len(matches)
                else len(content)
            )

            raw_date = match.group(1)

            date = datetime.strptime(
                raw_date,
                "%d/%m/%Y"
            )


            text = content[start:end]
            # Normalize line endings
            text = text.replace("\r\n", "\n")

            # Remove separator lines made of underscores
            text = re.sub(
                r"^\s*[\\_]+\s*$",
                "",
                text,
                flags=re.MULTILINE
            )

            # Collapse excessive blank lines
            text = re.sub(r"\n{3,}", "\n\n", text)

            text = text.strip()

			

            if previous_date is not None:
                if date < previous_date:
                    print(
                        f"Warning: entry {i} "
                        f"is earlier than previous entry."
                    )

            previous_date = date

            entries.append(
                DiaryEntry(
                    id=i,
                    date=date,
                    text=text,
                    word_count=len(text.split()),
                    character_count=len(text)
                )
            )

        return entries


# ==========================================================
# Statistics
# ==========================================================

def print_statistics(entries: List[DiaryEntry]):

    print("\n========== Diary Statistics ==========\n")

    print(f"Entries          : {len(entries)}")

    print(
        f"First Entry      : "
        f"{entries[0].date.strftime('%d/%m/%Y')}"
    )

    print(
        f"Last Entry       : "
        f"{entries[-1].date.strftime('%d/%m/%Y')}"
    )

    total_words = sum(e.word_count for e in entries)

    print(f"Total Words      : {total_words:,}")

    average = total_words / len(entries)

    print(f"Average Words    : {average:.1f}")

    longest = max(entries, key=lambda e: e.word_count)

    print(
        f"Longest Entry    : "
        f"{longest.word_count} words "
        f"({longest.date.strftime('%d/%m/%Y')})"
    )

    shortest = min(entries, key=lambda e: e.word_count)

    print(
        f"Shortest Entry   : "
        f"{shortest.word_count} words "
        f"({shortest.date.strftime('%d/%m/%Y')})"
    )


# ==========================================================
# Main
# ==========================================================

def main():

    parser = DiaryParser("LJDB_EN_FIXED.md")

    entries = parser.parse()

    print_statistics(entries)

    print("\n========== First Entry ==========\n")

    first = entries[0]

    print(first.date.strftime("%d/%m/%Y"))
    print()
    print(first.text[:500])

    if len(first.text) > 500:
        print("\n...")


if __name__ == "__main__":
    main()