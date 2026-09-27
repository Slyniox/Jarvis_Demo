#!/usr/bin/env python3

import re

INPUT_FILE = "LJDB_EN_FIXED.md"
OUTPUT_FILE = "LJDB_no_summary.md"

DATE_PATTERN = re.compile(
    r"(\d{2}/\d{2}/\d{4}\s*:)"
)

# First sentence ends with . ! or ?
FIRST_SENTENCE = re.compile(
    r"^\s*(.+?[.!?])(\s+|$)",
    re.DOTALL
)


with open(INPUT_FILE, "r", encoding="utf-8") as f:
    text = f.read()

parts = DATE_PATTERN.split(text)

result = parts[0]

for i in range(1, len(parts), 2):

    date = parts[i]
    body = parts[i + 1]

    match = FIRST_SENTENCE.search(body)

    if match:
        body = body[match.end():]

    result += date + body

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(result)

print("Finished.")