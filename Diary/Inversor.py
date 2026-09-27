from pathlib import Path
import re

INPUT_FILE = "LJDB_EN.md"
OUTPUT_FILE = "LJDB_EN_FIXED.md"

# Matches: mm/dd/yyyy:
DATE_PATTERN = re.compile(r"\b(\d{2})/(\d{2})/(\d{4})(\s*:)")


def swap_date(match):
    month = match.group(1)
    day = match.group(2)
    year = match.group(3)
    colon = match.group(4)

    return f"{day}/{month}/{year}{colon}"


def main():
    text = Path(INPUT_FILE).read_text(encoding="utf-8")

    fixed = DATE_PATTERN.sub(swap_date, text)

    Path(OUTPUT_FILE).write_text(fixed, encoding="utf-8")

    print("Done!")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()