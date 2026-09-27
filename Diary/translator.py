from openai import OpenAI
from pathlib import Path
import re
import time

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

API_BASE = "http://localhost:8081/v1"      # Your proxy endpoint
API_KEY = "not-needed"

MODEL = "translategemma"

INPUT_FILE = "LJDB.md"
OUTPUT_FILE = "LJDB_EN.md"

# ------------------------------------------------------------------

client = OpenAI(
    base_url=API_BASE,
    api_key=API_KEY,
)

# Split on markdown H1 dates:
# # 2026-07-30
ENTRY_REGEX = r"(?=^\d{2}/\d{2}/\d{4}\s*:)"


def split_entries(text: str):
    entries = re.split(ENTRY_REGEX, text, flags=re.MULTILINE)
    return [e.strip() for e in entries if e.strip()]


def translate(entry: str) -> str:

    prompt = f"""<start_of_turn>user
type:text,source_lang_code:fr,target_lang_code:en,text:
{entry}

<end_of_turn>
<start_of_turn>model
type:text,source_lang_code:fr,target_lang_code:en,text:
"""

    import requests

    response = requests.post(
        "http://localhost:8081/completion",
        json={
            "prompt": prompt,
            "temperature": 0,
            "n_predict": 2048,
            "stop": [
                    "____________________",
                    "<end_of_turn>"
            ]
        }
    )
    response.raise_for_status()

    data = response.json()

    return data["content"].strip()


def main():

    text = Path(INPUT_FILE).read_text(encoding="utf-8")

    entries = re.split(ENTRY_REGEX, text, flags=re.MULTILINE)

    print(f"Found {len(entries)} entries")

    Path(OUTPUT_FILE).write_text("", encoding="utf-8")

    for i, entry in enumerate(entries, 1):

        print(f"[{i}/{len(entries)}] Translating...")

        translated = translate(entry)

        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(translated)
            f.write("\n\n")

        # Prevent hammering the server
        time.sleep(0.2)

    print("Done.")


if __name__ == "__main__":
    main()