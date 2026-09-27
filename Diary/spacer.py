from pathlib import Path
import re

INPUT_FILE = "LJDB_EN_FIXED.md"
OUTPUT_FILE = "LJDB_EN_FIXED.md"   # Overwrite the file

text = Path(INPUT_FILE).read_text(encoding="utf-8")

# Convert:
# 31/07/2026:
# into:
# 31/07/2026 :
text = re.sub(
    r"(\b\d{2}/\d{2}/\d{4}):",
    r"\1 :",
    text
)

Path(OUTPUT_FILE).write_text(text, encoding="utf-8")

print("Done!")