import re
import tiktoken
from datetime import datetime
from API import scrape_query

# -----------------------------------------------------------------------------
# Tokenizer
# -----------------------------------------------------------------------------

tokenizer = tiktoken.get_encoding("cl100k_base")

# -----------------------------------------------------------------------------
# Keywords commonly found in navigation/footer sections
# -----------------------------------------------------------------------------

LOWER_GIBERISH_BOUND = 0.57
HIGHER_GIBERISH_BOUND = 0.90
MAX_CHUNKS = 32

NOISE_KEYWORDS = {
    "home",
    "menu",
    "navigation",
    "search",
    "login",
    "sign in",
    "sign up",
    "register",
    "privacy",
    "privacy policy",
    "cookie",
    "cookies",
    "terms",
    "terms of service",
    "copyright",
    "all rights reserved",
    "newsletter",
    "subscribe",
    "contact us",
    "follow us",
    "facebook",
    "twitter",
    "linkedin",
    "instagram",
    "youtube",
    "reddit",
    "back to top",
    "related articles",
    "recommended",
    "advertisement",
}

def looks_like_gibberish(text: str) -> bool:
    """
    Detect SVG, CSS, URL-encoded strings, base64, etc.
    """

    lower = text.lower()

    # ------------------------------------------------------------------
    # URL-encoded content
    # ------------------------------------------------------------------

    if lower.count("%20") > 20:
        return True

    if lower.count("%3c") > 10:
        return True

    # ------------------------------------------------------------------
    # Too many symbols
    # ------------------------------------------------------------------

    letters = sum(c.isalpha() for c in text)

    symbols = sum(
        not c.isalnum() and not c.isspace()
        for c in text
    )

    if letters > 0 and symbols / letters > 0.8:
        return True

    # ------------------------------------------------------------------
    # Long words (base64 / encoded)
    # ------------------------------------------------------------------

    long_words = re.findall(r"\S{60,}", text)

    if len(long_words) > 3:
        return True

    return False


# -----------------------------------------------------------------------------
# Chunking
# -----------------------------------------------------------------------------

def split_markdown(text, max_tokens=400, overlap=100):

    tokens = tokenizer.encode(text)

    chunks = []

    start = 0

    while start < len(tokens):

        end = min(start + max_tokens, len(tokens))
        chunks.append(tokenizer.decode(tokens[start:end]))

        if end == len(tokens):
            break

        start += max_tokens - overlap

    return chunks

# -----------------------------------------------------------------------------
# Noise detection
# -----------------------------------------------------------------------------

def is_noise_chunk(chunk,
                   min_chars=200,
                   keyword_threshold=3):
    """
    Returns True if the chunk looks like navigation,
    menus, metadata or a footer.
    """

    text = chunk.lower()

    alpha = sum(c.isalpha() for c in text)
    total = len(text)

    ratio = alpha / total

    if (ratio < LOWER_GIBERISH_BOUND or ratio > HIGHER_GIBERISH_BOUND):
        return True
    # ---------------------------------------------------------
    # Too small
    # ---------------------------------------------------------

    #if looks_like_gibberish(text):
    #    return True

    if len(text) < min_chars:
        return True

    # ---------------------------------------------------------
    # Keyword count
    # ---------------------------------------------------------

    keyword_hits = 0

    for keyword in NOISE_KEYWORDS:
        if keyword in text:
            keyword_hits += 1

    if keyword_hits >= keyword_threshold:
        return True

    # ---------------------------------------------------------
    # Very few sentences
    # ---------------------------------------------------------

    sentence_count = len(re.findall(r"[.!?]", text))

    if sentence_count < 2:
        return True

    # ---------------------------------------------------------
    # Too many short lines
    # Typical of menus
    # ---------------------------------------------------------

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if len(lines) > 8:

        short_lines = sum(
            1
            for line in lines
            if len(line.split()) <= 4
        )

        if short_lines / len(lines) > 0.7:
            return True

    return False

# -----------------------------------------------------------------------------
# Process one webpage
# -----------------------------------------------------------------------------

def process_web_page(
        page,
        max_tokens=400,
        overlap=100):

    raw_chunks = split_markdown(
        page["markdown"],
        max_tokens=max_tokens,
        overlap=overlap
    )

    documents = []

    kept = []

    for chunk in raw_chunks:

        if not is_noise_chunk(chunk):
            kept.append(chunk)
        if len(kept) >= MAX_CHUNKS:
            break

    for idx, chunk in enumerate(kept):

        documents.append({
            "source": "web",
            "url": page["url"],
            "title": page.get("title", ""),
            "retrieved_at": datetime.now().isoformat(),
            "chunk_id": idx,
            "num_chunks": len(kept),
            "text": chunk,
        })

    return documents

# -----------------------------------------------------------------------------
# Process all webpages
# -----------------------------------------------------------------------------

def process_web_pages(
        pages,
        max_tokens=400,
        overlap=100):

    documents = []

    for page in pages:

        documents.extend(
            process_web_page(
                page,
                max_tokens=max_tokens,
                overlap=overlap
            )
        )

    return documents


if __name__ == "__main__":

    query = "Who is the US President ?"

    pages = scrape_query(query)

    documents = process_web_pages(
        pages,
        max_tokens=400,
        overlap=100
    )

    print(f"\nKept {len(documents)} chunks.\n")

    for doc in documents:
        print("=" * 80)
        print(doc["url"])
        print(f"Chunk {doc['chunk_id'] + 1}/{doc['num_chunks']}")
        print("-" * 80)
        print(doc["text"][:1000])  # Print first 1000 characters
        print()