import requests

from API import search_firecrawl, scrape_urls
from chunking import process_web_pages
from chunk_reranker import rerank
from web_fetcher import fetch_readable_many

import time

# ---------------------------------------------------------------------
# Llama.cpp server
# ---------------------------------------------------------------------

LLAMA_SERVER = "http://localhost:8081/v1/chat/completions"

SYSTEM_PROMPT = """
You are Jarvis.

You have access to web documents retrieved for the user's question.

Each document has:

- a title
- a URL
- a chunk of text

Instructions:

- Answer using the retrieved information whenever possible.
- Ignore irrelevant chunks.
- If multiple sources disagree, mention it.
- Cite the source number in your answer using (Source X).
- Do not invent information that is not present in the web context.
"""

# ---------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------

def build_web_context(documents):

    sections = []

    for i, doc in enumerate(documents, 1):

        sections.append(
f"""Source {i}

Title:
{doc["title"]}

URL:
{doc["url"]}

Content:

{doc["text"]}
"""
        )

    return "\n\n" + ("=" * 80 + "\n\n").join(sections)


# ---------------------------------------------------------------------
# GPT call
# ---------------------------------------------------------------------

def ask_gpt(user_query, web_context):

    prompt = f"""
WEB CONTEXT

{web_context}

END OF WEB CONTEXT

User question:

{user_query}
"""

    payload = {
        "model": "gpt-oss",
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.2,
        "stream": False,
    }

    response = requests.post(
        LLAMA_SERVER,
        json=payload,
        timeout=300,
    )

    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    query = input("Question: ")

    print("\nSearching the web...")

    urls = search_firecrawl(query)

    print(f"Found {len(urls)} webpages.")

    start = time.perf_counter()


    pages = fetch_readable_many(urls)

    end = time.perf_counter()

    print(f"Scraped {len(pages)} webpages.")
    print(f"Scraping time: {(end - start) * 1000:.2f} ms")


    start = time.perf_counter()
    documents = process_web_pages(
        pages,
        max_tokens=400,
        overlap=100,
    )
    end = time.perf_counter()
    print(f"Chunking time: {(end - start) * 1000:.2f} ms")


    print(f"Reranking {len(pages)} webpages.")

    start = time.perf_counter()
    docs = rerank(query, documents)
    end = time.perf_counter()

    print(f"Reranking time: {(end - start) * 1000:.2f} ms")

    for score in docs:
        print(f"Score: {score['score']:.4f}")
    print(f"Reranked {len(pages)} webpages.")

    print(f"Generated {len(docs)} chunks.")
    print("=" * 80)
    print(docs[-1]["text"])  # Print first 1000 characters

    context = build_web_context(documents)

    print("\nGenerating answer...\n")

    #answer = ask_gpt(
    #    query,
    #    context,
    #)

    #print("=" * 80)
    #print(answer)


if __name__ == "__main__":
    main()