# web_context.py

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "web" / "firecrawl"))



from API import search_firecrawl, scrape_urls
from web_fetcher import fetch_readable_many
from chunking import process_web_pages
from chunk_reranker import rerank

###############################################################################

MAX_URLS = 9
MAX_CHUNKS = 6

###############################################################################

def distribute_chunk_budget(total_budget, num_queries):

    if num_queries <= 0:
        return []

    base = total_budget // num_queries
    remainder = total_budget % num_queries

    return [
        base + (1 if i < remainder else 0)
        for i in range(num_queries)
    ]



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

    return "\n\n".join(sections)


###############################################################################


def build_websearch_context(search_queries, original_query):

    """
    Parameters
    ----------
    search_queries : list[str]
        Queries produced by the planner and used for reranking.

    original_query : str
        outdated, kept for debugging.

    Returns
    -------
    str
        Formatted web context.
    """

    ###########################################################################
    # Collect URLs
    ###########################################################################

    urls = []

    seen = set()

    for query in search_queries:

        try:

            results = search_firecrawl(query)

        except Exception as e:

            print(f"FireCrawl failed for '{query}': {e}")

            continue

        for url in results:

            if url not in seen:

                seen.add(url)

                urls.append(url)

    if not urls:

        return ""

    ###########################################################################
    # Scrape
    ###########################################################################


    pages = fetch_readable_many(urls)

    if not pages:

        return ""

    ###########################################################################
    # Chunk
    ###########################################################################

    documents = process_web_pages(
        pages,
        max_tokens=400,
        overlap=100,
    )

    if not documents:
        return ""

    ###########################################################################
    # Distribute chunk budget between planner queries
    ###########################################################################

    if len(search_queries) == 1 :
        MAX_CHUNKS = 4
    else:
        MAX_CHUNKS = 6

    budgets = distribute_chunk_budget(
        MAX_CHUNKS,
        len(search_queries),
    )

    ###########################################################################
    # Rerank using the GENERATED SEARCH queries
    ###########################################################################

    reranked = []
    for query, budget in zip(search_queries, budgets):

        if budget <= 0:
            continue

        results = rerank(
            query,
            documents,
            top_k=budget,
        )

        reranked.extend(results)

    ###########################################################################
    # Remove duplicate chunks
    ###########################################################################

    unique_documents = []
    seen_documents = set()

    for doc in reranked:

        text = doc["text"]

        if text not in seen_documents:
            seen_documents.add(text)
            unique_documents.append(doc)

    ###########################################################################

    context = build_web_context(unique_documents)

    return f"""WEB SEARCH RESULTS

{context}

END OF WEB SEARCH RESULTS"""
