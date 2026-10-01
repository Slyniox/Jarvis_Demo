# web_context.py

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "web" / "firecrawl"))



from API import search_firecrawl, scrape_urls
from web_fetcher import fetch_readable
from chunking import process_web_pages
from chunk_reranker import rerank

###############################################################################

MAX_QUERY_URLS = 6
MAX_CHUNKS = 12
MAX_SEARCH_WORKERS = 3
MAX_FETCH_WORKERS = 18

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

    if not search_queries:
        return ""

    # Start all searches together. As each one returns, start fetching its URLs
    # immediately so page downloads overlap with the searches still in flight.
    urls = []
    url_order = {}
    fetch_futures = {}
    search_workers = min(MAX_SEARCH_WORKERS, len(search_queries))


    with ThreadPoolExecutor(max_workers=search_workers) as search_pool, \
            ThreadPoolExecutor(max_workers=MAX_FETCH_WORKERS) as fetch_pool:
        search_futures = {
            search_pool.submit(search_firecrawl, query, MAX_QUERY_URLS): (index, query)
            for index, query in enumerate(search_queries)
        }

        for search_future in as_completed(search_futures):
            query_index, query = search_futures[search_future]

            try:
                results = search_future.result()
            except Exception as e:
                print(f"FireCrawl failed for '{query}': {e}")
                continue

            for result_index, url in enumerate(results):
                order = (query_index, result_index)
                if url in url_order:
                    # A later-finishing search may reveal that this URL belongs
                    # earlier in planner order; keep final output deterministic.
                    url_order[url] = min(url_order[url], order)
                    continue

                url_order[url] = order
                urls.append(url)
                fetch_futures[fetch_pool.submit(fetch_readable, url)] = url

        pages_by_order = {}
        for fetch_future in as_completed(fetch_futures):
            url = fetch_futures[fetch_future]
            try:
                page = fetch_future.result()
                print(f"[WEB FETCH] OK ({len(page['markdown'])} chars): {url}")
                pages_by_order[url_order[url]] = page
            except Exception as e:
                print(f"[WEB FETCH] FAILED {url}: {e}")

    if not urls:
        return ""

    pages = [pages_by_order[key] for key in sorted(pages_by_order)]

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
        MAX_CHUNKS = 8
    else:
        MAX_CHUNKS = 12

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
