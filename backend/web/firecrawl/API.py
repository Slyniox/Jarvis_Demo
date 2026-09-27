import requests
from firecrawl import Firecrawl
from pathlib import Path
import json

CONFIG_FILE = Path(__file__).resolve().parents[2] / "manager" / "config.json"
with CONFIG_FILE.open("r", encoding="utf-8") as config_file:
    config = json.load(config_file)

SEARXNG_URL = config["web"]["searxng_url"]
FIRECRAWL_API_KEY = config["web"]["firecrawl_api_key"]

firecrawl = Firecrawl(api_key=FIRECRAWL_API_KEY)


def search_searxng(query: str, max_results: int = 1):
    response = requests.get(
        SEARXNG_URL,
        params={
            "q": f"{query} -site:youtube.com",
            "format": "json",
        },
        headers={
            "User-Agent": "Jarvis/1.0",
        },
        timeout=15,
    )

    response.raise_for_status()

    results = response.json()["results"]

    urls = []

    for result in results:
        url = result["url"]

        if url not in urls:
            urls.append(url)

        if len(urls) >= max_results:
            break

    return urls


def search_firecrawl(query: str, max_results: int = 2):
    print(f"[WEB SEARCH] Query: {query}")
    try:
        result = firecrawl.search(
            query,
            limit=max_results,
            exclude_domains=["reddit.com", "youtube.com", "llm-stats.com"],
        )
        urls = []
        for item in result.web:
            url = item.url
            if url not in urls:
                urls.append(url)
            if len(urls) >= max_results:
                break
        print(f"[WEB SEARCH] Found {len(urls)} result(s)")
        for url in urls:
            print(f"[WEB SEARCH] {url}")
        return urls
    except Exception as e:
        print(f"[WEB SEARCH] Search failed: {e}")
        return []


def scrape_urls(urls):
    pages = []

    for url in urls:
        print(f"Scraping {url}")

        try:
            doc = firecrawl.scrape(
                url,
                formats=["markdown"],
                only_main_content=True,
            )

            pages.append({
                "url": url,
                "title": doc.metadata.title if doc.metadata else "",
                "markdown": doc.markdown,
            })

        except Exception as e:
            print(f"Failed: {url}")
            print(e)

    return pages


def scrape_query(query: str):
    urls = search_firecrawl(query)
    pages = scrape_urls(urls)
    return (pages)
    


def main():
    query = input("Search: ")

    print("\nSearching FireCrawl...")
    urls = search_firecrawl(query)

    print(f"Found {len(urls)} results.\n")

    pages = scrape_urls(urls)

    for i, page in enumerate(pages, start=1):
        print(f"\n{'=' * 80}")
        print(f"Result #{i}")
        print(f"URL : {page['url']}")
        print(f"Size: {len(page['markdown'])} characters\n")
        print(page["markdown"][:500])

if __name__ == "__main__":
    main()
