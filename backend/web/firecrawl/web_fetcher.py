import httpx
import trafilatura
from bs4 import BeautifulSoup
from markdownify import markdownify
from urllib.parse import urljoin, urlparse


MAX_BYTES = 4096 * 4096       # 4 MiB
DEFAULT_TIMEOUT = 10.0
MAX_REDIRECTS = 0

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/150.0.0.0 Safari/537.36"
)


class FetchError(Exception):
    pass


class ResponseTooLarge(FetchError):
    pass


def validate_url(url: str):
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise FetchError(
            f"Unsupported URL scheme: {parsed.scheme}"
        )

    if not parsed.hostname:
        raise FetchError("URL has no hostname")


def fetch_html(
    url: str,
    timeout: float = DEFAULT_TIMEOUT,
):
    validate_url(url)

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": (
            "text/html,application/xhtml+xml,"
            "application/xml;q=0.9,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    current_url = url

    timeout_config = httpx.Timeout(timeout)

    with httpx.Client(
        timeout=timeout_config,
        headers=headers,
        follow_redirects=False,
    ) as client:

        for redirect_count in range(MAX_REDIRECTS + 1):

            validate_url(current_url)

            try:
                with client.stream(
                    "GET",
                    current_url,
                ) as response:

                    # Redirect
                    if response.status_code in {
                        301, 302, 303, 307, 308
                    }:
                        location = response.headers.get("location")

                        if not location:
                            raise FetchError(
                                "Redirect without Location header"
                            )

                        if redirect_count >= MAX_REDIRECTS:
                            raise FetchError(
                                "Too many redirects"
                            )

                        current_url = urljoin(
                            current_url,
                            location,
                        )

                        continue

                    response.raise_for_status()

                    # Content type
                    content_type = response.headers.get(
                        "content-type",
                        "",
                    ).lower()

                    if (
                        "text/html" not in content_type
                        and "application/xhtml+xml" not in content_type
                    ):
                        raise FetchError(
                            f"Unsupported content type: "
                            f"{content_type}"
                        )

                    # Content-Length check
                    content_length = response.headers.get(
                        "content-length"
                    )

                    if content_length:
                        try:
                            if int(content_length) > MAX_BYTES:
                                raise ResponseTooLarge(
                                    "Response exceeds 4 MiB"
                                )
                        except ValueError:
                            pass

                    # Stream with hard limit
                    chunks = []
                    total = 0

                    for chunk in response.iter_bytes():

                        total += len(chunk)

                        if total > MAX_BYTES:
                            raise ResponseTooLarge(
                                "Response exceeded 4 MiB"
                            )

                        chunks.append(chunk)

                    raw = b"".join(chunks)

                    encoding = response.encoding or "utf-8"

                    html = raw.decode(
                        encoding,
                        errors="replace",
                    )

                    return str(response.url), html

            except httpx.HTTPError as e:
                raise FetchError(
                    f"HTTP request failed: {e}"
                ) from e

    raise FetchError(
        f"Failed to fetch {url}"
    )


def extract_readable_markdown(
    html: str,
    url: str | None = None,
):
    if not html:
        return "", ""

    extracted_html = trafilatura.extract(
        html,
        output_format="html",
        include_links=True,
        include_images=True,
        include_tables=True,
        include_formatting=True,
        url=url,
    )

    if not extracted_html:

        raise FetchError(
            "Could not extract readable content"
        )

    markdown = markdownify(
        extracted_html,
        heading_style="ATX",
    ).strip()

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    title_tag = soup.find("title")

    title = (
        title_tag.get_text(" ", strip=True)
        if title_tag
        else ""
    )

    return title, markdown


def fetch_readable(
    url: str,
    timeout: float = DEFAULT_TIMEOUT,
):
    final_url, html = fetch_html(
        url,
        timeout=timeout,
    )

    title, markdown = extract_readable_markdown(
        html,
        url=final_url,
    )

    return {
        "url": final_url,
        "title": title,
        "markdown": markdown,
    }


def fetch_readable_many(
    urls: list[str],
    timeout: float = DEFAULT_TIMEOUT,
):
    results = []

    for url in urls:

        try:
            print(f"[WEB FETCH] {url}")

            result = fetch_readable(
                url,
                timeout=timeout,
            )

            print(
                f"[WEB FETCH] OK "
                f"({len(result['markdown'])} chars)"
            )

            results.append(result)

        except Exception as e:

            print(
                f"[WEB FETCH] FAILED "
                f"{url}: {e}"
            )

    return results


if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:
        print(
            f"Usage: python {sys.argv[0]} <url>"
        )
        raise SystemExit(1)

    result = fetch_readable(sys.argv[1])

    print("=" * 80)
    print("URL:", result["url"])
    print("TITLE:", result["title"])
    print("=" * 80)
    print(result["markdown"])