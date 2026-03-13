from duckduckgo_search import DDGS

def web_search(query: str, max_results: int = 3) -> list[dict]:
    """
    Performs a live DuckDuckGo search — no API key needed.
    Returns title, snippet, and URL for each result.
    """
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title":   r.get("title", ""),
                    "content": r.get("body", "")[:400],
                    "url":     r.get("href", "")
                })
        return results
    except Exception as e:
        return [{"title": "Search Error", "content": str(e), "url": ""}]