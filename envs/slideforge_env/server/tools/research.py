"""Research tools: web_search, fetch_url."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...models import SlideForgeState


def web_search(state: SlideForgeState, query: str) -> tuple[str, bool]:
    """Search the web for information using Serper API."""
    if state.brief and state.brief.confidence >= 1.0:
        return "Skipped: confidence=1.0, no web verification needed.", True

    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        # Return mock results if no API key
        result = (
            f"[Search results for: {query}]\n"
            f"1. Overview and key facts about {query}.\n"
            f"2. Latest developments and trends in {query}.\n"
            f"3. Expert analysis and market insights for {query}."
        )
        state.research_context.append({"query": query, "results": result})
        return result, True

    try:
        import httpx
        resp = httpx.post(
            "https://google.serper.dev/search",
            json={"q": query, "num": 5},
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        
        results = []
        # Knowledge graph if available
        kg = data.get("knowledgeGraph", {})
        if kg:
            kg_title = kg.get("title", "")
            kg_desc = kg.get("description", "")
            if kg_title and kg_desc:
                results.append(f"[Knowledge Graph] {kg_title}: {kg_desc}")
        
        # Organic results
        for item in data.get("organic", [])[:5]:
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            if title and snippet:
                results.append(f"- {title}: {snippet}")
        
        result = "\n".join(results) if results else f"No specific results found for: {query}"
        state.research_context.append({"query": query, "results": result})
        return result, True
    except httpx.HTTPStatusError as e:
        error_msg = f"Search API error ({e.response.status_code}): {e.response.text[:200]}"
        return error_msg, False
    except Exception as e:
        return f"Search failed: {type(e).__name__}: {e}", False


def fetch_url(state: SlideForgeState, url: str) -> tuple[str, bool]:
    """Fetch content from a URL."""
    try:
        import httpx
        resp = httpx.get(url, timeout=10, follow_redirects=True)
        resp.raise_for_status()
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)[:2000]
        state.research_context.append({"url": url, "content": text})
        return text, True
    except Exception as e:
        return f"Fetch failed: {e}", False
