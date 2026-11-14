"""실시간 웹 검색 제공자. 결과 URL은 수집기가 다시 가져가 인덱싱한다."""
from __future__ import annotations

import os
from urllib.parse import quote, urlencode

from .fetcher import FetchError, Fetcher


def web_search(fetcher: Fetcher, cfg, query: str, limit: int) -> list[dict]:
    s = cfg["search"]
    provider = s.get("provider", "none")
    if provider == "wikipedia":
        lang = s.get("wikipedia_lang") or "ko"
        base = f"https://{lang}.wikipedia.org"
        data = fetcher.get_json(
            f"{base}/w/api.php?"
            + urlencode({"action": "query", "list": "search", "srsearch": query, "srlimit": limit, "format": "json"})
        )
        return [
            {"url": f"{base}/wiki/{quote(r['title'].replace(' ', '_'))}", "title": r["title"]}
            for r in data.get("query", {}).get("search", [])
        ]
    if provider == "searxng":
        base = (s.get("searxng_url") or "").rstrip("/")
        if not base:
            raise FetchError("SearXNG 주소가 설정되지 않았습니다.")
        data = fetcher.get_json(f"{base}/search?" + urlencode({"q": query, "format": "json"}))
        return [{"url": r["url"], "title": r.get("title", "")} for r in data.get("results", [])][:limit]
    if provider == "brave":
        key = s.get("brave_api_key") or os.environ.get("BRAVE_API_KEY", "")
        if not key:
            raise FetchError("Brave Search API 키가 없습니다 (설정 또는 BRAVE_API_KEY 환경변수).")
        data = fetcher.get_json(
            "https://api.search.brave.com/res/v1/web/search?" + urlencode({"q": query, "count": limit}),
            headers={"X-Subscription-Token": key},
        )
        return [{"url": r["url"], "title": r.get("title", "")} for r in data.get("web", {}).get("results", [])]
    raise FetchError("검색 제공자가 설정되지 않았습니다 (설정 > 검색 제공자).")
