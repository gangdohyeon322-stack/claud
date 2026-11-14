"""수집 작업: 지정 링크 학습, 실시간 검색 후 수집."""
from __future__ import annotations

from collections import deque
from urllib.parse import urlsplit

from .fetcher import BlockedURL, FetchError, Fetcher
from .search_providers import web_search
from .timegate import OutsideWindow


def _norm(url: str) -> str:
    p = urlsplit(url)
    return f"{p.scheme}://{(p.hostname or '').lower()}{p.path or '/'}{'?' + p.query if p.query else ''}"


def collect_links(ctx, cfg, db, fetcher: Fetcher, urls: list[str], depth: int = 0,
                  same_domain: bool = True, source: str = "link", query: str | None = None,
                  max_pages: int | None = None) -> dict:
    max_pages = int(max_pages or cfg["max_pages_per_job"])
    min_chars = int(cfg["min_text_chars"])
    queue = deque((u.strip(), 0) for u in urls if u.strip())
    seen: set[str] = set()
    stats = {"new": 0, "updated": 0, "unchanged": 0, "skipped": 0, "errors": [], "stopped": None}
    visited = 0
    while queue and visited < max_pages:
        ctx.check_cancel()
        url, d = queue.popleft()
        key = _norm(url)
        if key in seen:
            continue
        seen.add(key)
        visited += 1
        ctx.progress(visited / max_pages, f"[{visited}/{max_pages}] {url}")
        try:
            page = fetcher.fetch_page(url)
        except OutsideWindow as e:
            stats["stopped"] = str(e)
            break
        except (BlockedURL, FetchError) as e:
            stats["errors"].append(str(e))
            continue
        except Exception as e:  # 개별 페이지 오류로 작업 전체를 멈추지 않음
            stats["errors"].append(f"{url}: {e}")
            continue
        if len(page.text) >= min_chars:
            stats[db.add_document(page.url, page.title, page.text, source, query)] += 1
        else:
            stats["skipped"] += 1
        if d < depth:
            base_host = urlsplit(url).hostname
            for link in page.links:
                if same_domain and urlsplit(link).hostname != base_host:
                    continue
                if _norm(link) not in seen:
                    queue.append((link, d + 1))
    stats["visited"] = visited
    stats["errors"] = stats["errors"][:30]
    return stats


def collect_search(ctx, cfg, db, fetcher: Fetcher, query: str, limit: int | None = None) -> dict:
    limit = int(limit or cfg["search"].get("results_to_fetch", 5))
    ctx.progress(0.0, f"검색 중: {query}")
    try:
        results = web_search(fetcher, cfg, query, limit)
    except OutsideWindow as e:
        return {"stopped": str(e), "results": []}
    urls = [r["url"] for r in results]
    stats = collect_links(ctx, cfg, db, fetcher, urls, depth=0, source="search", query=query, max_pages=len(urls) or 1)
    stats["results"] = results
    return stats
