"""HTML에서 제목, 본문 텍스트, 링크를 뽑는다 (표준 라이브러리만 사용)."""
from __future__ import annotations

import re
from html.parser import HTMLParser
from urllib.parse import urldefrag, urljoin

SKIP = {"script", "style", "noscript", "template", "svg", "nav", "footer", "aside", "form", "iframe",
        "button", "select", "canvas"}
BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "article",
         "blockquote", "pre", "td", "th", "dd", "dt", "table", "ul", "ol", "main", "figcaption", "header", "hr"}


class _Parser(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base = base_url
        self.skip = 0
        self.in_title = False
        self.title = ""
        self.parts: list[str] = []
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self.in_title = True
        elif tag in SKIP:
            self.skip += 1
        elif tag == "a":
            href = dict(attrs).get("href")
            if href and not href.startswith(("javascript:", "mailto:", "tel:", "#")):
                url, _ = urldefrag(urljoin(self.base, href))
                if url.startswith(("http://", "https://")):
                    self.links.append(url)
        if tag in BLOCK:
            self.parts.append("\n")

    def handle_startendtag(self, tag, attrs):
        if tag in BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
        elif tag in SKIP and self.skip:
            self.skip -= 1
        if tag in BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.in_title:
            self.title += data
        elif not self.skip:
            self.parts.append(data)


def html_to_text(html: str, base_url: str = "") -> tuple[str, str, list[str]]:
    p = _Parser(base_url)
    try:
        p.feed(html)
        p.close()
    except Exception:  # 깨진 HTML이어도 읽은 데까지 사용
        pass
    lines = []
    for line in "".join(p.parts).split("\n"):
        line = re.sub(r"[ \t\r\f\v ]+", " ", line).strip()
        if len(line) >= 2:
            lines.append(line)
    seen, links = set(), []
    for u in p.links:
        if u not in seen:
            seen.add(u)
            links.append(u)
    return re.sub(r"\s+", " ", p.title).strip(), "\n".join(lines), links


_META_CHARSET = re.compile(rb"""<meta[^>]+charset=["']?([A-Za-z0-9_\-]+)""", re.I)


def decode(body: bytes, content_type: str) -> str:
    charset = None
    m = re.search(r"charset=([A-Za-z0-9_\-]+)", content_type or "", re.I)
    if m:
        charset = m.group(1)
    else:
        m2 = _META_CHARSET.search(body[:4096])
        if m2:
            charset = m2.group(1).decode("ascii", "ignore")
    charset = (charset or "utf-8").lower()
    if charset in ("euc-kr", "ks_c_5601-1987", "ksc5601", "x-windows-949"):
        charset = "cp949"
    try:
        return body.decode(charset, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")
