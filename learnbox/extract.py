"""HTML에서 제목, 본문 텍스트, 링크를 뽑는다 (표준 라이브러리만 사용)."""
from __future__ import annotations

import re
from html.parser import HTMLParser
from urllib.parse import urldefrag, urljoin

SKIP = {"script", "style", "noscript", "template", "svg", "nav", "footer", "aside", "form", "iframe",
        "button", "select", "canvas",
        "annotation"}  # MathML의 LaTeX 원문({\displaystyle ...})은 본문 글이 아니다 (화면에 보이는 수식 글자는 남는다)
BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "article",
         "blockquote", "pre", "td", "th", "dd", "dt", "table", "ul", "ol", "main", "figcaption", "header", "hr"}

# 본문이 아닌 덩어리(메뉴, 언어 목록, 분류, 편집 링크 …)를 class / id / role로 알아본다. 위키백과(MediaWiki) 화면 구성이 대표적이다.
BOILER_CLASSES = frozenset({
    "mw-jump-link", "mw-editsection", "mw-portlet", "mw-interlanguage-selector", "navbox", "printfooter", "catlinks",
    "vector-header-container", "vector-main-menu-container", "vector-column-start", "vector-column-end",
    "vector-sticky-header", "vector-page-toolbar", "vector-dropdown", "vector-toc", "vector-settings",
    "noprint-nav", "cookie-banner", "cookie-notice", "breadcrumb", "breadcrumbs",
})
BOILER_IDS = frozenset({"catlinks", "toc", "mw-navigation", "mw-panel", "siteNotice", "p-lang-btn", "footer"})
BOILER_ROLES = frozenset({"navigation", "contentinfo", "complementary", "search"})
BOILER_TAG_CLASS = frozenset({("table", "sidebar"), ("sup", "reference")})  # 위키백과 옆 상자 표, 각주 번호 [1]
VOID = frozenset({"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"})


def _is_boilerplate(tag: str, attrs) -> bool:
    d = dict(attrs)
    classes = (d.get("class") or "").split()
    if any(c in BOILER_CLASSES for c in classes) or (d.get("id") or "") in BOILER_IDS:
        return True
    if (d.get("role") or "").lower() in BOILER_ROLES:
        return True
    return any((tag, c) in BOILER_TAG_CLASS for c in classes)


class _Parser(HTMLParser):
    def __init__(self, base_url: str, strip_boilerplate: bool = True):
        super().__init__(convert_charrefs=True)
        self.base = base_url
        self.strip = strip_boilerplate
        self.bp_tag: str | None = None  # 지금 건너뛰는 덩어리의 바깥 태그
        self.bp_depth = 0
        self.skip = 0
        self.in_title = False
        self.title = ""
        self.parts: list[str] = []
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        if self.bp_tag is not None:
            if tag == self.bp_tag:
                self.bp_depth += 1
        elif self.strip and tag not in VOID and _is_boilerplate(tag, attrs):
            self.bp_tag, self.bp_depth = tag, 1
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
        if self.bp_tag is not None and tag == self.bp_tag:
            self.bp_depth -= 1
            if self.bp_depth <= 0:
                self.bp_tag, self.bp_depth = None, 0
        if tag == "title":
            self.in_title = False
        elif tag in SKIP and self.skip:
            self.skip -= 1
        if tag in BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.in_title:
            self.title += data
        elif not self.skip and self.bp_tag is None:
            self.parts.append(data)


def _parse(html: str, base_url: str, strip: bool) -> tuple[str, str, list[str]]:
    p = _Parser(base_url, strip)
    try:
        p.feed(html)
        p.close()
    except Exception:  # 깨진 HTML이어도 읽은 데까지 사용
        pass
    lines = []
    for line in "".join(p.parts).split("\n"):
        line = re.sub(r"[ \t\r\f\v ]+", " ", line).strip()
        if len(line) >= 2:
            lines.append(line)
    seen, links = set(), []
    for u in p.links:
        if u not in seen:
            seen.add(u)
            links.append(u)
    return re.sub(r"\s+", " ", p.title).strip(), "\n".join(lines), links


def html_to_text(html: str, base_url: str = "") -> tuple[str, str, list[str]]:
    title, text, links = _parse(html, base_url, True)
    if len(text) < 200:
        # 덩어리 건너뛰기가 본문까지 지워 버렸을 수 있다(닫히지 않은 태그 등): 건너뛰지 않은 결과가 더 길면 그것을 쓴다
        t2, x2, l2 = _parse(html, base_url, False)
        if len(x2) > len(text):
            return t2, x2, l2
    return title, text, links


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
