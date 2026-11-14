"""웹 요청: 시간대 확인, robots.txt 준수, 도메인 허용/차단, 요청 간격, 내부망 차단."""
from __future__ import annotations

import ipaddress
import socket
import threading
import time
import urllib.error
import urllib.request
import urllib.robotparser
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from .extract import decode, html_to_text
from .timegate import ensure_allowed


class FetchError(Exception):
    pass


class BlockedURL(FetchError):
    pass


@dataclass
class Page:
    url: str
    title: str
    text: str
    links: list[str] = field(default_factory=list)


def host_matches(host: str, domains: list[str]) -> bool:
    host = host.lower().rstrip(".")
    for d in domains:
        d = d.lower().strip().lstrip(".")
        if d and (host == d or host.endswith("." + d)):
            return True
    return False


class Fetcher:
    ROBOTS_TTL = 3600

    def __init__(self, cfg):
        self.cfg = cfg
        self._last: dict[str, float] = {}
        self._robots: dict[str, tuple[float, urllib.robotparser.RobotFileParser]] = {}
        self._lock = threading.Lock()
        fetcher = self

        class _Redirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                fetcher.check_url(newurl)  # 리다이렉트 대상도 같은 규칙 적용
                return super().redirect_request(req, fp, code, msg, headers, newurl)

        self.opener = urllib.request.build_opener(_Redirect)

    # ---------- 규칙 ----------
    def check_url(self, url: str) -> None:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise BlockedURL(f"http/https 주소만 가능합니다: {url}")
        host = parts.hostname
        if host_matches(host, self.cfg["blocked_domains"]):
            raise BlockedURL(f"차단 목록의 도메인입니다: {host}")
        allowed = self.cfg["allowed_domains"]
        if allowed and not host_matches(host, allowed):
            raise BlockedURL(f"허용 목록에 없는 도메인입니다: {host}")
        if not self.cfg["allow_private_network"]:
            self._check_public(host)

    @staticmethod
    def _check_public(host: str) -> None:
        try:
            infos = socket.getaddrinfo(host, None)
        except socket.gaierror as e:
            raise FetchError(f"주소를 찾을 수 없습니다: {host}") from e
        for info in infos:
            ip = ipaddress.ip_address(info[4][0].split("%")[0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
                raise BlockedURL(f"내부망 주소는 수집할 수 없습니다: {host}")

    def _wait_turn(self, host: str) -> None:
        interval = float(self.cfg["request_interval_sec"])
        with self._lock:
            last = self._last.get(host, 0.0)
            wait = last + interval - time.monotonic()
            self._last[host] = max(time.monotonic(), last + interval)
        if wait > 0:
            time.sleep(wait)

    # ---------- 저수준 요청 ----------
    def _request(self, url: str, headers: dict | None = None, max_bytes: int | None = None) -> tuple[int, str, str, bytes]:
        ensure_allowed(self.cfg)  # 요청 직전마다 시간대 확인
        self.check_url(url)
        self._wait_turn(urlsplit(url).hostname or "")
        req = urllib.request.Request(url, headers={"User-Agent": self.cfg["user_agent"], **(headers or {})})
        limit = max_bytes or int(self.cfg["max_page_bytes"])
        try:
            with self.opener.open(req, timeout=20) as r:
                body = r.read(limit + 1)
                if len(body) > limit:
                    raise FetchError(f"페이지가 너무 큽니다 (>{limit} bytes): {url}")
                return r.status, r.geturl(), r.headers.get("Content-Type", ""), body
        except urllib.error.HTTPError as e:
            return e.code, url, e.headers.get("Content-Type", "") if e.headers else "", b""
        except urllib.error.URLError as e:
            raise FetchError(f"접속 실패: {url} ({e.reason})") from e

    def get_json(self, url: str, headers: dict | None = None):
        import json

        status, _, _, body = self._request(url, {"Accept": "application/json", **(headers or {})})
        if status != 200:
            raise FetchError(f"HTTP {status}: {url}")
        return json.loads(body.decode("utf-8"))

    # ---------- robots.txt ----------
    def robots_allows(self, url: str) -> bool:
        parts = urlsplit(url)
        key = f"{parts.scheme}://{parts.netloc}"
        cached = self._robots.get(key)
        if not cached or time.time() - cached[0] > self.ROBOTS_TTL:
            rp = urllib.robotparser.RobotFileParser()
            try:
                status, _, _, body = self._request(key + "/robots.txt", max_bytes=512_000)
            except FetchError:
                status, body = 503, b""
            if status == 200:
                rp.parse(body.decode("utf-8", errors="replace").splitlines())
            elif 400 <= status < 500:
                rp.allow_all = True  # RFC 9309: 4xx -> 제한 없음
            else:
                rp.disallow_all = True  # 5xx/접속 불가 -> 전부 금지
            self._robots[key] = (time.time(), rp)
            cached = self._robots[key]
        return cached[1].can_fetch(self.cfg["user_agent"].split("/")[0], url)

    # ---------- 페이지 ----------
    def fetch_page(self, url: str) -> Page:
        self.check_url(url)
        if not self.robots_allows(url):
            raise BlockedURL(f"robots.txt가 수집을 허용하지 않습니다: {url}")
        status, final_url, ctype, body = self._request(url)
        if status != 200:
            raise FetchError(f"HTTP {status}: {url}")
        ctype_l = ctype.lower()
        text = decode(body, ctype)
        if "html" in ctype_l or not ctype_l:
            title, content, links = html_to_text(text, final_url)
        elif ctype_l.startswith("text/"):
            title, content, links = final_url.rsplit("/", 1)[-1], text, []
        else:
            raise FetchError(f"지원하지 않는 형식({ctype}): {url}")
        return Page(url=final_url, title=title or final_url, text=content, links=links)
