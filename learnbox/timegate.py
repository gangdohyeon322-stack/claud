"""수집 허용 시간대 판정. 인터넷에 나가는 모든 요청 직전에 호출된다."""
from __future__ import annotations

from datetime import datetime, timedelta, tzinfo

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:  # pragma: no cover
    ZoneInfo = None
    ZoneInfoNotFoundError = Exception

DAY_NAMES = ["월", "화", "수", "목", "금", "토", "일"]


class OutsideWindow(Exception):
    """허용 시간대 밖에서 인터넷 접속을 시도함."""


def _tz(name: str) -> tzinfo | None:
    if ZoneInfo is None:
        return None
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        # Windows에는 시간대 DB가 없을 수 있다: pip install tzdata
        return None


def _minutes(hhmm: str) -> int:
    h, m = hhmm.strip().split(":")
    v = int(h) * 60 + int(m)
    if not 0 <= v <= 1440:
        raise ValueError(f"잘못된 시각: {hhmm}")
    return v


def now_in(tz_name: str) -> datetime:
    tz = _tz(tz_name)
    return datetime.now(tz) if tz else datetime.now().astimezone()


def is_allowed(windows: list[dict], tz_name: str, now: datetime | None = None) -> bool:
    if now is None:
        now = now_in(tz_name)
    else:
        tz = _tz(tz_name)
        if tz is not None:
            now = now.replace(tzinfo=tz) if now.tzinfo is None else now.astimezone(tz)
    minute = now.hour * 60 + now.minute
    wd = now.weekday()
    for w in windows:
        start, end = _minutes(w["start"]), _minutes(w["end"])
        days = set(w.get("days", range(7)))
        if start <= end:
            if wd in days and start <= minute < end:
                return True
        else:  # 자정을 넘기는 구간: 시작한 요일 기준
            if wd in days and minute >= start:
                return True
            if (wd - 1) % 7 in days and minute < end:
                return True
    return False


def next_allowed(windows: list[dict], tz_name: str, now: datetime | None = None) -> datetime | None:
    """다음 허용 시각(분 단위 탐색, 최대 8일)."""
    now = now or now_in(tz_name)
    t = now.replace(second=0, microsecond=0)
    for _ in range(8 * 24 * 60):
        if is_allowed(windows, tz_name, t):
            return t
        t += timedelta(minutes=1)
    return None


def describe(windows: list[dict]) -> str:
    parts = []
    for w in windows:
        days = w.get("days", list(range(7)))
        d = "매일" if len(set(days)) == 7 else ",".join(DAY_NAMES[i] for i in sorted(days))
        parts.append(f"{d} {w['start']}~{w['end']}")
    return " / ".join(parts) or "허용 시간 없음"


def ensure_allowed(cfg) -> None:
    if not is_allowed(cfg["allowed_windows"], cfg["timezone"]):
        raise OutsideWindow(f"수집 허용 시간이 아닙니다 ({describe(cfg['allowed_windows'])}).")
