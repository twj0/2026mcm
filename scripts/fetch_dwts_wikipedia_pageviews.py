"""Fetch DWTS celebrity popularity proxy via Wikipedia Pageviews.

Main external proxy signal (recommended):
- Wikipedia pageviews during each season's airing window.

How it works:
1) Read the official contest dataset to get (season, celebrity_name).
2) For each season, fetch the season page wikitext from Wikipedia and parse:
   - first_aired
   - last_aired
3) For each celebrity, search a likely Wikipedia article title.
4) Query Wikimedia Pageviews API for that article in the season time window.
5) Save a tidy CSV to data/raw/ with a small metadata JSON for reproducibility.

References:
- MediaWiki API: https://www.mediawiki.org/wiki/API:Main_page
- Wikimedia Pageviews API: https://wikitech.wikimedia.org/wiki/Analytics/AQS/Pageviews

Notes:
- Pageviews coverage is limited historically; for many projects it is reliably available
  from ~2015-07 onward. Earlier seasons may be NA by design.
- This script uses `requests` (no browser automation).
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

WIKI_API = "https://en.wikipedia.org/w/api.php"
PAGEVIEWS_API_TMPL = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia.org/all-access/user/{article}/daily/{start}/{end}"
)

WIKI_HEADERS = {
    "User-Agent": "2026mcm-mcm2026/0.1 (MCM modeling; requests)",
    "Accept": "application/json",
}

DEFAULT_OUTPUT_FILENAME = "dwts_wikipedia_pageviews.csv"
DEFAULT_META_FILENAME = "dwts_wikipedia_pageviews.meta.json"

# Empirically safe lower bound for Pageviews API coverage for many use-cases.
# If the requested window ends before this date, we output NA.
PAGEVIEWS_EARLIEST_DATE = date(2015, 7, 1)

_SESSION = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _get_session():
    import requests

    global _SESSION
    if _SESSION is None:
        s = requests.Session()
        s.headers.update(WIKI_HEADERS)
        _SESSION = s
    return _SESSION


def _request_json(
    url: str,
    params: dict | None,
    timeout: float,
    *,
    max_retries: int,
) -> dict | list:
    session = _get_session()
    last_err: Exception | None = None

    for attempt in range(1, int(max_retries) + 1):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            if resp.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
                ra = resp.headers.get("Retry-After")
                if ra and ra.isdigit():
                    time.sleep(float(ra))
                else:
                    time.sleep(min(0.8 * (2 ** (attempt - 1)), 8.0))
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as err:
            last_err = err
            if attempt < max_retries:
                time.sleep(min(0.8 * (2 ** (attempt - 1)), 8.0))
                continue
            raise

    if last_err is not None:
        raise last_err
    raise RuntimeError("request failed")


def _wiki_search_top_title(
    query: str,
    timeout: float,
    *,
    banned_substrings: tuple[str, ...] = (),
    limit: int = 5,
    max_retries: int,
) -> str | None:
    """Return the top Wikipedia page title for a search query."""

    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srnamespace": 0,
        "format": "json",
        "srlimit": limit,
    }
    data = _request_json(WIKI_API, params=params, timeout=timeout, max_retries=max_retries)
    hits = data.get("query", {}).get("search", [])
    if not hits:
        return None

    banned_lower = tuple(s.lower() for s in banned_substrings)
    for h in hits:
        title = h.get("title")
        if not title:
            continue
        t = title.lower()
        if any(b in t for b in banned_lower):
            continue
        return title

    return None


def _wiki_get_wikitext(page_title: str, timeout: float, *, max_retries: int) -> str:
    """Fetch raw wikitext for a Wikipedia page title."""

    params = {
        "action": "query",
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "format": "json",
        "titles": page_title,
    }
    data = _request_json(WIKI_API, params=params, timeout=timeout, max_retries=max_retries)
    pages = data.get("query", {}).get("pages", {})
    # pages is a dict keyed by pageid
    for _, page in pages.items():
        revs = page.get("revisions", [])
        if not revs:
            continue
        slots = revs[0].get("slots", {})
        main = slots.get("main", {})
        content = main.get("*") or main.get("content") or ""
        return content
    raise RuntimeError(f"Failed to fetch wikitext for page: {page_title}")


_START_DATE_RE = re.compile(
    r"\{\{\s*(?:start|end)\s*date(?:\s*and\s*age)?\s*\|\s*(\d{4})\s*\|\s*(\d{1,2})\s*\|\s*(\d{1,2})",
    re.IGNORECASE,
)


def _clean_wikitext_value(v: str) -> str:
    s = v.strip()
    s = re.sub(r"<!--.*?-->", "", s)
    s = re.sub(r"<ref[^>/]*/>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"<ref[^>]*>.*?</ref>", "", s, flags=re.IGNORECASE | re.DOTALL)
    s = s.replace("<br />", " ").replace("<br/>", " ").replace("<br>", " ")
    # Remove common wiki markup
    s = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]", r"\1", s)  # [[A|B]] -> B
    s = re.sub(r"\{\{.*?\}\}", "", s)  # drop templates (we handle start date separately)
    s = s.replace("&nbsp;", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _parse_date_from_wikitext(v: str) -> date | None:
    """Parse dates like 'June 1, 2005' or '{{Start date|2005|6|1}}'."""

    m = _START_DATE_RE.search(v.lower())
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return date(y, mo, d)

    cleaned = _clean_wikitext_value(v)
    if not cleaned:
        return None

    # Pandas uses dateutil internally (already a dependency via pandas).
    try:
        dt = pd.to_datetime(cleaned, errors="raise").to_pydatetime()
        return dt.date()
    except Exception:
        return None


def _extract_infobox_field(wikitext: str, field_name: str) -> str | None:
    """Extract a field from wikitext lines like: | first_aired = ..."""

    # Use MULTILINE to match the start of a line.
    pattern = re.compile(rf"^\|\s*{re.escape(field_name)}\s*=\s*(.+)$", re.IGNORECASE | re.MULTILINE)
    m = pattern.search(wikitext)
    if not m:
        return None
    return m.group(1).strip()


def get_season_airing_window(
    season: int,
    buffer_days: int,
    timeout: float,
    *,
    max_retries: int,
) -> tuple[date | None, date | None, str | None]:
    """Return (window_start, window_end, season_page_title)."""

    candidates = [
        f"Dancing with the Stars (American TV series) season {season}",
        f"Dancing with the Stars (American season {season})",
        f"Dancing with the Stars season {season}",
    ]

    page_title = None
    for q in candidates:
        page_title = _wiki_search_top_title(q, timeout=timeout, max_retries=max_retries)
        if page_title:
            break

    if not page_title:
        return None, None, None

    wikitext = _wiki_get_wikitext(page_title, timeout=timeout, max_retries=max_retries)
    first_raw = _extract_infobox_field(wikitext, "first_aired")
    last_raw = _extract_infobox_field(wikitext, "last_aired")

    if not first_raw or not last_raw:
        return None, None, page_title

    first = _parse_date_from_wikitext(first_raw)
    last = _parse_date_from_wikitext(last_raw)
    if not first or not last:
        return None, None, page_title

    window_start = first - timedelta(days=buffer_days)
    window_end = last + timedelta(days=buffer_days)

    # Avoid querying future dates.
    today_utc = datetime.utcnow().date()
    if window_end >= today_utc:
        window_end = today_utc - timedelta(days=1)

    return window_start, window_end, page_title


def _format_pageviews_ts(d: date) -> str:
    # Pageviews API expects YYYYMMDD00
    return d.strftime("%Y%m%d") + "00"


def _get_pageviews(
    article_title: str,
    start: date,
    end: date,
    timeout: float,
    *,
    max_retries: int,
) -> dict:
    session = _get_session()

    encoded = urllib.parse.quote(article_title.replace(" ", "_"), safe="")
    url = PAGEVIEWS_API_TMPL.format(article=encoded, start=_format_pageviews_ts(start), end=_format_pageviews_ts(end))

    last_err: Exception | None = None
    for attempt in range(1, int(max_retries) + 1):
        try:
            resp = session.get(url, timeout=timeout)
            if resp.status_code == 404:
                return {"status": "not_found", "views_sum": None, "views_mean": None, "views_max": None, "days": 0}
            if resp.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
                ra = resp.headers.get("Retry-After")
                if ra and ra.isdigit():
                    time.sleep(float(ra))
                else:
                    time.sleep(min(0.8 * (2 ** (attempt - 1)), 8.0))
                continue
            resp.raise_for_status()
            break
        except Exception as err:
            last_err = err
            if attempt < max_retries:
                time.sleep(min(0.8 * (2 ** (attempt - 1)), 8.0))
                continue
            raise

    if last_err is not None and 'resp' not in locals():
        raise last_err

    payload = resp.json()
    items = payload.get("items", [])
    if not items:
        return {"status": "no_items", "views_sum": None, "views_mean": None, "views_max": None, "days": 0}

    views = [int(it.get("views", 0)) for it in items]
    return {
        "status": "ok",
        "views_sum": int(sum(views)),
        "views_mean": float(sum(views) / len(views)),
        "views_max": int(max(views)),
        "days": int(len(views)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch Wikipedia pageviews for DWTS celebrities during season airing windows."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="Input dataset CSV. Default: <repo_root>/mcm2026c/2026_MCM_Problem_C_Data.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: <repo_root>/data/raw",
    )
    parser.add_argument(
        "--season-min",
        type=int,
        default=None,
        help="Optional minimum season (inclusive).",
    )
    parser.add_argument(
        "--season-max",
        type=int,
        default=None,
        help="Optional maximum season (inclusive).",
    )
    parser.add_argument(
        "--buffer-days",
        type=int,
        default=7,
        help="Extend the airing window by +/- N days (default: 7).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds (default: 30).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries for transient network/API errors (default: 3).",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.2,
        help="Sleep between API calls to be polite (default: 0.2).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files if present.",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    input_csv = Path(args.input_csv) if args.input_csv else (repo_root / "mcm2026c" / "2026_MCM_Problem_C_Data.csv")
    output_dir = Path(args.output_dir) if args.output_dir else (repo_root / "data" / "raw")

    df = pd.read_csv(input_csv)
    df = df[["season", "celebrity_name"]].dropna()
    df["season"] = df["season"].astype(int)

    if args.season_min is not None:
        df = df[df["season"] >= args.season_min]
    if args.season_max is not None:
        df = df[df["season"] <= args.season_max]

    pairs = df.drop_duplicates().sort_values(["season", "celebrity_name"]).reset_index(drop=True)

    # Cache season -> airing window and wiki page title.
    season_window: dict[int, tuple[date | None, date | None, str | None]] = {}

    # Cache celebrity_name -> wikipedia article title.
    title_cache: dict[str, str | None] = {}

    records: list[dict] = []

    for row in pairs.itertuples(index=False):
        season = int(row.season)
        name = str(row.celebrity_name)

        if season not in season_window:
            ws, we, season_page = get_season_airing_window(
                season=season,
                buffer_days=args.buffer_days,
                timeout=args.timeout,
                max_retries=args.max_retries,
            )
            season_window[season] = (ws, we, season_page)
            time.sleep(args.sleep_seconds)

        window_start, window_end, season_page_title = season_window[season]

        if name not in title_cache:
            q1 = f'"{name}" Dancing with the Stars'
            title = _wiki_search_top_title(
                q1,
                timeout=args.timeout,
                banned_substrings=("Dancing with the Stars",),
                max_retries=args.max_retries,
            )
            if not title:
                title = _wiki_search_top_title(
                    name,
                    timeout=args.timeout,
                    banned_substrings=("Dancing with the Stars",),
                    max_retries=args.max_retries,
                )
            title_cache[name] = title
            time.sleep(args.sleep_seconds)

        article_title = title_cache[name]

        rec = {
            "season": season,
            "celebrity_name": name,
            "season_wikipedia_page": season_page_title,
            "window_start": window_start.isoformat() if window_start else None,
            "window_end": window_end.isoformat() if window_end else None,
            "wikipedia_article": article_title,
            "pageviews_status": None,
            "pageviews_sum": None,
            "pageviews_mean": None,
            "pageviews_max": None,
            "pageviews_days": 0,
        }

        if not article_title or not window_start or not window_end:
            rec["pageviews_status"] = "missing_title_or_window"
            records.append(rec)
            continue

        if window_end < PAGEVIEWS_EARLIEST_DATE:
            rec["pageviews_status"] = "out_of_coverage"
            records.append(rec)
            continue

        api_start = max(window_start, PAGEVIEWS_EARLIEST_DATE)
        api_end = window_end

        try:
            pv = _get_pageviews(
                article_title=article_title,
                start=api_start,
                end=api_end,
                timeout=args.timeout,
                max_retries=args.max_retries,
            )
            rec["pageviews_status"] = pv["status"]
            rec["pageviews_sum"] = pv["views_sum"]
            rec["pageviews_mean"] = pv["views_mean"]
            rec["pageviews_max"] = pv["views_max"]
            rec["pageviews_days"] = pv["days"]
            rec["api_start"] = api_start.isoformat()
            rec["api_end"] = api_end.isoformat()
        except Exception as e:
            rec["pageviews_status"] = f"error:{type(e).__name__}"

        records.append(rec)
        time.sleep(args.sleep_seconds)

    out_df = pd.DataFrame.from_records(records)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / DEFAULT_OUTPUT_FILENAME
    meta_path = output_dir / DEFAULT_META_FILENAME

    if not args.overwrite and (csv_path.exists() or meta_path.exists()):
        raise FileExistsError(f"Output exists. Use --overwrite. Existing: {csv_path} / {meta_path}")

    out_df.to_csv(csv_path, index=False, encoding="utf-8")

    meta = {
        "source": "Wikipedia + Wikimedia Pageviews API",
        "wikipedia_api": WIKI_API,
        "pageviews_api_template": PAGEVIEWS_API_TMPL,
        "pageviews_earliest_date": PAGEVIEWS_EARLIEST_DATE.isoformat(),
        "input_csv": str(input_csv),
        "fetched_at_utc": _utc_now_iso(),
        "notes": [
            "Season windows are parsed from Wikipedia season pages (first_aired/last_aired) with a buffer.",
            "Celebrity article titles are resolved by Wikipedia search with DWTS context.",
            "If a season ends before pageviews coverage, values are NA (out_of_coverage).",
        ],
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {meta_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
