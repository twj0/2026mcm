"""Fetch a *rough* popularity proxy via Google Trends (pytrends).

This script is intentionally NOT treated as a high-precision data source.
It is meant as a comparison signal against the main Wikipedia Pageviews proxy.

How it works:
1) Read the official contest dataset to get (season, celebrity_name).
2) For each season, fetch the season page wikitext from Wikipedia and parse:
   - first_aired
   - last_aired
3) For each celebrity, query Google Trends for the celebrity name during that window.

Important notes:
- `pytrends` is an unofficial library; Google may throttle or block automated requests.
- Keyword ambiguity is expected (same-name people). This is acceptable since this is a
  comparison signal, not a ground-truth feature.

References:
- pytrends: https://github.com/GeneralMills/pytrends
- MediaWiki API: https://www.mediawiki.org/wiki/API:Main_page
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


WIKI_API = "https://en.wikipedia.org/w/api.php"
DEFAULT_OUTPUT_FILENAME = "dwts_google_trends.csv"
DEFAULT_META_FILENAME = "dwts_google_trends.meta.json"

WIKI_HEADERS = {
    "User-Agent": "2026mcm-mcm2026/0.1 (MCM modeling; requests)",
    "Accept": "application/json",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _request_json(url: str, params: dict | None, timeout: float) -> dict | list:
    import requests

    resp = requests.get(url, params=params, timeout=timeout, headers=WIKI_HEADERS)
    resp.raise_for_status()
    return resp.json()


def _wiki_search_top_title(query: str, timeout: float) -> str | None:
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": 1,
    }
    data = _request_json(WIKI_API, params=params, timeout=timeout)
    hits = data.get("query", {}).get("search", [])
    if not hits:
        return None
    return hits[0].get("title")


def _wiki_get_wikitext(page_title: str, timeout: float) -> str:
    params = {
        "action": "query",
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "format": "json",
        "titles": page_title,
    }
    data = _request_json(WIKI_API, params=params, timeout=timeout)
    pages = data.get("query", {}).get("pages", {})
    for _, page in pages.items():
        revs = page.get("revisions", [])
        if not revs:
            continue
        slots = revs[0].get("slots", {})
        main = slots.get("main", {})
        return main.get("*") or main.get("content") or ""
    raise RuntimeError(f"Failed to fetch wikitext for page: {page_title}")


_START_DATE_RE = re.compile(
    r"\{\{\s*(?:start|end)\s*date(?:\s*and\s*age)?\s*\|\s*(\d{4})\s*\|\s*(\d{1,2})\s*\|\s*(\d{1,2})",
    re.IGNORECASE,
)


def _parse_date_from_wikitext(v: str) -> date | None:
    m = _START_DATE_RE.search(v.lower())
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return date(y, mo, d)

    # Fall back to dateutil via pandas.
    cleaned = re.sub(r"<!--.*?-->", "", v).strip()
    cleaned = re.sub(r"<ref[^>/]*/>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<ref[^>]*>.*?</ref>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]", r"\1", cleaned)
    cleaned = cleaned.replace("<br />", " ").replace("<br/>", " ").replace("<br>", " ")
    cleaned = cleaned.replace("&nbsp;", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    try:
        dt = pd.to_datetime(cleaned, errors="raise").to_pydatetime()
        return dt.date()
    except Exception:
        return None


def _extract_infobox_field(wikitext: str, field_name: str) -> str | None:
    pattern = re.compile(
        rf"^\|\s*{re.escape(field_name)}\s*=\s*(.+)$", re.IGNORECASE | re.MULTILINE
    )
    m = pattern.search(wikitext)
    if not m:
        return None
    return m.group(1).strip()


def get_season_airing_window(season: int, buffer_days: int, timeout: float) -> tuple[date | None, date | None, str | None]:
    candidates = [
        f"Dancing with the Stars (American TV series) season {season}",
        f"Dancing with the Stars (American season {season})",
        f"Dancing with the Stars season {season}",
    ]

    page_title = None
    for q in candidates:
        page_title = _wiki_search_top_title(q, timeout=timeout)
        if page_title:
            break

    if not page_title:
        return None, None, None

    wikitext = _wiki_get_wikitext(page_title, timeout=timeout)
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

    today_utc = datetime.utcnow().date()
    if window_end >= today_utc:
        window_end = today_utc - timedelta(days=1)

    return window_start, window_end, page_title


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch Google Trends (pytrends) signal for DWTS celebrities during season windows."
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
    parser.add_argument("--season-min", type=int, default=None, help="Optional minimum season.")
    parser.add_argument("--season-max", type=int, default=None, help="Optional maximum season.")
    parser.add_argument(
        "--buffer-days",
        type=int,
        default=7,
        help="Extend the season window by +/- N days (default: 7).",
    )
    parser.add_argument(
        "--geo",
        type=str,
        default="US",
        help="Google Trends geo code (default: US).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout seconds for Wikipedia API calls (default: 30).",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.0,
        help="Sleep between pytrends queries to reduce throttling (default: 1.0).",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=200,
        help="Safety cap: maximum number of pytrends queries (default: 200).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files if present.",
    )

    args = parser.parse_args()

    try:
        from pytrends.request import TrendReq
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "Missing dependency 'pytrends'. Install it first, e.g. `pip install pytrends`."
        ) from e

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

    season_window: dict[int, tuple[date | None, date | None, str | None]] = {}

    pytrends = TrendReq(hl="en-US", tz=360)

    records: list[dict] = []
    query_count = 0

    for row in pairs.itertuples(index=False):
        if query_count >= args.max_queries:
            break

        season = int(row.season)
        name = str(row.celebrity_name)

        if season not in season_window:
            ws, we, season_page = get_season_airing_window(season, args.buffer_days, args.timeout)
            season_window[season] = (ws, we, season_page)

        window_start, window_end, season_page_title = season_window[season]

        rec = {
            "season": season,
            "celebrity_name": name,
            "season_wikipedia_page": season_page_title,
            "window_start": window_start.isoformat() if window_start else None,
            "window_end": window_end.isoformat() if window_end else None,
            "geo": args.geo,
            "trends_status": None,
            "trends_mean": None,
            "trends_max": None,
            "trends_sum": None,
            "n_points": 0,
        }

        if not window_start or not window_end:
            rec["trends_status"] = "missing_window"
            records.append(rec)
            continue

        timeframe = f"{window_start.isoformat()} {window_end.isoformat()}"

        try:
            pytrends.build_payload([name], cat=0, timeframe=timeframe, geo=args.geo)
            it = pytrends.interest_over_time()
            query_count += 1

            if it is None or it.empty:
                rec["trends_status"] = "empty"
            else:
                series = it[name]
                rec["trends_status"] = "ok"
                rec["n_points"] = int(series.shape[0])
                rec["trends_mean"] = float(series.mean())
                rec["trends_max"] = float(series.max())
                rec["trends_sum"] = float(series.sum())
        except Exception as e:
            rec["trends_status"] = f"error:{type(e).__name__}"
        finally:
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
        "source": "Google Trends (pytrends, unofficial)",
        "input_csv": str(input_csv),
        "fetched_at_utc": _utc_now_iso(),
        "notes": [
            "This is a comparison-only proxy. It is not guaranteed stable due to Google throttling.",
            "Windows are aligned using Wikipedia season pages (first_aired/last_aired) with a buffer.",
            "Keyword ambiguity is expected (same-name people).",
        ],
        "defaults": {
            "buffer_days": 7,
            "geo": "US",
            "max_queries": 200,
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {meta_path}")
    print(f"Queried: {query_count} keywords")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
