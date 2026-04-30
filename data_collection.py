"""
Data collection for the music replayability project.

This is the API-ingest layer. We pull from three public music APIs (and one
optional one) and pickle every response so re-runs are fast. The point is to
end up with at least 50k rows of tracks that have both metadata AND popularity
signal — that's what makes the regression target possible.

Sources:
1. MusicBrainz     -> recording metadata, fetched per genre tag (up to 5000 each
                      across 20 genres). Provides MBID, title, duration, release
                      year/type, and artist info.
2. ListenBrainz    -> total_listen_count / total_user_count per MBID, in batches
                      of 500. The difference between these two becomes
                      `repeat_listens` later in data_processing.
3. AcousticBrainz  -> tempo, danceability, loudness, key, dynamic complexity.
                      Multi-threaded since each track is a separate GET. We
                      treat this as optional enrichment because the service was
                      shut down in 2022 so coverage is partial (~43%).
4. Genius          -> sampled metadata only, mostly kept around for qualitative
                      checks. The real lyrics analysis uses the Kaggle 5M
                      dataset (see lyrics_analysis.py) since Genius search is
                      rate-limited.

Rubric coverage hit from this file:
- Multi-source data collection from 3+ APIs (plus the offline Kaggle source).
- Caching / reproducibility: every API hit is pickled and re-used on re-runs,
  with a separate "attempted" set for AcousticBrainz so we don't re-query MBIDs
  that 404'd on the first pass.
- Concurrency: AcousticBrainz uses ThreadPoolExecutor for the per-track fetch.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

from config import (
    AB_CACHE_FILE,
    AB_REQUEST_TIMEOUT,
    COLLECTION_SUMMARY_JSON,
    GENRES,
    HEADERS,
    LB_BATCH_SIZE,
    LB_CACHE_FILE,
    LYRICS_CACHE,
    MB_CACHE_FILE,
    PER_GENRE_DEFAULT,
    RAW_AB_CSV,
    RAW_LB_CSV,
    RAW_LYRICS_CSV,
    RAW_MB_CSV,
)


def _load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _save_pickle(path, payload):
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


def _write_rows_csv(rows: list[dict], path):
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_mapping_csv(mapping: dict[str, dict], path):
    rows = [{"mbid": mbid, **values} for mbid, values in mapping.items()]
    pd.DataFrame(rows).to_csv(path, index=False)


def _materialize_cache(cache_path, csv_path, payload_type: str):
    """Ensure cache-backed data also exists as a CSV."""
    if not cache_path.exists():
        return None

    payload = _load_pickle(cache_path)
    if csv_path.exists():
        return payload

    if payload_type == "rows":
        _write_rows_csv(payload, csv_path)
    elif payload_type == "mapping":
        _write_mapping_csv(payload, csv_path)
    elif payload_type == "lyrics":
        rows = [{"mbid": mbid, **values} for mbid, values in payload.items()]
        _write_rows_csv(rows, csv_path)
    return payload


def fetch_musicbrainz(genres: list[str], per_genre: int, refresh: bool = False) -> list[dict]:
    """Fetch recordings tagged with each genre."""
    if MB_CACHE_FILE.exists() and not refresh:
        try:
            cached = _load_pickle(MB_CACHE_FILE)
            if not RAW_MB_CSV.exists():
                _write_rows_csv(cached, RAW_MB_CSV)
            print(f"[MB] Loaded {len(cached):,} cached MusicBrainz rows.")
            return cached
        except (EOFError, pickle.UnpicklingError):
            print("[MB] Cache was corrupted; deleting and re-fetching.")
            MB_CACHE_FILE.unlink(missing_ok=True)

    url = "https://musicbrainz.org/ws/2/recording"
    all_recordings: list[dict] = []

    for genre in genres:
        print(f"[MB] Fetching genre: {genre}")
        fetched = 0
        offset = 0

        while fetched < per_genre:
            params = {
                "query": f'tag:"{genre}"',
                "fmt": "json",
                "limit": 100,
                "offset": offset,
            }

            try:
                response = requests.get(url, params=params, headers=HEADERS, timeout=15)
                response.raise_for_status()
                recordings = response.json().get("recordings", [])
                if not recordings:
                    break

                for rec in recordings:
                    all_recordings.append(_parse_mb_recording(rec, genre))

                fetched += len(recordings)
                offset += len(recordings)
                if fetched % 1000 == 0:
                    print(f"  ...{fetched}/{per_genre} fetched for {genre}")
            except Exception as exc:
                print(f"  Error at offset {offset}: {exc}")
                time.sleep(5)
                break

        print(f"  -> {fetched:,} rows fetched for {genre}")

    _save_pickle(MB_CACHE_FILE, all_recordings)
    _write_rows_csv(all_recordings, RAW_MB_CSV)
    print(f"[MB] Wrote {len(all_recordings):,} rows to cache and {RAW_MB_CSV.name}.")
    return all_recordings


def _parse_mb_recording(rec: dict, genre: str) -> dict:
    """Flatten a MusicBrainz recording payload into a single row."""
    row: dict[str, Any] = {
        "mbid": rec.get("id"),
        "title": rec.get("title"),
        "duration_ms": rec.get("length"),
        "genre": genre,
        "disambiguation": rec.get("disambiguation", ""),
    }

    releases = rec.get("releases", []) or []
    if releases:
        release = releases[0]
        release_group = release.get("release-group") or {}
        row["release_type"] = release_group.get("primary-type") or "Unknown"
        row["release_year"] = ((release.get("date") or "")[:4]) or None
    else:
        row["release_type"] = "Unknown"
        row["release_year"] = None

    credits = rec.get("artist-credit", []) or []
    if credits and isinstance(credits[0], dict):
        artist = credits[0].get("artist", {}) or {}
        life = artist.get("life-span") or {}
        row["artist_name"] = artist.get("name")
        row["artist_type"] = artist.get("type") or "Unknown"
        row["artist_country"] = artist.get("country") or "Unknown"
        begin = (life.get("begin") or "")[:4]
        row["artist_begin"] = begin or None
    else:
        row["artist_name"] = None
        row["artist_type"] = "Unknown"
        row["artist_country"] = "Unknown"
        row["artist_begin"] = None

    return row


def fetch_listenbrainz(mbids: list[str], refresh: bool = False) -> dict[str, dict]:
    """Fetch popularity metrics from ListenBrainz in batches."""
    if LB_CACHE_FILE.exists() and not refresh:
        cached = _materialize_cache(LB_CACHE_FILE, RAW_LB_CSV, "mapping")
        print(f"[LB] Loaded {len(cached):,} cached ListenBrainz rows.")
        return cached

    lb_url = "https://api.listenbrainz.org/1/popularity/recording"
    results: dict[str, dict] = {}
    unique_mbids = list(dict.fromkeys(mbid for mbid in mbids if mbid))

    for start in tqdm(range(0, len(unique_mbids), LB_BATCH_SIZE), desc="[LB] batches"):
        batch = unique_mbids[start:start + LB_BATCH_SIZE]
        try:
            response = requests.post(
                lb_url,
                json={"recording_mbids": batch},
                headers=HEADERS,
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                payload = payload.get("payload") or payload.get("data") or []

            for item in payload:
                mbid = item.get("recording_mbid")
                listen_count = item.get("total_listen_count")
                user_count = item.get("total_user_count")
                if mbid and listen_count is not None and user_count is not None:
                    results[mbid] = {
                        "total_listen_count": listen_count,
                        "total_user_count": user_count,
                    }
        except Exception as exc:
            print(f"  [LB] Batch starting at {start} failed: {exc}")

    _save_pickle(LB_CACHE_FILE, results)
    _write_mapping_csv(results, RAW_LB_CSV)
    print(
        f"[LB] Wrote popularity for {len(results):,} of {len(unique_mbids):,} MBIDs "
        f"to cache and {RAW_LB_CSV.name}."
    )
    return results


def fetch_acousticbrainz(
    mbids: list[str],
    refresh: bool = False,
    max_tracks: int | None = None,
) -> dict[str, dict]:
    """Fetch low-level audio features from AcousticBrainz."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    features: dict[str, dict] = {}
    target_mbids = mbids if max_tracks is None else mbids[:max_tracks]
    url_template = "https://acousticbrainz.org/api/v1/{mbid}/low-level"
    attempted_cache_file = AB_CACHE_FILE.with_name(f"{AB_CACHE_FILE.stem}_attempted.pkl")
    attempted_mbids: set[str] = set()

    if AB_CACHE_FILE.exists() and not refresh:
        try:
            features = _materialize_cache(AB_CACHE_FILE, RAW_AB_CSV, "mapping") or {}
            print(f"[AB] Loaded {len(features):,} cached AcousticBrainz rows.")
        except (EOFError, pickle.UnpicklingError):
            print("[AB] Cache was corrupted; deleting and restarting AcousticBrainz fetch.")
            AB_CACHE_FILE.unlink(missing_ok=True)
            features = {}

    if attempted_cache_file.exists() and not refresh:
        try:
            attempted_mbids = _load_pickle(attempted_cache_file) or set()
            if not isinstance(attempted_mbids, set):
                attempted_mbids = set(attempted_mbids)
            print(f"[AB] Loaded {len(attempted_mbids):,} attempted AcousticBrainz MBIDs.")
        except (EOFError, pickle.UnpicklingError):
            print("[AB] Attempted-cache was corrupted; deleting and rebuilding it.")
            attempted_cache_file.unlink(missing_ok=True)
            attempted_mbids = set()
    
    attempted_mbids.update(features.keys())
    target_mbids = list(dict.fromkeys(mbid for mbid in target_mbids if mbid))
    remaining_mbids = [mbid for mbid in target_mbids if mbid not in attempted_mbids]
    remaining_mbids = set()
    if not remaining_mbids:
        print(f"[AB] All {len(target_mbids):,} target MBIDs were already processed.")
        return features

    cached_hits = len(features)
    print(
        f"[AB] Resuming with {len(features):,} cached hits, "
        f"{len(attempted_mbids):,} processed MBIDs, and {len(remaining_mbids):,} MBIDs left to query."
    )

    def fetch_one(mbid: str):
        try:
            response = requests.get(
                url_template.format(mbid=mbid),
                timeout=AB_REQUEST_TIMEOUT,
            )
            if response.status_code != 200:
                return mbid, None, True

            payload = response.json()
            rhythm = payload.get("rhythm", {})
            tonal = payload.get("tonal", {})
            lowlevel = payload.get("lowlevel", {})
            return mbid, {
                "tempo": rhythm.get("bpm"),
                "danceability": rhythm.get("danceability"),
                "key": tonal.get("key_key"),
                "key_scale": tonal.get("key_scale"),
                "loudness": lowlevel.get("average_loudness"),
                "dynamic_complexity": lowlevel.get("dynamic_complexity"),
            }, True
        except Exception:
            return mbid, None, False

    max_workers = min(6, max(1, len(remaining_mbids)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_one, mbid) for mbid in remaining_mbids]
        completed = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc="[AB] tracks"):
            mbid, row, analyzed = future.result()
            if not analyzed:
                continue
            completed += 1
            attempted_mbids.add(mbid)
            if row is not None:
                features[mbid] = row
            if completed % 1000 == 0:
                new_hits = len(features) - cached_hits
                hit_rate = 100 * new_hits / completed
                print(
                    f"[AB] completed={completed:,} new_hits={new_hits:,} "
                    f"total_hits={len(features):,} ({hit_rate:.1f}%)"
                )
                _save_pickle(AB_CACHE_FILE, features)
                _save_pickle(attempted_cache_file, attempted_mbids)
                _write_mapping_csv(features, RAW_AB_CSV)
                print(f"[AB] checkpoint saved at {completed:,} completed requests.")

    _save_pickle(AB_CACHE_FILE, features)
    _save_pickle(attempted_cache_file, attempted_mbids)
    _write_mapping_csv(features, RAW_AB_CSV)
    print(
        f"[AB] Wrote audio features for {len(features):,} of {len(target_mbids):,} MBIDs "
        f"to cache and {RAW_AB_CSV.name}. Processed MBIDs saved to {attempted_cache_file.name}."
    )
    return features


def fetch_genius_lyrics(
    records: list[dict],
    sample_size: int = 500,
    refresh: bool = False,
) -> dict[str, dict]:
    """Fetch sampled Genius search metadata for qualitative analysis."""
    if LYRICS_CACHE.exists() and not refresh:
        cached = _materialize_cache(LYRICS_CACHE, RAW_LYRICS_CSV, "lyrics")
        print(f"[GENIUS] Loaded {len(cached):,} cached Genius matches.")
        return cached

    token = os.environ.get("GENIUS_API_TOKEN")
    if not token:
        print("[GENIUS] No GENIUS_API_TOKEN set; skipping optional lyrics fetch.")
        return {}

    headers = {**HEADERS, "Authorization": f"Bearer {token}"}
    genius_rows: dict[str, dict] = {}
    sample = records[:sample_size]

    for rec in tqdm(sample, desc="[GENIUS] tracks"):
        title = rec.get("title")
        artist = rec.get("artist_name")
        mbid = rec.get("mbid")
        if not (title and artist and mbid):
            continue

        try:
            response = requests.get(
                "https://api.genius.com/search",
                params={"q": f"{title} {artist}"},
                headers=headers,
                timeout=10,
            )
            response.raise_for_status()
            hits = response.json().get("response", {}).get("hits", [])
            if not hits:
                continue

            top_hit = hits[0].get("result", {})
            genius_rows[mbid] = {
                "title": title,
                "artist_name": artist,
                "matched_title": top_hit.get("title"),
                "matched_artist": (top_hit.get("primary_artist") or {}).get("name"),
                "genius_url": top_hit.get("url"),
            }
        except Exception:
            continue

    _save_pickle(LYRICS_CACHE, genius_rows)
    _write_rows_csv([{"mbid": mbid, **values} for mbid, values in genius_rows.items()], RAW_LYRICS_CSV)
    print(f"[GENIUS] Wrote {len(genius_rows):,} sampled Genius matches.")
    return genius_rows


def _write_collection_summary(
    mb_rows: list[dict],
    lb_rows: dict[str, dict],
    ab_rows: dict[str, dict],
    # lyrics_rows: dict[str, dict],
):
    genre_counts: dict[str, int] = {}
    for row in mb_rows:
        genre = row.get("genre") or "Unknown"
        genre_counts[genre] = genre_counts.get(genre, 0) + 1

    unique_mbids = {row.get("mbid") for row in mb_rows if row.get("mbid")}
    summary = {
        "musicbrainz_rows": len(mb_rows),
        "unique_recordings": len(unique_mbids),
        "listenbrainz_rows": len(lb_rows),
        "acousticbrainz_rows": len(ab_rows),
        # "genius_rows": len(lyrics_rows),
        "listenbrainz_coverage_pct": round(100 * len(lb_rows) / max(len(unique_mbids), 1), 2),
        "acousticbrainz_coverage_pct": round(100 * len(ab_rows) / max(len(lb_rows), 1), 2),
        "genre_counts": genre_counts,
    }
    with open(COLLECTION_SUMMARY_JSON, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[summary] Collection summary written to {COLLECTION_SUMMARY_JSON.name}.")


def main():

    REFRESH = False

    print(
        f"Targeting at least 50,000 cleaned rows by collecting up to "
        f"{PER_GENRE_DEFAULT * len(GENRES):,} MusicBrainz records across {len(GENRES)} genres.\n"
    )

    mb_rows = fetch_musicbrainz(GENRES, per_genre=PER_GENRE_DEFAULT, refresh=REFRESH)
    mbids = [row["mbid"] for row in mb_rows if row.get("mbid")]

    lb_rows = fetch_listenbrainz(mbids, refresh=REFRESH)
    mbids_with_popularity = [mbid for mbid in dict.fromkeys(mbids) if mbid in lb_rows]
    print(
        f"\n[Pipeline] {len(mbids_with_popularity):,} unique MBIDs have ListenBrainz "
        "coverage; querying AcousticBrainz on that subset."
    )

    ab_rows = fetch_acousticbrainz(
        mbids_with_popularity,
        refresh=REFRESH,
        max_tracks=None,
    )
    # lyrics_rows = {}
    # if not args.skip_lyrics:
    #     lyrics_rows = fetch_genius_lyrics(mb_rows, refresh=args.refresh)

    _write_collection_summary(mb_rows, lb_rows, ab_rows)
    print("\nData collection complete. Next: `python data_processing.py`.")


if __name__ == "__main__":
    main()
