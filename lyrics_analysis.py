"""
Lyrics analysis for music replayability.

This is the NLP half of the project. We pair every track in our processed
dataset (where possible) with its lyrics from the Kaggle 5M Song Lyrics CSV,
extract a small set of text features per song, and then compare what shows up
in the high-replay quartile vs the low-replay quartile.

How the matching works (this is the "entity linking" piece):
- The Kaggle CSV is 9.2 GB and has no MBID column, so we can't just join.
- We normalize artist + title on both sides (lowercase, strip punctuation,
  collapse whitespace) and then do an exact lookup. The match is cached as a
  pickle so the 9.2 GB file gets streamed exactly once even on re-runs.
- ~37% match rate (20,759 songs out of ~55k tracks). That's the cross-dataset
  reconciliation moment — internal MBID-keyed data linking to an external
  source that doesn't share a key.

Per-song features computed:
- VADER sentiment (compound, positive, negative, neutral).
- Type-token ratio for vocabulary richness.
- Repetitiveness as the duplicate-line fraction.
- Flesch reading ease + Flesch-Kincaid grade level.
- Word/line counts, average word length, average line length.
- Rare-word ratio via the `wordfreq` corpus.

Aggregate comparisons:
- Top-30 word frequency comparison, high-replay vs low-replay quartile.
- Top-30 bigram comparison (same split).
- 8-topic LDA model to surface latent lyrical themes (relationship/nocturnal/
  spiritual/etc.) and per-quartile topic distribution.
- Lyrics-only and combined (lyrics + metadata) Ridge / GBM regressions to test
  whether NLP features add lift on top of metadata.
"""
from __future__ import annotations

import os
import pickle
import re
import string
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from config import (
    AUDIO_NUMERIC_FEATURES,
    CACHE_DIR,
    NUMERIC_FEATURES,
    OUTPUT_DIR,
    PROCESSED_CSV,
    RANDOM_STATE,
    RAW_TARGET,
    TARGET,
)

warnings.filterwarnings("ignore")

# ── user configuration ────────────────────────────────────────────────────────
# Point this at the downloaded Kaggle CSV before running.
KAGGLE_LYRICS_CSV = Path(
    os.environ.get("KAGGLE_LYRICS_CSV", "lyrics/ds2.csv")
)

N_TOPICS   = 8
CHUNK_SIZE = 50_000
RARE_FREQ_THRESHOLD = 1e-5  # words with English frequency below this are "rare"

LYRICS_CACHE = CACHE_DIR / "kaggle_lyrics_cache.pkl"

# ── output paths ──────────────────────────────────────────────────────────────
LYRICS_FEATURES_CSV  = OUTPUT_DIR / "lyrics_features.csv"
LYRICS_WORD_FREQ_CSV = OUTPUT_DIR / "lyrics_word_freq_comparison.csv"
LYRICS_BIGRAMS_CSV   = OUTPUT_DIR / "lyrics_bigrams_comparison.csv"
LYRICS_TOPIC_CSV     = OUTPUT_DIR / "lyrics_topic_summary.csv"
LYRICS_SUMMARY_MD    = OUTPUT_DIR / "lyrics_summary.md"

LYRICS_MODEL_CSV  = OUTPUT_DIR / "lyrics_model_results.csv"
LYRICS_MODEL_PLOT = OUTPUT_DIR / "lyrics11_model_comparison.png"

# NLP feature columns used in the lyrics-only and combined models
_LYRICS_FEAT_COLS = [
    "sentiment_compound", "sentiment_positive", "sentiment_negative",
    "type_token_ratio", "repetitiveness", "word_count",
    "avg_line_length", "avg_word_length", "rare_word_ratio",
]
_LYRICS_OPT_COLS  = ["flesch_reading_ease", "flesch_kincaid_grade"]
_META_CAT_ENC     = ["genre_enc", "release_type_enc", "artist_type_enc", "artist_country_enc"]

LYRICS_SENTIMENT_PLOT  = OUTPUT_DIR / "lyrics1_sentiment_by_quartile.png"
LYRICS_WORD_FREQ_PLOT  = OUTPUT_DIR / "lyrics2_top_words_comparison.png"
LYRICS_WORDCLOUD_PLOT  = OUTPUT_DIR / "lyrics3_wordclouds.png"
LYRICS_CORR_PLOT       = OUTPUT_DIR / "lyrics4_features_correlation.png"
LYRICS_TOPICS_PLOT     = OUTPUT_DIR / "lyrics5_topic_distribution.png"
LYRICS_COMPLEXITY_PLOT = OUTPUT_DIR / "lyrics6_complexity_vs_replay.png"
LYRICS_GENRE_SENT_PLOT = OUTPUT_DIR / "lyrics7_sentiment_by_genre.png"
LYRICS_BIGRAMS_PLOT    = OUTPUT_DIR / "lyrics8_bigrams_comparison.png"
LYRICS_RARITY_PLOT     = OUTPUT_DIR / "lyrics9_rarity_vs_replay.png"
LYRICS_WORDLEN_PLOT    = OUTPUT_DIR / "lyrics10_word_length_vs_replay.png"

# ── text helpers ──────────────────────────────────────────────────────────────

_STOPWORDS: set[str] | None = None


def _get_stopwords() -> set[str]:
    global _STOPWORDS
    if _STOPWORDS is None:
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords as sw
        _STOPWORDS = set(sw.words("english"))
        _STOPWORDS.update({
            "oh", "yeah", "na", "la", "ooh", "ah", "hey", "uh", "mm",
            "gonna", "wanna", "gotta", "ain", "em", "im", "ive",
            "chorus", "verse", "bridge", "outro", "intro", "hook",
            "repeat", "x2", "x3", "pre",
        })
    return _STOPWORDS


def _normalize(s: str) -> str:
    """Lowercase, strip punctuation/whitespace — used for artist/title matching."""
    s = s.lower().strip().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s).strip()


def _clean_lyrics(text: str) -> str:
    """Remove section headers and common dataset artifacts from raw lyrics."""
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"\d+\s+Contributors?.*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"You might also like.*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"Embed\s*$", " ", text, flags=re.IGNORECASE)
    return text.strip()


def _tokenize(text: str) -> list[str]:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    sw = _get_stopwords()
    return [t for t in text.split() if t not in sw and len(t) > 1 and not t.isdigit()]


def _bigrams(tokens: list[str]) -> list[tuple[str, str]]:
    return list(zip(tokens, tokens[1:]))


# wordfreq lookup with module-level cache to avoid redundant calls across songs
_WFREQ_CACHE: dict[str, float] = {}

def _word_freq(word: str) -> float:
    """Return English frequency from wordfreq (0..1). Falls back to 0 if unavailable."""
    if word not in _WFREQ_CACHE:
        try:
            from wordfreq import word_frequency
            _WFREQ_CACHE[word] = word_frequency(word, "en")
        except ImportError:
            _WFREQ_CACHE[word] = 0.0
    return _WFREQ_CACHE[word]


# ── 1. data loading ───────────────────────────────────────────────────────────

def load_lyrics_from_dataset(
    processed_df: pd.DataFrame,
    kaggle_path: Path = KAGGLE_LYRICS_CSV,
    refresh: bool = False,
    fuzzy_threshold: int = 85,
) -> dict[str, str]:
    """
    Match every song in processed_df against the Kaggle lyrics CSV.

    Matching strategy (in order):
      1. Exact normalized match on (artist, title).
      2. Exact artist + fuzzy title match via rapidfuzz token_sort_ratio.
         Also tries stripping/adding "the " from the artist name to handle
         "The Beatles" vs "Beatles" style mismatches.

    On the first run this scans the full Kaggle CSV and saves a pickle cache.
    Every subsequent run loads the cache instantly — the big CSV is not touched again.
    Pass refresh=True to force a re-scan.
    """
    if LYRICS_CACHE.exists() and not refresh:
        with open(LYRICS_CACHE, "rb") as fh:
            cached: dict[str, str] = pickle.load(fh)
        print(f"[LYRICS] Loaded {len(cached):,} cached lyrics from {LYRICS_CACHE.name}")
        return cached

    if not kaggle_path.exists():
        print(
            f"[LYRICS] Kaggle dataset not found at {kaggle_path}.\n"
            "  1. Download from: https://www.kaggle.com/datasets/nikhilnayak123/5-million-song-lyrics-dataset\n"
            f"  2. Place the CSV at {kaggle_path.resolve()}\n"
            "     (or set the KAGGLE_LYRICS_CSV env var to the actual path)"
        )
        return {}

    try:
        from rapidfuzz import fuzz as _fuzz
        from rapidfuzz import process as _rfp
        _has_rapidfuzz = True
    except ImportError:
        _has_rapidfuzz = False
        print("[LYRICS] rapidfuzz not installed; falling back to exact matching only.  pip install rapidfuzz")

    target = (
        processed_df
        .sort_values(TARGET, ascending=False)
        .dropna(subset=["mbid", "title", "artist_name"])
    )

    # exact lookup: (artist_norm, title_norm) -> mbid
    lookup: dict[tuple[str, str], str] = {
        (_normalize(r["artist_name"]), _normalize(r["title"])): r["mbid"]
        for _, r in target.iterrows()
    }

    # fuzzy fallback index: artist_norm -> [(title_norm, mbid), ...]
    # indexed under both "the x" and "x" forms to handle "the" prefix mismatches
    artist_to_songs: dict[str, list[tuple[str, str]]] = {}
    for _, r in target.iterrows():
        an = _normalize(r["artist_name"])
        tn = _normalize(r["title"])
        entry = (tn, r["mbid"])
        artist_to_songs.setdefault(an, []).append(entry)
        if an.startswith("the "):
            artist_to_songs.setdefault(an[4:], []).append(entry)
        else:
            artist_to_songs.setdefault(f"the {an}", []).append(entry)

    def _fuzzy_lookup(a_norm: str, t_norm: str) -> str | None:
        candidates = artist_to_songs.get(a_norm)
        if not candidates:
            # try flipping "the" on the Kaggle artist side
            alt = a_norm[4:] if a_norm.startswith("the ") else f"the {a_norm}"
            candidates = artist_to_songs.get(alt)
        if not candidates:
            return None
        titles = [c[0] for c in candidates]
        result = _rfp.extractOne(
            t_norm, titles,
            scorer=_fuzz.token_sort_ratio,
            score_cutoff=fuzzy_threshold,
        )
        return candidates[result[2]][1] if result else None

    matched: dict[str, str] = {}
    reader = pd.read_csv(
        kaggle_path,
        usecols=["title", "artist", "lyrics"],
        chunksize=CHUNK_SIZE,
        dtype=str,
        on_bad_lines="skip",
    )

    with tqdm(reader, desc="[LYRICS] scanning dataset", unit="chunk") as bar:
        for chunk in bar:
            chunk = chunk.dropna(subset=["title", "artist", "lyrics"])
            a_norms = chunk["artist"].map(_normalize).values
            t_norms = chunk["title"].map(_normalize).values
            lyrics_vals = chunk["lyrics"].values

            for a_norm, t_norm, lyrics in zip(a_norms, t_norms, lyrics_vals):
                mbid = lookup.get((a_norm, t_norm))
                if mbid is None and _has_rapidfuzz:
                    mbid = _fuzzy_lookup(a_norm, t_norm)
                if mbid and mbid not in matched:
                    matched[mbid] = _clean_lyrics(lyrics)

            bar.set_postfix(matched=f"{len(matched)}/{len(lookup)}")
            if len(matched) == len(lookup):
                break

    with open(LYRICS_CACHE, "wb") as fh:
        pickle.dump(matched, fh)

    print(f"[LYRICS] Matched {len(matched):,} / {len(lookup):,} songs — cached to {LYRICS_CACHE.name}")
    return matched


# ── 2. feature extraction ─────────────────────────────────────────────────────

def extract_features(lyrics_dict: dict[str, str]) -> pd.DataFrame:
    """Compute per-song text features from raw lyrics text."""
    nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()

    try:
        import textstat as ts
        _has_textstat = True
    except ImportError:
        _has_textstat = False
        print("[LYRICS] textstat not found; readability scores skipped.  pip install textstat")

    rows: list[dict] = []
    for mbid, text in tqdm(lyrics_dict.items(), desc="[LYRICS] extracting features", unit="song"):
        lines  = [ln for ln in text.splitlines() if ln.strip()]
        tokens = _tokenize(text)
        total_words  = len(tokens)
        unique_words = len(set(tokens))

        line_counts    = Counter(ln.strip().lower() for ln in lines)
        dup_lines      = sum(v - 1 for v in line_counts.values() if v > 1)
        repetitiveness = dup_lines / max(len(lines), 1)
        ttr            = unique_words / max(total_words, 1)
        sent           = sia.polarity_scores(text)

        # bigrams: top 5 as "word1 word2" strings
        bigram_counts = Counter(_bigrams(tokens))
        top_bigrams   = [f"{a} {b}" for (a, b), _ in bigram_counts.most_common(5)]

        # word length
        word_lengths  = [len(t) for t in tokens]
        avg_word_len  = round(sum(word_lengths) / max(total_words, 1), 3)
        sorted_unique = sorted(set(tokens), key=len, reverse=True)
        longest_words = sorted_unique[:10]  # top 10 by character count

        # rarity: words whose English frequency is below the threshold
        rare_tokens  = [t for t in tokens if _word_freq(t) < RARE_FREQ_THRESHOLD]
        rare_count   = len(rare_tokens)
        rare_ratio   = round(rare_count / max(total_words, 1), 4)
        rarest_words = sorted(set(rare_tokens), key=_word_freq)[:10]  # 10 least frequent

        row: dict = {
            "mbid":               mbid,
            "word_count":         total_words,
            "unique_word_count":  unique_words,
            "type_token_ratio":   round(ttr, 4),
            "line_count":         len(lines),
            "avg_line_length":    round(
                sum(len(ln.split()) for ln in lines) / max(len(lines), 1), 2
            ),
            "repetitiveness":     round(repetitiveness, 4),
            "sentiment_compound": round(sent["compound"], 4),
            "sentiment_positive": round(sent["pos"], 4),
            "sentiment_negative": round(sent["neg"], 4),
            "sentiment_neutral":  round(sent["neu"], 4),
            "avg_word_length":    avg_word_len,
            "rare_word_count":    rare_count,
            "rare_word_ratio":    rare_ratio,
            "top_bigrams":        "; ".join(top_bigrams),
            "longest_words":      "; ".join(longest_words),
            "rarest_words":       "; ".join(rarest_words),
        }

        if _has_textstat:
            try:
                row["flesch_reading_ease"]  = ts.flesch_reading_ease(text)
                row["flesch_kincaid_grade"] = ts.flesch_kincaid_grade(text)
            except Exception:
                row["flesch_reading_ease"]  = None
                row["flesch_kincaid_grade"] = None

        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"[LYRICS] Extracted features for {len(df):,} tracks.")
    return df


# ── 3. word frequency comparison ──────────────────────────────────────────────

def compute_word_frequencies(
    lyrics_dict: dict[str, str],
    replay_map: dict[str, float],
    top_n: int = 200,
) -> pd.DataFrame:
    """
    Compare word frequencies between the top-25 % and bottom-25 % replay songs.
    Returns a DataFrame sorted by high/low frequency ratio.
    """
    values   = [v for v in replay_map.values() if v is not None]
    high_cut = np.percentile(values, 75)
    low_cut  = np.percentile(values, 25)

    high_counter: Counter = Counter()
    low_counter:  Counter = Counter()

    for mbid, text in tqdm(lyrics_dict.items(), desc="[LYRICS] word frequencies", unit="song"):
        replay = replay_map.get(mbid)
        if replay is None:
            continue
        tokens = _tokenize(text)
        if replay >= high_cut:
            high_counter.update(tokens)
        elif replay <= low_cut:
            low_counter.update(tokens)

    high_total = max(sum(high_counter.values()), 1)
    low_total  = max(sum(low_counter.values()), 1)

    candidates = (
        {w for w, _ in high_counter.most_common(top_n)} |
        {w for w, _ in low_counter.most_common(top_n)}
    )
    rows = [
        {
            "word":        w,
            "high_freq":   high_counter[w] / high_total,
            "low_freq":    low_counter[w]  / low_total,
            "high_count":  high_counter[w],
            "low_count":   low_counter[w],
            "freq_ratio":  (high_counter[w] / high_total) / (low_counter[w] / low_total + 1e-9),
            "total_count": high_counter[w] + low_counter[w],
        }
        for w in candidates
    ]

    df = (
        pd.DataFrame(rows)
        .query("total_count >= 5")
        .sort_values("freq_ratio", ascending=False)
        .reset_index(drop=True)
    )
    df.to_csv(LYRICS_WORD_FREQ_CSV, index=False)
    print(f"[LYRICS] Word-frequency comparison -> {LYRICS_WORD_FREQ_CSV.name}")
    return df


# ── 3b. bigram frequency comparison ──────────────────────────────────────────

def compute_bigram_frequencies(
    lyrics_dict: dict[str, str],
    replay_map: dict[str, float],
    top_n: int = 150,
) -> pd.DataFrame:
    """
    Compare 2-word phrase frequencies between top-25 % and bottom-25 % replay songs.
    Returns a DataFrame sorted by high/low frequency ratio.
    """
    values   = [v for v in replay_map.values() if v is not None]
    high_cut = np.percentile(values, 75)
    low_cut  = np.percentile(values, 25)

    high_counter: Counter = Counter()
    low_counter:  Counter = Counter()

    for mbid, text in tqdm(lyrics_dict.items(), desc="[LYRICS] bigram frequencies", unit="song"):
        replay = replay_map.get(mbid)
        if replay is None:
            continue
        tokens = _tokenize(text)
        pairs  = [f"{a} {b}" for a, b in _bigrams(tokens)]
        if replay >= high_cut:
            high_counter.update(pairs)
        elif replay <= low_cut:
            low_counter.update(pairs)

    high_total = max(sum(high_counter.values()), 1)
    low_total  = max(sum(low_counter.values()), 1)

    candidates = (
        {p for p, _ in high_counter.most_common(top_n)} |
        {p for p, _ in low_counter.most_common(top_n)}
    )
    rows = [
        {
            "bigram":      p,
            "high_freq":   high_counter[p] / high_total,
            "low_freq":    low_counter[p]  / low_total,
            "high_count":  high_counter[p],
            "low_count":   low_counter[p],
            "freq_ratio":  (high_counter[p] / high_total) / (low_counter[p] / low_total + 1e-9),
            "total_count": high_counter[p] + low_counter[p],
        }
        for p in candidates
    ]

    df = (
        pd.DataFrame(rows)
        .query("total_count >= 3")
        .sort_values("freq_ratio", ascending=False)
        .reset_index(drop=True)
    )
    df.to_csv(LYRICS_BIGRAMS_CSV, index=False)
    print(f"[LYRICS] Bigram comparison -> {LYRICS_BIGRAMS_CSV.name}")
    return df


# ── 4. LDA topic model ────────────────────────────────────────────────────────

def run_lda(
    lyrics_dict: dict[str, str],
    n_topics: int = N_TOPICS,
) -> tuple[pd.DataFrame, list[list[str]]]:
    """
    Fit an LDA topic model over all lyrics.
    Returns (per-song topic-weight DataFrame, list of top-word lists per topic).
    """
    mbids = list(lyrics_dict.keys())
    texts = [" ".join(_tokenize(t)) for t in lyrics_dict.values()]

    vec    = CountVectorizer(max_features=3000, min_df=3, max_df=0.80)
    dtm    = vec.fit_transform(texts)
    lda    = LatentDirichletAllocation(
        n_components=n_topics, random_state=RANDOM_STATE,
        max_iter=25, learning_method="online", n_jobs=1,
    )
    weights = lda.fit_transform(dtm)

    vocab       = vec.get_feature_names_out()
    topic_words = [
        [vocab[i] for i in topic.argsort()[:-16:-1]]
        for topic in lda.components_
    ]

    topic_df = pd.DataFrame(weights, columns=[f"topic_{i}" for i in range(n_topics)])
    topic_df.insert(0, "mbid", mbids)
    topic_df["dominant_topic"] = weights.argmax(axis=1)
    topic_df.to_csv(LYRICS_TOPIC_CSV, index=False)

    print(f"[LYRICS] LDA ({n_topics} topics) fitted.")
    for i, words in enumerate(topic_words):
        print(f"  Topic {i}: {', '.join(words[:8])}")
    return topic_df, topic_words


# ── 5. plots ──────────────────────────────────────────────────────────────────

def _plot_sentiment_by_quartile(merged_df: pd.DataFrame) -> None:
    df = merged_df.dropna(subset=["sentiment_compound", TARGET]).copy()
    df["replay_quartile"] = pd.qcut(
        df[TARGET], q=4, labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"]
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Sentiment vs. Replayability", fontsize=14, fontweight="bold")

    sns.violinplot(
        data=df, x="replay_quartile", y="sentiment_compound",
        palette="RdYlGn", ax=axes[0], inner="quartile",
    )
    axes[0].axhline(0, color="grey", lw=0.8, ls="--")
    axes[0].set_title("VADER Compound Score by Replay Quartile")
    axes[0].set_xlabel("Replay Quartile")
    axes[0].set_ylabel("Compound Score (−1 -> +1)")

    cols  = ["sentiment_positive", "sentiment_negative", "sentiment_neutral"]
    means = df.groupby("replay_quartile")[cols].mean()
    means.plot(
        kind="bar", stacked=True, ax=axes[1],
        color=["#2ecc71", "#e74c3c", "#bdc3c7"], width=0.6, edgecolor="white",
    )
    axes[1].set_title("Sentiment Composition by Replay Quartile")
    axes[1].set_xlabel("Replay Quartile")
    axes[1].set_ylabel("Mean Proportion")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    axes[1].legend(["Positive", "Negative", "Neutral"], loc="upper right", fontsize=8)

    plt.tight_layout()
    fig.savefig(LYRICS_SENTIMENT_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[LYRICS] Saved {LYRICS_SENTIMENT_PLOT.name}")


def _plot_word_frequency(word_freq_df: pd.DataFrame) -> None:
    top_high = word_freq_df.nlargest(20, "freq_ratio")
    top_low  = word_freq_df.nsmallest(20, "freq_ratio")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Words That Distinguish High vs. Low Replay Songs", fontsize=14, fontweight="bold")

    axes[0].barh(top_high["word"], top_high["high_freq"] * 1e4, color="#1f6f78")
    axes[0].set_title("More Common in High-Replay Songs\n(top 20 by frequency ratio)")
    axes[0].set_xlabel("Occurrences per 10,000 words")
    axes[0].invert_yaxis()

    axes[1].barh(top_low["word"], top_low["low_freq"] * 1e4, color="#e07a5f")
    axes[1].set_title("More Common in Low-Replay Songs\n(top 20 by inverse frequency ratio)")
    axes[1].set_xlabel("Occurrences per 10,000 words")
    axes[1].invert_yaxis()

    plt.tight_layout()
    fig.savefig(LYRICS_WORD_FREQ_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[LYRICS] Saved {LYRICS_WORD_FREQ_PLOT.name}")


def _plot_word_clouds(word_freq_df: pd.DataFrame) -> None:
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("[LYRICS] wordcloud not installed; skipping.  pip install wordcloud")
        return

    high_freq = {w: c for w, c in zip(word_freq_df["word"], word_freq_df["high_count"]) if c > 0}
    low_freq  = {w: c for w, c in zip(word_freq_df["word"], word_freq_df["low_count"])  if c > 0}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Lyric Word Clouds by Replayability", fontsize=14, fontweight="bold")

    for ax, freq, title, cmap in [
        (axes[0], high_freq, "High-Replay Songs", "YlGn"),
        (axes[1], low_freq,  "Low-Replay Songs",  "OrRd"),
    ]:
        if not freq:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(title)
            ax.axis("off")
            continue
        wc = WordCloud(
            width=800, height=400, background_color="white",
            colormap=cmap, max_words=80,
        ).generate_from_frequencies(freq)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(LYRICS_WORDCLOUD_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[LYRICS] Saved {LYRICS_WORDCLOUD_PLOT.name}")


def _plot_feature_correlations(merged_df: pd.DataFrame) -> None:
    feature_cols = [
        "word_count", "unique_word_count", "type_token_ratio",
        "line_count", "avg_line_length", "repetitiveness",
        "sentiment_compound", "sentiment_positive", "sentiment_negative",
    ]
    if "flesch_reading_ease" in merged_df.columns:
        feature_cols += ["flesch_reading_ease", "flesch_kincaid_grade"]

    cols = [c for c in feature_cols if c in merged_df.columns] + [TARGET]
    corr = merged_df[cols].dropna().corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        linewidths=0.5, square=True, ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Lyrics Feature Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(LYRICS_CORR_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[LYRICS] Saved {LYRICS_CORR_PLOT.name}")


_TOPIC_NAMES = [
    "Foreign Language",
    "Conversational",
    "Romance",
    "Hip-Hop / Street",
    "Spiritual / Gospel",
    "Country / Blues",
    "Party / Dance",
    "Reflective / Longing",
]

_TOPIC_COLORS = [
    "#5C6BC0",  # T0 Foreign Language
    "#E64A19",  # T1 Conversational
    "#D81B60",  # T2 Romance
    "#558B2F",  # T3 Hip-Hop / Street
    "#F57F17",  # T4 Spiritual / Gospel
    "#795548",  # T5 Country / Blues
    "#00838F",  # T6 Party / Dance
    "#6A1B9A",  # T7 Reflective / Longing
]


def _plot_topic_distribution(
    topic_df: pd.DataFrame,
    topic_words: list[list[str]],
    replay_map: dict[str, float],
) -> None:
    import matplotlib.colors as mcolors
    from matplotlib.gridspec import GridSpec

    n_topics = len(topic_words)
    names  = (_TOPIC_NAMES  + [f"Topic {i}" for i in range(n_topics)])[:n_topics]
    colors = (_TOPIC_COLORS + ["#888888"]   * n_topics)[:n_topics]

    # ── Per-topic stats ──────────────────────────────────────────────────────────
    df = topic_df.copy()
    df["log_repeat_listens"] = df["mbid"].map(replay_map)
    df = df.dropna(subset=["log_repeat_listens"])

    topic_cols = [f"topic_{i}" for i in range(n_topics)]
    q75 = np.percentile(df["log_repeat_listens"], 75)
    q25 = np.percentile(df["log_repeat_listens"], 25)
    diff = (
        df[df["log_repeat_listens"] >= q75][topic_cols].mean().values
        - df[df["log_repeat_listens"] <= q25][topic_cols].mean().values
    )
    mean_replay = np.array([
        df.loc[df["dominant_topic"] == i, "log_repeat_listens"].mean()
        if (df["dominant_topic"] == i).sum() > 0 else 0.0
        for i in range(n_topics)
    ])
    song_counts = [(df["dominant_topic"] == i).sum() for i in range(n_topics)]

    # ── Figure ───────────────────────────────────────────────────────────────────
    BG      = "#F8F9FC"
    CARD_BG = "#FFFFFF"
    INK     = "#1E293B"
    DIMTEXT = "#64748B"
    GRID    = "#E2E8F0"

    fig = plt.figure(figsize=(20, 10), facecolor=BG)
    fig.suptitle(
        "Lyrical Topic Landscape  ·  What Themes Drive Replay?",
        fontsize=17, fontweight="bold", color=INK, y=0.97,
    )
    gs = GridSpec(
        2, 5, figure=fig,
        left=0.03, right=0.97, top=0.91, bottom=0.06,
        wspace=0.28, hspace=0.45,
    )

    # ── Topic cards (2 rows × 4 cols) ────────────────────────────────────────────
    for i in range(min(n_topics, 8)):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        c  = colors[i]

        ax.set_facecolor(CARD_BG)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])

        for side, sp in ax.spines.items():
            sp.set_color(c if side == "left" else GRID)
            sp.set_linewidth(4 if side == "left" else 0.8)

        # Header band
        ax.axhspan(0.80, 1.0, color=mcolors.to_rgba(c, 0.10), zorder=0)

        # Title row
        ax.text(0.08, 0.90, f"T{i}", transform=ax.transAxes,
                ha="left", va="center", fontsize=10, fontweight="bold",
                color=c, family="monospace")
        ax.text(0.26, 0.90, names[i], transform=ax.transAxes,
                ha="left", va="center", fontsize=9, fontweight="bold",
                color=INK)

        # Words (fading with rank)
        for j, word in enumerate(topic_words[i][:6]):
            ax.text(0.09, 0.72 - j * 0.115, word,
                    transform=ax.transAxes,
                    ha="left", va="top",
                    fontsize=max(9.5 - j * 0.3, 7.5),
                    color=INK, alpha=max(1.0 - j * 0.13, 0.40))

        # Footer
        ax.text(0.09, 0.06,
                f"{song_counts[i]:,} songs · avg replay {mean_replay[i]:.2f}",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=6.8, color=DIMTEXT)

    # ── Divergence chart (right column, spans both rows) ─────────────────────────
    ax_bar = fig.add_subplot(gs[:, 4])
    ax_bar.set_facecolor(CARD_BG)

    order          = np.argsort(diff)
    sorted_diff    = diff[order]
    sorted_labels  = [f"T{i}  {names[i]}" for i in order]
    sorted_colors  = [colors[i] for i in order]

    bars = ax_bar.barh(
        range(n_topics), sorted_diff,
        color=sorted_colors, edgecolor="none", height=0.65,
    )
    ax_bar.set_yticks(range(n_topics))
    ax_bar.set_yticklabels(sorted_labels, fontsize=8, color=INK)
    ax_bar.axvline(0, color=DIMTEXT, lw=0.9, alpha=0.6)
    ax_bar.set_title("Which topics drive replay?",
                     fontsize=10, fontweight="bold", color=INK, pad=10)
    ax_bar.set_xlabel("Δ mean topic weight\n(high − low replay quartile)",
                      fontsize=7.5, color=DIMTEXT)
    for sp in ax_bar.spines.values():
        sp.set_color(GRID)
    ax_bar.tick_params(axis="both", colors=DIMTEXT, labelsize=7.5)
    ax_bar.xaxis.set_tick_params(labelcolor=DIMTEXT)

    for bar, val in zip(bars, sorted_diff):
        ax_bar.text(
            val + (0.0002 if val >= 0 else -0.0002),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.4f}", va="center",
            ha="left" if val >= 0 else "right",
            fontsize=6.5, color=INK, alpha=0.80,
        )

    fig.savefig(LYRICS_TOPICS_PLOT, dpi=180, bbox_inches="tight",
                facecolor=BG)
    plt.close(fig)
    print(f"[LYRICS] Saved {LYRICS_TOPICS_PLOT.name}")


def _plot_complexity_vs_replay(merged_df: pd.DataFrame) -> None:
    pairs = [
        ("type_token_ratio",   "Vocabulary Richness (TTR)"),
        ("repetitiveness",     "Repetitiveness (dup-line ratio)"),
        ("word_count",         "Word Count"),
        ("sentiment_compound", "Sentiment Compound Score"),
    ]
    available = [(c, l) for c, l in pairs if c in merged_df.columns]
    if not available:
        return

    n   = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle("Lyrics Complexity vs. Replayability", fontsize=13, fontweight="bold")

    sample = merged_df.dropna(subset=[TARGET]).sample(
        min(1500, len(merged_df)), random_state=RANDOM_STATE
    )
    for ax, (col, label) in zip(axes, available):
        s = sample.dropna(subset=[col])
        ax.scatter(s[col], s[TARGET], alpha=0.22, s=12, color="#1f6f78")
        try:
            m, b = np.polyfit(s[col], s[TARGET], 1)
            xr = np.linspace(s[col].min(), s[col].max(), 100)
            ax.plot(xr, m * xr + b, color="#e07a5f", lw=2)
        except Exception:
            pass
        r = s[[col, TARGET]].corr().iloc[0, 1]
        ax.set_title(f"r = {r:.3f}", fontsize=11)
        ax.set_xlabel(label)
        ax.set_ylabel("log_repeat_listens")

    plt.tight_layout()
    fig.savefig(LYRICS_COMPLEXITY_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[LYRICS] Saved {LYRICS_COMPLEXITY_PLOT.name}")


def _plot_sentiment_by_genre(merged_df: pd.DataFrame) -> None:
    df = merged_df.dropna(subset=["sentiment_compound", "genre"]).copy()
    if df["genre"].nunique() < 2:
        return

    order = (
        df.groupby("genre")["sentiment_compound"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(
        data=df, x="genre", y="sentiment_compound",
        order=order, palette="coolwarm", ax=ax, width=0.55,
    )
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_title("Lyric Sentiment Distribution by Genre", fontsize=13, fontweight="bold")
    ax.set_xlabel("Genre")
    ax.set_ylabel("VADER Compound Score")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(LYRICS_GENRE_SENT_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[LYRICS] Saved {LYRICS_GENRE_SENT_PLOT.name}")


# ── 5b. new plots ─────────────────────────────────────────────────────────────

def _plot_bigrams(bigram_df: pd.DataFrame) -> None:
    top_high = bigram_df.nlargest(20, "freq_ratio")
    top_low  = bigram_df.nsmallest(20, "freq_ratio")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("2-Word Phrases That Distinguish High vs. Low Replay Songs",
                 fontsize=14, fontweight="bold")

    axes[0].barh(top_high["bigram"], top_high["high_freq"] * 1e4, color="#1f6f78")
    axes[0].set_title("More Common in High-Replay Songs\n(top 20 by frequency ratio)")
    axes[0].set_xlabel("Occurrences per 10,000 bigrams")
    axes[0].invert_yaxis()

    axes[1].barh(top_low["bigram"], top_low["low_freq"] * 1e4, color="#e07a5f")
    axes[1].set_title("More Common in Low-Replay Songs\n(top 20 by inverse frequency ratio)")
    axes[1].set_xlabel("Occurrences per 10,000 bigrams")
    axes[1].invert_yaxis()

    plt.tight_layout()
    fig.savefig(LYRICS_BIGRAMS_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[LYRICS] Saved {LYRICS_BIGRAMS_PLOT.name}")


def _plot_rarity_vs_replay(merged_df: pd.DataFrame) -> None:
    df = merged_df.dropna(subset=["rare_word_ratio", TARGET]).copy()
    if len(df) < 10:
        return

    df["replay_quartile"] = pd.qcut(
        df[TARGET], q=4, labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"]
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Lexical Rarity vs. Replayability", fontsize=14, fontweight="bold")

    sns.violinplot(
        data=df, x="replay_quartile", y="rare_word_ratio",
        palette="viridis", ax=axes[0], inner="quartile",
    )
    axes[0].set_title("Rare Word Ratio by Replay Quartile")
    axes[0].set_xlabel("Replay Quartile")
    axes[0].set_ylabel(f"Fraction of words with freq < {RARE_FREQ_THRESHOLD:.0e}")

    sample = df.sample(min(1500, len(df)), random_state=RANDOM_STATE)
    axes[1].scatter(sample["rare_word_ratio"], sample[TARGET],
                    alpha=0.25, s=14, color="#1f6f78")
    try:
        m, b = np.polyfit(sample["rare_word_ratio"], sample[TARGET], 1)
        xr = np.linspace(sample["rare_word_ratio"].min(), sample["rare_word_ratio"].max(), 100)
        axes[1].plot(xr, m * xr + b, color="#e07a5f", lw=2)
    except Exception:
        pass
    r = sample[["rare_word_ratio", TARGET]].corr().iloc[0, 1]
    axes[1].set_title(f"Rare Word Ratio vs. Replay  (r = {r:.3f})")
    axes[1].set_xlabel("Rare Word Ratio")
    axes[1].set_ylabel("log_repeat_listens")

    plt.tight_layout()
    fig.savefig(LYRICS_RARITY_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[LYRICS] Saved {LYRICS_RARITY_PLOT.name}")


def _plot_word_length_vs_replay(merged_df: pd.DataFrame) -> None:
    df = merged_df.dropna(subset=["avg_word_length", TARGET]).copy()
    if len(df) < 10:
        return

    df["replay_quartile"] = pd.qcut(
        df[TARGET], q=4, labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"]
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Word Length vs. Replayability", fontsize=14, fontweight="bold")

    sns.boxplot(
        data=df, x="replay_quartile", y="avg_word_length",
        palette="Blues", ax=axes[0], width=0.5,
    )
    axes[0].set_title("Average Word Length by Replay Quartile")
    axes[0].set_xlabel("Replay Quartile")
    axes[0].set_ylabel("Avg characters per word")

    sample = df.sample(min(1500, len(df)), random_state=RANDOM_STATE)
    axes[1].scatter(sample["avg_word_length"], sample[TARGET],
                    alpha=0.25, s=14, color="#355c7d")
    try:
        m, b = np.polyfit(sample["avg_word_length"], sample[TARGET], 1)
        xr = np.linspace(sample["avg_word_length"].min(), sample["avg_word_length"].max(), 100)
        axes[1].plot(xr, m * xr + b, color="#e07a5f", lw=2)
    except Exception:
        pass
    r = sample[["avg_word_length", TARGET]].corr().iloc[0, 1]
    axes[1].set_title(f"Avg Word Length vs. Replay  (r = {r:.3f})")
    axes[1].set_xlabel("Avg Word Length (chars)")
    axes[1].set_ylabel("log_repeat_listens")

    plt.tight_layout()
    fig.savefig(LYRICS_WORDLEN_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[LYRICS] Saved {LYRICS_WORDLEN_PLOT.name}")


# ── 6. lyrics model ──────────────────────────────────────────────────────────

def run_lyrics_model(
    features_df: pd.DataFrame,
    topic_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Train and compare three feature sets on the matched-lyrics subset:
      • Lyrics Only  — NLP features + LDA topic weights
      • Metadata Only — numeric/encoded metadata (fair baseline on same subset)
      • Combined      — all of the above

    Each feature set is evaluated with Ridge and GBM using 5-fold CV + holdout.
    Results are saved to lyrics_model_results.csv.
    """
    # load processed CSV with full metadata columns
    meta_load_cols = (
        ["mbid", TARGET]
        + [c for c in NUMERIC_FEATURES]
        + [c for c in AUDIO_NUMERIC_FEATURES]
        + _META_CAT_ENC
    )
    try:
        processed_df = pd.read_csv(
            PROCESSED_CSV,
            usecols=lambda c: c in meta_load_cols,
            dtype={"mbid": "string"},
        )
    except Exception as exc:
        print(f"[LYRICS MODEL] Could not load processed dataset: {exc}")
        return pd.DataFrame()

    # merge lyrics features + LDA topic weights
    topic_cols = [c for c in topic_df.columns if c.startswith("topic_")]
    df = (
        features_df
        .merge(topic_df[["mbid"] + topic_cols], on="mbid", how="left")
        .merge(processed_df, on="mbid", how="inner")
        .dropna(subset=[TARGET])
    )
    print(f"[LYRICS MODEL] {len(df):,} songs with lyrics features + metadata")

    lyric_cols    = [c for c in _LYRICS_FEAT_COLS + _LYRICS_OPT_COLS + topic_cols if c in df.columns]
    meta_cols     = [c for c in NUMERIC_FEATURES + AUDIO_NUMERIC_FEATURES + _META_CAT_ENC if c in df.columns]
    combined_cols = lyric_cols + [c for c in meta_cols if c not in lyric_cols]

    feature_sets = {
        "Lyrics Only":      lyric_cols,
        "Metadata Only":    meta_cols,
        "Lyrics + Metadata": combined_cols,
    }
    estimators = {
        "Ridge": Ridge(),
        "GBM":   GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE),
    }

    y   = df[TARGET].values
    idx = np.arange(len(df))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE)
    y_train, y_test = y[idx_train], y[idx_test]

    results: list[dict] = []
    for fs_name, cols in feature_sets.items():
        X = df[cols].values
        X_train, X_test = X[idx_train], X[idx_test]
        for model_name, estimator in estimators.items():
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model",   estimator),
            ])
            cv_r2   = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2")
            pipe.fit(X_train, y_train)
            y_pred  = pipe.predict(X_test)
            h_r2    = r2_score(y_test, y_pred)
            h_rmse  = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            results.append({
                "model":       f"{fs_name} — {model_name}",
                "feature_set": fs_name,
                "algorithm":   model_name,
                "cv_r2_mean":  round(float(cv_r2.mean()), 4),
                "cv_r2_std":   round(float(cv_r2.std()),  4),
                "holdout_r2":  round(h_r2,   4),
                "holdout_rmse": round(h_rmse, 4),
                "n_features":  len(cols),
                "n_songs":     len(df),
            })
            print(
                f"  [{fs_name} | {model_name}]  "
                f"CV R²={cv_r2.mean():.3f}±{cv_r2.std():.3f}  "
                f"Holdout R²={h_r2:.3f}  RMSE={h_rmse:.3f}"
            )

    results_df = pd.DataFrame(results)
    results_df.to_csv(LYRICS_MODEL_CSV, index=False)
    print(f"[LYRICS MODEL] Results -> {LYRICS_MODEL_CSV.name}")
    _plot_lyrics_model_comparison(results_df)
    return results_df


def _plot_lyrics_model_comparison(results_df: pd.DataFrame) -> None:
    order = ["Lyrics Only", "Metadata Only", "Lyrics + Metadata"]
    pivot_r2   = results_df.pivot(index="feature_set", columns="algorithm", values="holdout_r2").reindex(order)
    pivot_rmse = results_df.pivot(index="feature_set", columns="algorithm", values="holdout_rmse").reindex(order)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Lyrics Model — Holdout Performance by Feature Set", fontsize=14, fontweight="bold")

    colors = ["#1f6f78", "#e07a5f"]
    for ax, pivot, ylabel, title in [
        (axes[0], pivot_r2,   "R²",   "Holdout R²"),
        (axes[1], pivot_rmse, "RMSE", "Holdout RMSE"),
    ]:
        pivot.plot(kind="bar", ax=ax, color=colors, width=0.6, edgecolor="white")
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
        ax.legend(title="Algorithm")
    axes[0].axhline(0, color="black", lw=0.5, ls="--")

    plt.tight_layout()
    fig.savefig(LYRICS_MODEL_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[LYRICS] Saved {LYRICS_MODEL_PLOT.name}")


# ── 7. summary ─────────────────────────────────────────────────────────────────

def write_lyrics_summary(
    features_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    word_freq_df: pd.DataFrame,
    bigram_df: pd.DataFrame,
    topic_words: list[list[str]],
    model_results_df: pd.DataFrame | None = None,
) -> None:
    def _r(col: str) -> str:
        try:
            return f"{merged_df[[col, TARGET]].dropna().corr().iloc[0, 1]:.3f}"
        except Exception:
            return "n/a"

    top_high_words   = word_freq_df.nlargest(10, "freq_ratio")["word"].tolist()
    top_low_words    = word_freq_df.nsmallest(10, "freq_ratio")["word"].tolist()
    top_high_bigrams = bigram_df.nlargest(10, "freq_ratio")["bigram"].tolist()
    top_low_bigrams  = bigram_df.nsmallest(10, "freq_ratio")["bigram"].tolist()

    # Aggregate longest and rarest words across all songs
    all_longest = Counter()
    all_rarest  = Counter()
    for words in features_df["longest_words"].dropna():
        all_longest.update(w for w in words.split("; ") if w)
    for words in features_df["rarest_words"].dropna():
        all_rarest.update(w for w in words.split("; ") if w)

    lines = [
        "# Lyrics Analysis Summary",
        "",
        f"**Tracks analyzed:** {len(features_df):,}",
        f"**Average sentiment (VADER compound):** {features_df['sentiment_compound'].mean():.3f}",
        f"**Average repetitiveness:** {features_df['repetitiveness'].mean():.3f}",
        f"**Average vocabulary richness (TTR):** {features_df['type_token_ratio'].mean():.3f}",
        f"**Average word length:** {features_df['avg_word_length'].mean():.2f} chars",
        f"**Average rare word ratio:** {features_df['rare_word_ratio'].mean():.3f}",
        "",
        "## Correlations with log_repeat_listens",
        "",
        "| Feature | r |",
        "|---------|---|",
        f"| Repetitiveness (dup-line ratio) | {_r('repetitiveness')} |",
        f"| Vocabulary richness (TTR)        | {_r('type_token_ratio')} |",
        f"| Sentiment compound               | {_r('sentiment_compound')} |",
        f"| Sentiment positive               | {_r('sentiment_positive')} |",
        f"| Sentiment negative               | {_r('sentiment_negative')} |",
        f"| Word count                       | {_r('word_count')} |",
        f"| Avg line length                  | {_r('avg_line_length')} |",
        f"| Avg word length                  | {_r('avg_word_length')} |",
        f"| Rare word ratio                  | {_r('rare_word_ratio')} |",
        "",
        "## Word Frequency",
        "",
        f"**Top words in high-replay songs:** {', '.join(top_high_words)}",
        f"**Top words in low-replay songs:**  {', '.join(top_low_words)}",
        "",
        "## Top 2-Word Phrases",
        "",
        f"**Top phrases in high-replay songs:** {', '.join(top_high_bigrams)}",
        f"**Top phrases in low-replay songs:**  {', '.join(top_low_bigrams)}",
        "",
        "## Longest Words (most frequently appearing across all songs)",
        "",
        ", ".join(w for w, _ in all_longest.most_common(20)),
        "",
        "## Rarest Words (lowest English frequency, most frequently appearing)",
        "",
        ", ".join(w for w, _ in all_rarest.most_common(20)),
        "",
        "## LDA Topics",
        "",
        *[f"- **Topic {i}:** {', '.join(words[:10])}" for i, words in enumerate(topic_words)],
        "",
        *(
            [
                "## Lyrics Prediction Model Results",
                "",
                "| Feature Set | Algorithm | CV R² | Holdout R² | RMSE |",
                "|-------------|-----------|-------|------------|------|",
                *[
                    f"| {r['feature_set']} | {r['algorithm']} | {r['cv_r2_mean']:.3f}±{r['cv_r2_std']:.3f} | {r['holdout_r2']:.3f} | {r['holdout_rmse']:.3f} |"
                    for _, r in model_results_df.iterrows()
                ],
                "",
            ]
            if model_results_df is not None and not model_results_df.empty
            else []
        ),
        "## Output Files",
        "",
        *[
            f"- `{p.name}`"
            for p in [
                LYRICS_FEATURES_CSV, LYRICS_WORD_FREQ_CSV, LYRICS_BIGRAMS_CSV, LYRICS_TOPIC_CSV,
                LYRICS_SENTIMENT_PLOT, LYRICS_WORD_FREQ_PLOT, LYRICS_WORDCLOUD_PLOT,
                LYRICS_CORR_PLOT, LYRICS_TOPICS_PLOT, LYRICS_COMPLEXITY_PLOT,
                LYRICS_GENRE_SENT_PLOT, LYRICS_BIGRAMS_PLOT, LYRICS_RARITY_PLOT, LYRICS_WORDLEN_PLOT,
                LYRICS_MODEL_CSV, LYRICS_MODEL_PLOT,
            ]
        ],
    ]

    LYRICS_SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[LYRICS] Summary written to {LYRICS_SUMMARY_MD.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main(refresh_cache: bool = False) -> None:
    if not PROCESSED_CSV.exists():
        print(
            f"[LYRICS] Processed dataset not found at {PROCESSED_CSV}.\n"
            "  Run `python data_processing.py` first."
        )
        return

    print("[LYRICS] Loading processed dataset...")
    processed_df = pd.read_csv(
        PROCESSED_CSV,
        usecols=["mbid", "title", "artist_name", TARGET, RAW_TARGET, "genre"],
        dtype={"mbid": "string", "title": "string", "artist_name": "string", "genre": "string"},
    )

    lyrics_dict = load_lyrics_from_dataset(processed_df, refresh=refresh_cache)
    if not lyrics_dict:
        print("[LYRICS] No lyrics matched — cannot run analysis.")
        return

    print(f"\n[LYRICS] Analyzing {len(lyrics_dict):,} tracks...")

    features_df = extract_features(lyrics_dict)
    features_df.to_csv(LYRICS_FEATURES_CSV, index=False)
    print(f"[LYRICS] Features -> {LYRICS_FEATURES_CSV.name}")

    replay_map = processed_df.set_index("mbid")[TARGET].to_dict()
    merged_df  = features_df.merge(
        processed_df[["mbid", TARGET, RAW_TARGET, "genre"]].drop_duplicates("mbid"),
        on="mbid", how="left",
    )

    word_freq_df          = compute_word_frequencies(lyrics_dict, replay_map)
    bigram_df             = compute_bigram_frequencies(lyrics_dict, replay_map)
    topic_df, topic_words = run_lda(lyrics_dict)

    print("\n[LYRICS] Generating plots...")
    _plots = [
        ("sentiment by quartile",    lambda: _plot_sentiment_by_quartile(merged_df)),
        ("top words comparison",     lambda: _plot_word_frequency(word_freq_df)),
        ("word clouds",              lambda: _plot_word_clouds(word_freq_df)),
        ("features correlation",     lambda: _plot_feature_correlations(merged_df)),
        ("topic distribution",       lambda: _plot_topic_distribution(topic_df, topic_words, replay_map)),
        ("complexity vs replay",     lambda: _plot_complexity_vs_replay(merged_df)),
        ("sentiment by genre",       lambda: _plot_sentiment_by_genre(merged_df)),
        ("bigrams comparison",       lambda: _plot_bigrams(bigram_df)),
        ("rarity vs replay",         lambda: _plot_rarity_vs_replay(merged_df)),
        ("word length vs replay",    lambda: _plot_word_length_vs_replay(merged_df)),
    ]
    for name, fn in tqdm(_plots, desc="[LYRICS] plotting", unit="plot"):
        fn()

    print("\n[LYRICS] Training prediction models...")
    model_results_df = run_lyrics_model(features_df, topic_df)

    write_lyrics_summary(features_df, merged_df, word_freq_df, bigram_df, topic_words, model_results_df)

    print(
        "\n[LYRICS] Done.\n"
        f"  Features CSV : {LYRICS_FEATURES_CSV}\n"
        f"  Summary      : {LYRICS_SUMMARY_MD}\n"
        f"  Plots        : {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
