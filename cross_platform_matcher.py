"""
Cross-Platform Market Matcher: Polymarket <-> Kalshi
=====================================================
Matches prediction markets across Polymarket and Kalshi using fuzzy text
matching on market titles/questions, then computes cross-platform features
(price spreads, volume ratios, liquidity ratios).

Approach:
  1. Load the latest Polymarket snapshot from the master CSV.
  2. Load the latest Kalshi snapshot from the master CSV.
  3. Deduplicate each platform to the most recent snapshot per market.
  4. For each Kalshi market, find the best-matching Polymarket market using
     fuzzy string matching (rapidfuzz preferred, fuzzywuzzy as fallback).
  5. Keep pairs that exceed a configurable similarity threshold (default 65).
  6. Compute cross-platform features:
       - yes_price_spread:  Polymarket yes_price - Kalshi yes_bid/100
       - volume_ratio:      Polymarket volume / Kalshi volume
       - liquidity_ratio:   Polymarket liquidity / Kalshi open_interest
       - spread_diff:       difference in bid-ask spreads
  7. Save matched pairs to matched_markets.csv

Dependencies:
  - pandas
  - rapidfuzz (preferred) or fuzzywuzzy + python-Levenshtein
    Install: pip install rapidfuzz   (faster, no C dependency)
         or: pip install fuzzywuzzy python-Levenshtein

Note: The fuzzy matching is O(N*M) where N = Polymarket markets and M = Kalshi
markets. For large datasets we pre-filter by category keywords to reduce the
search space before doing pairwise fuzzy comparisons.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import re
import logging

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POLYMARKET_MASTER = os.path.join(_SCRIPT_DIR, "daily_snapshots", "all_snapshots.csv")
KALSHI_DIR = os.path.join(_SCRIPT_DIR, "kalshi_snapshots")
OUTPUT_PATH = os.path.join(_SCRIPT_DIR, "matched_markets.csv")

# Fuzzy matching threshold (0-100). Lower = more matches but more false positives.
MATCH_THRESHOLD = 65

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cross_platform_matcher")


# ---------------------------------------------------------------------------
# Fuzzy matching backend
# ---------------------------------------------------------------------------
def _get_fuzzy_scorer():
    """
    Return a scoring function: scorer(str_a, str_b) -> int (0-100).
    Prefers rapidfuzz (faster, no extra C deps), falls back to fuzzywuzzy.
    """
    try:
        from rapidfuzz import fuzz as rf_fuzz
        logger.info("Using rapidfuzz for fuzzy matching.")

        def scorer(a, b):
            # token_sort_ratio handles word reordering well
            return rf_fuzz.token_sort_ratio(a, b)

        return scorer
    except ImportError:
        pass

    try:
        from fuzzywuzzy import fuzz as fw_fuzz
        logger.info("Using fuzzywuzzy for fuzzy matching (consider installing rapidfuzz for speed).")

        def scorer(a, b):
            return fw_fuzz.token_sort_ratio(a, b)

        return scorer
    except ImportError:
        pass

    raise ImportError(
        "Neither rapidfuzz nor fuzzywuzzy is installed. "
        "Install one of them:  pip install rapidfuzz"
    )


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------
_NOISE_PATTERNS = re.compile(
    r"\b(will|the|be|by|before|after|in|on|at|to|of|a|an|and|or)\b",
    re.IGNORECASE,
)


def normalize_title(text):
    """
    Clean up a market title for better fuzzy matching:
      - lowercase
      - strip common stop-words
      - collapse whitespace
      - strip question marks and punctuation
    """
    if not text or not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = text.replace("?", "").replace("!", "").replace("'", "").replace('"', "")
    text = _NOISE_PATTERNS.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_polymarket():
    """
    Load Polymarket snapshots from the master CSV and deduplicate
    to the most recent snapshot per market (by market_id).
    """
    POLY_DATA_DIR = os.path.dirname(POLYMARKET_MASTER)
    frames = []

    # Try to load master file (may have mixed column counts)
    if os.path.exists(POLYMARKET_MASTER):
        logger.info("Loading Polymarket master file: %s", POLYMARKET_MASTER)
        try:
            df_master = pd.read_csv(POLYMARKET_MASTER, low_memory=False, on_bad_lines="skip")
            frames.append(df_master)
            logger.info("  Master file rows loaded: %d", len(df_master))
        except Exception as exc:
            logger.warning("  Could not read master file: %s", exc)

    # Also load recent individual snapshot CSVs for richer data
    if os.path.isdir(POLY_DATA_DIR):
        snapshot_files = sorted(
            [f for f in os.listdir(POLY_DATA_DIR)
             if f.startswith("snapshot_") and f.endswith(".csv")]
        )
        # Take the last 5 snapshot files (most recent)
        for fname in snapshot_files[-5:]:
            fpath = os.path.join(POLY_DATA_DIR, fname)
            try:
                df_snap = pd.read_csv(fpath, low_memory=False)
                frames.append(df_snap)
                logger.info("  Loaded snapshot %s: %d rows", fname, len(df_snap))
            except Exception as exc:
                logger.warning("  Could not read %s: %s", fname, exc)

    if not frames:
        logger.error("No Polymarket data could be loaded.")
        return pd.DataFrame()

    # Concatenate all frames (columns will be unioned; missing cols get NaN)
    df = pd.concat(frames, ignore_index=True, sort=False)
    logger.info("  Combined raw rows: %d", len(df))

    # Keep only the latest snapshot per market
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp").drop_duplicates(subset=["market_id"], keep="last")

    # Filter to markets with real prices (not 0 or 1)
    if "yes_price" in df.columns:
        df["yes_price"] = pd.to_numeric(df["yes_price"], errors="coerce")
        df = df[(df["yes_price"] > 0.01) & (df["yes_price"] < 0.99)]

    df["_norm_title"] = df["question"].apply(normalize_title)
    logger.info("  Polymarket markets after dedup & filter: %d", len(df))
    return df


def load_kalshi():
    """
    Load the latest Kalshi snapshot. Tries the master file first, then
    falls back to the most recent individual snapshot CSV in the data dir.
    """
    master_path = os.path.join(KALSHI_DIR, "all_kalshi_snapshots.csv")

    if os.path.exists(master_path):
        logger.info("Loading Kalshi data from master file: %s", master_path)
        df = pd.read_csv(master_path, low_memory=False)
    else:
        # Fall back to latest individual snapshot
        logger.info("Master file not found, scanning for latest snapshot in %s", KALSHI_DIR)
        if not os.path.isdir(KALSHI_DIR):
            logger.error("Kalshi data directory does not exist: %s", KALSHI_DIR)
            return pd.DataFrame()

        csvs = sorted(
            [f for f in os.listdir(KALSHI_DIR) if f.startswith("kalshi_snapshot_") and f.endswith(".csv")]
        )
        if not csvs:
            logger.error("No Kalshi snapshot files found in %s", KALSHI_DIR)
            return pd.DataFrame()

        latest = os.path.join(KALSHI_DIR, csvs[-1])
        logger.info("Using latest snapshot: %s", latest)
        df = pd.read_csv(latest, low_memory=False)

    logger.info("  Raw Kalshi rows: %d", len(df))

    # Deduplicate to latest snapshot per ticker
    if "snapshot_ts" in df.columns:
        df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], errors="coerce")
        df = df.sort_values("snapshot_ts").drop_duplicates(subset=["ticker"], keep="last")

    df["_norm_title"] = df["title"].apply(normalize_title)
    logger.info("  Kalshi markets after dedup: %d", len(df))
    return df


# ---------------------------------------------------------------------------
# Category-based pre-filtering
# ---------------------------------------------------------------------------
# Maps broad topic keywords to a canonical bucket so we can reduce the
# number of pairwise comparisons in fuzzy matching.
_CATEGORY_KEYWORDS = {
    "politics": ["trump", "biden", "president", "election", "congress", "senate",
                 "governor", "democrat", "republican", "vote", "cabinet", "tariff",
                 "impeach", "nominee", "administration", "white house", "poll"],
    "crypto": ["bitcoin", "btc", "ethereum", "eth", "crypto", "solana", "dogecoin",
               "memecoin", "blockchain", "web3", "defi", "nft"],
    "sports": ["nfl", "nba", "mlb", "nhl", "super bowl", "world series",
               "premier league", "champions league", "ufc", "boxing", "golf",
               "tennis", "olympics", "formula 1", "f1", "world cup"],
    "finance": ["stock", "s&p", "dow", "nasdaq", "fed ", "interest rate",
                "inflation", "recession", "gdp", "earnings"],
    "tech": ["ai ", "openai", "google", "apple", "microsoft", "tesla", "spacex",
             "nasa", "fda"],
    "climate": ["climate", "temperature", "weather", "hurricane", "wildfire"],
}


def assign_bucket(norm_text):
    """Assign a topic bucket based on keyword presence."""
    if not norm_text:
        return "other"
    for bucket, keywords in _CATEGORY_KEYWORDS.items():
        if any(kw in norm_text for kw in keywords):
            return bucket
    return "other"


# ---------------------------------------------------------------------------
# Matching logic
# ---------------------------------------------------------------------------
def find_matches(df_poly, df_kalshi, threshold=MATCH_THRESHOLD):
    """
    For each Kalshi market, find the best-matching Polymarket market.

    Strategy:
      1. Assign each market a topic bucket based on title keywords.
      2. Only compare markets within the same bucket (+ 'other' bucket
         is compared against everything as a fallback).
      3. Use token_sort_ratio for fuzzy comparison.
      4. Keep matches above the threshold.

    Returns a list of dicts with match info.
    """
    scorer = _get_fuzzy_scorer()

    # Assign buckets
    df_poly = df_poly.copy()
    df_kalshi = df_kalshi.copy()
    df_poly["_bucket"] = df_poly["_norm_title"].apply(assign_bucket)
    df_kalshi["_bucket"] = df_kalshi["_norm_title"].apply(assign_bucket)

    logger.info("Bucket distribution (Polymarket): %s",
                df_poly["_bucket"].value_counts().to_dict())
    logger.info("Bucket distribution (Kalshi): %s",
                df_kalshi["_bucket"].value_counts().to_dict())

    matches = []
    total = len(df_kalshi)

    for idx, (_, k_row) in enumerate(df_kalshi.iterrows()):
        k_title = k_row["_norm_title"]
        k_bucket = k_row["_bucket"]
        if not k_title:
            continue

        # Candidate pool: same bucket + "other" markets
        if k_bucket == "other":
            candidates = df_poly
        else:
            candidates = df_poly[
                (df_poly["_bucket"] == k_bucket) | (df_poly["_bucket"] == "other")
            ]

        if candidates.empty:
            candidates = df_poly  # fallback to full comparison

        best_score = 0
        best_poly_idx = None

        for p_idx, p_row in candidates.iterrows():
            p_title = p_row["_norm_title"]
            if not p_title:
                continue
            score = scorer(k_title, p_title)
            if score > best_score:
                best_score = score
                best_poly_idx = p_idx

        if best_score >= threshold and best_poly_idx is not None:
            p_row = df_poly.loc[best_poly_idx]
            matches.append({
                "kalshi_ticker": k_row.get("ticker"),
                "kalshi_title": k_row.get("title"),
                "kalshi_category": k_row.get("category"),
                "polymarket_id": p_row.get("market_id"),
                "polymarket_question": p_row.get("question"),
                "polymarket_category": p_row.get("category"),
                "match_score": best_score,
                # Kalshi prices (cents -> probability)
                "kalshi_yes_bid": k_row.get("yes_bid"),
                "kalshi_yes_ask": k_row.get("yes_ask"),
                "kalshi_last_price": k_row.get("last_price"),
                "kalshi_volume": k_row.get("volume"),
                "kalshi_volume_24h": k_row.get("volume_24h"),
                "kalshi_open_interest": k_row.get("open_interest"),
                "kalshi_close_time": k_row.get("close_time"),
                # Polymarket prices
                "poly_yes_price": p_row.get("yes_price"),
                "poly_volume": p_row.get("volume"),
                "poly_volume_24h": p_row.get("volume_24h"),
                "poly_liquidity": p_row.get("liquidity"),
                "poly_end_date": p_row.get("end_date"),
            })

        if (idx + 1) % 100 == 0:
            logger.info("  Processed %d / %d Kalshi markets, matches so far: %d",
                        idx + 1, total, len(matches))

    logger.info("Total matches found (threshold=%d): %d", threshold, len(matches))
    return matches


# ---------------------------------------------------------------------------
# Cross-platform features
# ---------------------------------------------------------------------------
def compute_cross_features(df_matches):
    """
    Compute cross-platform features for each matched pair:
      - yes_price_spread: Poly yes_price - Kalshi midpoint (as probability)
      - abs_price_spread: absolute value of the spread
      - volume_ratio: Poly total volume / Kalshi total volume
      - volume_24h_ratio: Poly 24h vol / Kalshi 24h vol
      - liquidity_ratio: Poly liquidity / Kalshi open_interest
    """
    df = df_matches.copy()

    # Convert Kalshi prices from cents to probability (0-1 scale)
    df["kalshi_yes_mid"] = df.apply(
        lambda r: _midpoint(r.get("kalshi_yes_bid"), r.get("kalshi_yes_ask")),
        axis=1,
    )

    # Price spread: Polymarket - Kalshi (both on 0-1 scale)
    df["yes_price_spread"] = df["poly_yes_price"] - df["kalshi_yes_mid"]
    df["abs_price_spread"] = df["yes_price_spread"].abs()

    # Volume ratio
    df["volume_ratio"] = _safe_ratio(df["poly_volume"], df["kalshi_volume"])

    # 24h volume ratio
    df["volume_24h_ratio"] = _safe_ratio(df["poly_volume_24h"], df["kalshi_volume_24h"])

    # Liquidity ratio (Poly liquidity vs Kalshi open interest)
    df["liquidity_ratio"] = _safe_ratio(df["poly_liquidity"], df["kalshi_open_interest"])

    return df


def _midpoint(bid, ask):
    """
    Compute midpoint from Kalshi bid/ask in cents, return as probability (0-1).
    Falls back to bid or ask if one is missing.
    """
    bid = pd.to_numeric(bid, errors="coerce")
    ask = pd.to_numeric(ask, errors="coerce")

    if pd.notna(bid) and pd.notna(ask):
        return (bid + ask) / 2.0 / 100.0
    elif pd.notna(bid):
        return bid / 100.0
    elif pd.notna(ask):
        return ask / 100.0
    return np.nan


def _safe_ratio(numerator, denominator):
    """Element-wise ratio with NaN for zeros/missing."""
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    return num / den.replace(0, np.nan)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def match_and_save(threshold=MATCH_THRESHOLD):
    """
    Full matching pipeline:
      1. Load Polymarket + Kalshi data
      2. Run fuzzy matching
      3. Compute cross-platform features
      4. Save to CSV
    Returns the matched DataFrame.
    """
    logger.info("=" * 60)
    logger.info("CROSS-PLATFORM MATCHER - %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 60)

    # Load data
    df_poly = load_polymarket()
    df_kalshi = load_kalshi()

    if df_poly.empty:
        logger.error("No Polymarket data loaded. Cannot proceed.")
        return pd.DataFrame()
    if df_kalshi.empty:
        logger.error("No Kalshi data loaded. Cannot proceed.")
        return pd.DataFrame()

    # Find matches
    raw_matches = find_matches(df_poly, df_kalshi, threshold=threshold)
    if not raw_matches:
        logger.warning("No matches found above threshold %d.", threshold)
        return pd.DataFrame()

    df_matches = pd.DataFrame(raw_matches)

    # Compute cross-platform features
    df_matches = compute_cross_features(df_matches)

    # Sort by match score (best matches first)
    df_matches = df_matches.sort_values("match_score", ascending=False)

    # Add metadata
    df_matches["matched_at"] = datetime.utcnow().isoformat()

    # Save
    df_matches.to_csv(OUTPUT_PATH, index=False)
    logger.info("Saved %d matched pairs to %s", len(df_matches), OUTPUT_PATH)

    # Summary
    logger.info("-" * 60)
    logger.info("MATCH SUMMARY")
    logger.info("-" * 60)
    logger.info("Total matched pairs: %d", len(df_matches))
    logger.info("Match score distribution:")
    logger.info("  Mean: %.1f", df_matches["match_score"].mean())
    logger.info("  Median: %.1f", df_matches["match_score"].median())
    logger.info("  Min: %.1f  Max: %.1f",
                df_matches["match_score"].min(), df_matches["match_score"].max())

    if "abs_price_spread" in df_matches.columns:
        valid_spreads = df_matches["abs_price_spread"].dropna()
        if not valid_spreads.empty:
            logger.info("Price spread (absolute) distribution:")
            logger.info("  Mean: %.4f", valid_spreads.mean())
            logger.info("  Median: %.4f", valid_spreads.median())
            logger.info("  Max: %.4f", valid_spreads.max())

    logger.info("\nTop 10 matches:")
    for _, row in df_matches.head(10).iterrows():
        k_title = (row.get("kalshi_title") or "")[:35]
        p_question = (row.get("polymarket_question") or "")[:35]
        spread = row.get("yes_price_spread")
        spread_str = f"{spread:+.3f}" if pd.notna(spread) else "N/A"
        logger.info("  [%d] %s  <->  %s  | spread=%s",
                    row["match_score"], k_title, p_question, spread_str)

    logger.info("\nLargest price discrepancies (potential arb):")
    if "abs_price_spread" in df_matches.columns:
        arb = df_matches.dropna(subset=["abs_price_spread"]).nlargest(5, "abs_price_spread")
        for _, row in arb.iterrows():
            k_title = (row.get("kalshi_title") or "")[:30]
            p_question = (row.get("polymarket_question") or "")[:30]
            spread = row.get("yes_price_spread", 0)
            logger.info("  spread=%+.3f | match=%d | %s <-> %s",
                        spread, row["match_score"], k_title, p_question)

    logger.info("Done!")
    return df_matches


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    match_and_save()
