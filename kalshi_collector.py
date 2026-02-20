"""
Kalshi Market Data Collector
=============================
Collects ALL active markets from the Kalshi public API (no auth required).

API: https://api.elections.kalshi.com/trade-api/v2
Endpoints used:
  - GET /events?with_nested_markets=true  (events + nested markets)
  - GET /markets?status=open              (all open markets, paginated)
  - GET /markets/{ticker}/orderbook       (order book for each market)

Data collected per market:
  - Ticker, title, category, event info
  - Yes/No bid/ask prices
  - Volume (total + 24h), open interest
  - Order book snapshot (bid depth on yes and no sides)
  - Close date, status

Output:
  - Per-run snapshot:  kalshi_snapshots/kalshi_snapshot_{date}.csv
  - Master append file: kalshi_snapshots/all_kalshi_snapshots.csv

Rate limiting: 0.5s sleep between API calls to be polite to the public endpoint.
"""

import requests
import pandas as pd
from datetime import datetime
import time
import os
import logging

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kalshi_snapshots")
RATE_LIMIT_SECONDS = 0.2  # pause between API calls
REQUEST_TIMEOUT = 30      # seconds
MAX_ORDERBOOK_MARKETS = 100  # cap order book fetches to avoid very long runs

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("kalshi_collector")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ensure_dirs():
    """Create the output directory tree if it doesn't already exist."""
    os.makedirs(DATA_DIR, exist_ok=True)


def _get(path, params=None):
    """
    Issue a GET request to the Kalshi public API with rate limiting.
    Returns the parsed JSON or None on failure.
    """
    url = f"{BASE_URL}{path}"
    try:
        time.sleep(RATE_LIMIT_SECONDS)
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
        else:
            logger.warning("GET %s returned status %s", url, resp.status_code)
            return None
    except requests.RequestException as exc:
        logger.error("Request failed for %s: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# Market fetching
# ---------------------------------------------------------------------------
def fetch_all_markets():
    """
    Paginate through GET /markets?status=open to collect every active market.
    Uses cursor-based pagination (Kalshi returns a 'cursor' field).
    Returns a list of raw market dicts.
    """
    all_markets = []
    cursor = None
    page = 0

    logger.info("Fetching all open markets from Kalshi...")
    while True:
        params = {
            "limit": 1000,
            "status": "open",
        }
        if cursor:
            params["cursor"] = cursor

        data = _get("/markets", params=params)
        if data is None:
            logger.warning("Failed to fetch markets page %d – stopping.", page)
            break

        markets = data.get("markets", [])
        if not markets:
            break

        all_markets.extend(markets)
        page += 1
        logger.info("  Page %d: got %d markets (total so far: %d)", page, len(markets), len(all_markets))

        cursor = data.get("cursor")
        if not cursor:
            break

    logger.info("Total open markets fetched: %d", len(all_markets))
    return all_markets


def fetch_events_with_markets():
    """
    Paginate through GET /events?with_nested_markets=true to build an
    event_ticker -> event metadata mapping (category, title, series_ticker).
    Returns dict keyed by event_ticker.
    """
    event_map = {}
    cursor = None
    page = 0

    logger.info("Fetching events for category/metadata enrichment...")
    while True:
        params = {
            "limit": 200,
            "with_nested_markets": "true",
        }
        if cursor:
            params["cursor"] = cursor

        data = _get("/events", params=params)
        if data is None:
            break

        events = data.get("events", [])
        if not events:
            break

        for ev in events:
            eticker = ev.get("event_ticker")
            if eticker:
                event_map[eticker] = {
                    "event_title": ev.get("title"),
                    "event_category": ev.get("category"),
                    "series_ticker": ev.get("series_ticker"),
                    "mutually_exclusive": ev.get("mutually_exclusive"),
                }
        page += 1
        if page % 5 == 0:
            logger.info("  Events pages fetched: %d (events mapped: %d)", page, len(event_map))

        cursor = data.get("cursor")
        if not cursor:
            break

    logger.info("Events mapped: %d", len(event_map))
    return event_map


# ---------------------------------------------------------------------------
# Order book
# ---------------------------------------------------------------------------
def fetch_orderbook(ticker):
    """
    GET /markets/{ticker}/orderbook
    Returns the raw orderbook dict or None.

    Kalshi orderbook response structure:
      {
        "orderbook": {
          "yes": [[price, quantity], ...],
          "no":  [[price, quantity], ...]
        }
      }
    Price is in cents (1-99).
    """
    data = _get(f"/markets/{ticker}/orderbook")
    if data is None:
        return None
    return data.get("orderbook", data)


def compute_orderbook_metrics(ob):
    """
    Derive summary metrics from an orderbook snapshot.

    Kalshi orderbooks only show bids (yes-bids and no-bids).
    A yes bid at price P implies a no ask at (100 - P) and vice versa.

    Returns a dict of metrics.
    """
    if not ob:
        return {}

    yes_bids = [level for level in (ob.get("yes") or []) if level and len(level) >= 2]
    no_bids = [level for level in (ob.get("no") or []) if level and len(level) >= 2]

    def _depth(levels, n=5):
        """Sum of quantities for top-n levels."""
        return sum(q for _, q in levels[:n])

    def _liquidity(levels, n=10):
        """Sum of (price * quantity) for top-n levels (in cent-contracts)."""
        return sum(p * q for p, q in levels[:n])

    yes_depth_5 = _depth(yes_bids, 5)
    no_depth_5 = _depth(no_bids, 5)
    yes_liq = _liquidity(yes_bids, 10)
    no_liq = _liquidity(no_bids, 10)
    total_liq = yes_liq + no_liq

    best_yes_bid = yes_bids[0][0] if yes_bids else None
    best_yes_bid_size = yes_bids[0][1] if yes_bids else None
    best_no_bid = no_bids[0][0] if no_bids else None
    best_no_bid_size = no_bids[0][1] if no_bids else None

    # Implied best ask for YES = 100 - best NO bid
    implied_yes_ask = (100 - best_no_bid) if best_no_bid is not None else None
    # Spread in cents between implied yes ask and best yes bid
    spread = None
    if implied_yes_ask is not None and best_yes_bid is not None:
        spread = implied_yes_ask - best_yes_bid

    # Imbalance: positive means more yes-side liquidity
    imbalance = (yes_liq - no_liq) / total_liq if total_liq > 0 else 0.0

    return {
        "ob_best_yes_bid": best_yes_bid,
        "ob_best_yes_bid_size": best_yes_bid_size,
        "ob_best_no_bid": best_no_bid,
        "ob_best_no_bid_size": best_no_bid_size,
        "ob_implied_yes_ask": implied_yes_ask,
        "ob_spread_cents": spread,
        "ob_yes_depth_5": yes_depth_5,
        "ob_no_depth_5": no_depth_5,
        "ob_yes_liquidity": round(yes_liq, 2),
        "ob_no_liquidity": round(no_liq, 2),
        "ob_imbalance": round(imbalance, 4),
        "ob_yes_levels": len(yes_bids),
        "ob_no_levels": len(no_bids),
    }


# ---------------------------------------------------------------------------
# Parsing / row construction
# ---------------------------------------------------------------------------
def safe_float(val):
    """Convert a value to float, returning None on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def parse_market(m, event_map):
    """
    Convert a raw Kalshi market dict into a flat row dict suitable for a DataFrame.

    Fields pulled directly from the API response:
      ticker, event_ticker, title, subtitle, status, market_type,
      yes_bid, yes_ask, no_bid, no_ask, last_price,
      volume, volume_24h, open_interest,
      open_time, close_time, expiration_time, result
    """
    event_ticker = m.get("event_ticker", "")
    ev_info = event_map.get(event_ticker, {})

    return {
        "snapshot_ts": datetime.utcnow().isoformat(),
        # Identifiers
        "ticker": m.get("ticker"),
        "event_ticker": event_ticker,
        "series_ticker": ev_info.get("series_ticker"),
        "title": m.get("title"),
        "subtitle": m.get("subtitle"),
        "event_title": ev_info.get("event_title"),
        "category": ev_info.get("event_category"),
        "market_type": m.get("market_type"),
        "status": m.get("status"),
        "result": m.get("result"),
        # Prices (cents)
        "yes_bid": safe_float(m.get("yes_bid")),
        "yes_ask": safe_float(m.get("yes_ask")),
        "no_bid": safe_float(m.get("no_bid")),
        "no_ask": safe_float(m.get("no_ask")),
        "last_price": safe_float(m.get("last_price")),
        # Dollar-denominated prices
        "yes_bid_dollars": safe_float(m.get("yes_bid_dollars")),
        "yes_ask_dollars": safe_float(m.get("yes_ask_dollars")),
        "no_bid_dollars": safe_float(m.get("no_bid_dollars")),
        "no_ask_dollars": safe_float(m.get("no_ask_dollars")),
        "last_price_dollars": safe_float(m.get("last_price_dollars")),
        # Volume / liquidity
        "volume": safe_float(m.get("volume")),
        "volume_24h": safe_float(m.get("volume_24h")),
        "open_interest": safe_float(m.get("open_interest")),
        "notional_value_dollars": safe_float(m.get("notional_value_dollars")),
        # Dates
        "open_time": m.get("open_time"),
        "close_time": m.get("close_time"),
        "expiration_time": m.get("expiration_time"),
        # Flags
        "can_close_early": m.get("can_close_early"),
        "mutually_exclusive": ev_info.get("mutually_exclusive"),
    }


# ---------------------------------------------------------------------------
# Main collection pipeline
# ---------------------------------------------------------------------------
def collect():
    """
    Full collection pipeline:
      1. Fetch events (for category enrichment)
      2. Fetch all open markets
      3. Parse into DataFrame
      4. Fetch order book snapshots for top markets by volume
      5. Merge order book metrics
      6. Save snapshot CSV + append to master CSV
    """
    ensure_dirs()

    logger.info("=" * 60)
    logger.info("KALSHI SNAPSHOT - %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 60)

    # Step 1: events for metadata enrichment
    event_map = fetch_events_with_markets()

    # Step 2: all open markets
    raw_markets = fetch_all_markets()
    if not raw_markets:
        logger.error("No markets fetched – aborting.")
        return None

    # Step 3: parse
    rows = [parse_market(m, event_map) for m in raw_markets]
    df = pd.DataFrame(rows)
    logger.info("Parsed %d markets into DataFrame (%d columns)", len(df), len(df.columns))

    # Filter to markets with volume > 0 (drops ~99% of inactive markets)
    before = len(df)
    df = df[pd.to_numeric(df["volume"], errors="coerce").fillna(0) > 0]
    logger.info("Filtered to markets with volume > 0: %d -> %d", before, len(df))

    # Step 4: order books for top markets by volume
    df_sorted = df.dropna(subset=["volume"]).sort_values("volume", ascending=False)
    tickers_for_ob = df_sorted["ticker"].head(MAX_ORDERBOOK_MARKETS).tolist()
    logger.info("Fetching order books for top %d markets by volume...", len(tickers_for_ob))

    ob_rows = []
    for idx, ticker in enumerate(tickers_for_ob):
        ob = fetch_orderbook(ticker)
        metrics = compute_orderbook_metrics(ob)
        if metrics:
            metrics["ticker"] = ticker
            ob_rows.append(metrics)
        if (idx + 1) % 50 == 0:
            logger.info("  Order books fetched: %d / %d", idx + 1, len(tickers_for_ob))

    logger.info("Order book snapshots collected: %d", len(ob_rows))

    # Step 5: merge
    if ob_rows:
        df_ob = pd.DataFrame(ob_rows)
        df = df.merge(df_ob, on="ticker", how="left")

    # Step 6: save
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    snapshot_path = os.path.join(DATA_DIR, f"kalshi_snapshot_{date_str}.csv")
    df.to_csv(snapshot_path, index=False)
    logger.info("Snapshot saved: %s", snapshot_path)

    master_path = os.path.join(DATA_DIR, "all_kalshi_snapshots.csv")
    if os.path.exists(master_path):
        df.to_csv(master_path, mode="a", header=False, index=False)
        logger.info("Appended to master file: %s", master_path)
    else:
        df.to_csv(master_path, index=False)
        logger.info("Created master file: %s", master_path)

    # Summary
    logger.info("-" * 60)
    logger.info("SUMMARY")
    logger.info("-" * 60)
    logger.info("Markets saved: %d", len(df))
    logger.info("Markets with order book data: %d",
                df["ob_imbalance"].notna().sum() if "ob_imbalance" in df.columns else 0)
    logger.info("Markets with category: %d", df["category"].notna().sum())

    if "category" in df.columns:
        cat_counts = df["category"].value_counts()
        logger.info("Category breakdown:")
        for cat, count in cat_counts.head(10).items():
            vol = df.loc[df["category"] == cat, "volume"].sum()
            logger.info("  %s: %d markets, volume=%s", cat, count, f"{vol:,.0f}" if pd.notna(vol) else "N/A")

    if "volume" in df.columns:
        logger.info("Top 5 by volume:")
        for _, row in df.nlargest(5, "volume").iterrows():
            title = (row.get("title") or "")[:50]
            logger.info("  %s | vol=%s | yes_bid=%s",
                        title,
                        f"{row['volume']:,.0f}" if pd.notna(row["volume"]) else "?",
                        row.get("yes_bid", "?"))

    logger.info("Done!")
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    collect()
