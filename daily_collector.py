"""
Daily Data Collection Pipeline
Run this script daily to build historical price data for backtesting
Includes order book snapshots for liquidity analysis
"""

import requests
import pandas as pd
from datetime import datetime
import json
import os
import time

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "daily_snapshots")
CLOB_API = "https://clob.polymarket.com"

def ensure_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def fetch_order_book(token_id):
    """Fetch order book for a specific token from CLOB API"""
    try:
        url = f"{CLOB_API}/book"
        resp = requests.get(url, params={"token_id": token_id}, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        pass
    return None

def calculate_order_book_metrics(book):
    """Calculate useful metrics from order book data"""
    if not book:
        return {}

    bids = book.get("bids", [])
    asks = book.get("asks", [])

    # Total liquidity (sum of size * price for top 10 levels)
    bid_liquidity = sum(float(b["size"]) * float(b["price"]) for b in bids[:10]) if bids else 0
    ask_liquidity = sum(float(a["size"]) * float(a["price"]) for a in asks[:10]) if asks else 0
    total_liquidity = bid_liquidity + ask_liquidity

    # Order book imbalance: positive = more buy pressure, negative = more sell pressure
    imbalance = (bid_liquidity - ask_liquidity) / total_liquidity if total_liquidity > 0 else 0

    # Best prices and sizes
    best_bid_price = float(bids[0]["price"]) if bids else None
    best_bid_size = float(bids[0]["size"]) if bids else None
    best_ask_price = float(asks[0]["price"]) if asks else None
    best_ask_size = float(asks[0]["size"]) if asks else None

    # Depth at different levels (cumulative size)
    bid_depth_5 = sum(float(b["size"]) for b in bids[:5]) if len(bids) >= 5 else sum(float(b["size"]) for b in bids)
    ask_depth_5 = sum(float(a["size"]) for a in asks[:5]) if len(asks) >= 5 else sum(float(a["size"]) for a in asks)

    # Number of levels
    bid_levels = len(bids)
    ask_levels = len(asks)

    return {
        "ob_bid_liquidity": round(bid_liquidity, 2),
        "ob_ask_liquidity": round(ask_liquidity, 2),
        "ob_imbalance": round(imbalance, 4),
        "ob_best_bid": best_bid_price,
        "ob_best_bid_size": best_bid_size,
        "ob_best_ask": best_ask_price,
        "ob_best_ask_size": best_ask_size,
        "ob_bid_depth_5": round(bid_depth_5, 2),
        "ob_ask_depth_5": round(ask_depth_5, 2),
        "ob_bid_levels": bid_levels,
        "ob_ask_levels": ask_levels,
    }

def fetch_event_categories():
    """Fetch all events to build event_id -> category mapping"""
    event_categories = {}
    offset = 0
    max_offset = 25000  # Fetch up to 25k events

    print("Fetching event categories...")
    while offset < max_offset:
        try:
            url = "https://gamma-api.polymarket.com/events"
            resp = requests.get(url, params={"limit": 100, "offset": offset}, timeout=30)
            if resp.status_code != 200:
                break
            events = resp.json()
            if not events:
                break
            for e in events:
                event_id = e.get("id")
                category = e.get("category")
                if event_id and category:
                    event_categories[str(event_id)] = category
            if offset % 2000 == 0 and offset > 0:
                print(f"  Fetched {offset} events...")
            offset += 100
            time.sleep(0.1)
        except Exception as e:
            break

    print(f"  Found {len(event_categories)} events with categories")
    return event_categories

def infer_category(question):
    """Infer category from question text using keywords"""
    if not question:
        return None
    q = question.lower()

    # Sports keywords
    sports_keywords = ['nfl', 'nba', 'mlb', 'nhl', 'super bowl', 'world series', 'stanley cup',
                       'premier league', 'champions league', 'world cup', 'olympics', 'tennis',
                       'golf', 'ufc', 'boxing', 'formula 1', 'f1', 'cricket', 'soccer', 'football',
                       'basketball', 'baseball', 'hockey', 'playoffs', 'championship', 'win the 202',
                       'beat the', 'score more', 'mvp', 'touchdown', 'goal', 'match']
    if any(kw in q for kw in sports_keywords):
        return 'Sports'

    # Crypto keywords
    crypto_keywords = ['bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'token', 'defi', 'nft',
                       'solana', 'cardano', 'dogecoin', 'memecoin', 'blockchain', 'web3',
                       'hyperliquid', 'metamask', 'coinbase', 'binance']
    if any(kw in q for kw in crypto_keywords):
        return 'Crypto'

    # Politics keywords
    politics_keywords = ['trump', 'biden', 'president', 'election', 'congress', 'senate',
                         'governor', 'democrat', 'republican', 'vote', 'poll', 'government',
                         'cabinet', 'minister', 'parliament', 'tariff', 'shutdown', 'impeach',
                         'nominee', 'administration']
    if any(kw in q for kw in politics_keywords):
        return 'Politics'

    # Global Politics
    global_politics = ['ukraine', 'russia', 'china', 'nato', 'eu ', 'european union', 'ceasefire',
                       'war', 'invasion', 'sanction', 'treaty', 'diplomatic']
    if any(kw in q for kw in global_politics):
        return 'Global Politics'

    # Pop Culture
    pop_culture = ['movie', 'film', 'oscar', 'grammy', 'album', 'song', 'artist', 'celebrity',
                   'tv show', 'netflix', 'disney', 'marvel', 'gta', 'video game', 'gaming',
                   'twitch', 'youtube', 'tiktok', 'influencer']
    if any(kw in q for kw in pop_culture):
        return 'Pop Culture'

    # Science/Tech
    science_tech = ['ai ', 'artificial intelligence', 'spacex', 'nasa', 'space', 'mars',
                    'climate', 'vaccine', 'fda', 'drug', 'pharmaceutical', 'research',
                    'scientist', 'openai', 'google', 'apple', 'microsoft', 'tesla']
    if any(kw in q for kw in science_tech):
        return 'Science & Tech'

    # Business/Finance
    business = ['stock', 'market', 'fed ', 'federal reserve', 'interest rate', 'inflation',
                'recession', 'gdp', 'earnings', 'ipo', 'merger', 'acquisition', 'company',
                's&p', 'dow jones', 'nasdaq']
    if any(kw in q for kw in business):
        return 'Business'

    return None

def fetch_active_markets(max_pages=50):
    """Fetch OPEN markets (not closed) with live prices"""
    all_markets = []
    offset = 0
    page = 0

    print("Fetching OPEN markets (closed=false)...")
    while page < max_pages:
        url = "https://gamma-api.polymarket.com/markets"
        params = {"limit": 100, "offset": offset, "closed": "false"}

        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code != 200:
                break
            data = resp.json()
            if not data:
                break
            all_markets.extend(data)
            print(f"  Page {page+1}: {len(data)} markets (total: {len(all_markets)})")
            if len(data) < 100:
                break
            offset += 100
            page += 1
            time.sleep(0.3)
        except Exception as e:
            print(f"Error: {e}")
            break

    return all_markets

def parse_market(m):
    """Extract all useful fields from market"""
    # Parse prices
    try:
        prices = json.loads(m.get("outcomePrices", "[]"))
        yes_price = float(prices[0]) if prices else None
        no_price = float(prices[1]) if len(prices) > 1 else None
    except:
        yes_price = None
        no_price = None

    # Parse outcomes to check if binary
    try:
        outcomes = json.loads(m.get("outcomes", "[]"))
        is_binary = outcomes == ["Yes", "No"]
    except:
        outcomes = []
        is_binary = False

    # Extract event_id for category lookup
    event_id = None
    try:
        events = m.get("events", [])
        if events and len(events) > 0:
            event_id = str(events[0].get("id"))
    except:
        pass

    # Safe float conversion
    def safe_float(val):
        try:
            return float(val) if val is not None else None
        except:
            return None

    return {
        # Identifiers
        "timestamp": datetime.now().isoformat(),
        "market_id": m.get("id"),
        "event_id": event_id,
        "slug": m.get("slug"),
        "question": m.get("question"),

        # Prices (implied probabilities)
        "yes_price": yes_price,
        "no_price": no_price,
        "last_trade_price": safe_float(m.get("lastTradePrice")),
        "best_bid": safe_float(m.get("bestBid")),
        "best_ask": safe_float(m.get("bestAsk")),
        "spread": safe_float(m.get("spread")),

        # Price changes (momentum signals)
        "price_change_1h": safe_float(m.get("oneHourPriceChange")),
        "price_change_24h": safe_float(m.get("oneDayPriceChange")),
        "price_change_1w": safe_float(m.get("oneWeekPriceChange")),
        "price_change_1m": safe_float(m.get("oneMonthPriceChange")),

        # Volume
        "volume": safe_float(m.get("volume")),
        "volume_24h": safe_float(m.get("volume24hr")),
        "volume_1w": safe_float(m.get("volume1wk")),
        "volume_1m": safe_float(m.get("volume1mo")),

        # Liquidity & quality
        "liquidity": safe_float(m.get("liquidity")),
        "competitive": safe_float(m.get("competitive")),

        # Dates
        "start_date": m.get("startDate"),
        "end_date": m.get("endDate"),

        # Metadata
        "is_binary": is_binary,
        "outcomes": str(outcomes),

        # CLOB token IDs (for order book fetching)
        "clob_token_ids": m.get("clobTokenIds", "[]"),
    }

def main():
    ensure_dir()

    print(f"\n{'='*60}")
    print(f"DAILY POLYMARKET SNAPSHOT - ENHANCED")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # Fetch event categories first
    event_categories = fetch_event_categories()

    # Fetch up to 5000 markets (50 pages x 100)
    markets = fetch_active_markets(max_pages=50)
    print(f"\nTotal markets fetched: {len(markets)}")

    # Parse all markets
    parsed = [parse_market(m) for m in markets]
    df = pd.DataFrame(parsed)

    # Add category from event lookup, with keyword fallback
    df['category'] = df['event_id'].map(event_categories)
    # Use keyword-based inference for missing categories
    mask = df['category'].isna()
    df.loc[mask, 'category'] = df.loc[mask, 'question'].apply(infer_category)

    # Filter for:
    # 1. Binary outcomes only (Yes/No)
    # 2. Real prices (not 0 or 1)
    # 3. Volume > $1000
    df_filtered = df[
        (df['is_binary'] == True) &
        (df['yes_price'] > 0.01) &
        (df['yes_price'] < 0.99) &
        (df['volume'] > 1000)
    ].copy()

    print(f"Binary markets with real prices & volume > $1000: {len(df_filtered)}")

    # Fetch order book data for filtered markets (top 200 by volume to limit API calls)
    print(f"\nFetching order book snapshots...")
    top_markets = df_filtered.nlargest(200, 'volume') if len(df_filtered) > 200 else df_filtered

    ob_data = []
    for idx, (_, row) in enumerate(top_markets.iterrows()):
        try:
            token_ids = json.loads(row['clob_token_ids']) if isinstance(row['clob_token_ids'], str) else row['clob_token_ids']
            if token_ids and len(token_ids) > 0:
                # Fetch YES token order book (first token)
                book = fetch_order_book(token_ids[0])
                metrics = calculate_order_book_metrics(book)
                metrics["market_id"] = row["market_id"]
                metrics["timestamp"] = row["timestamp"]
                ob_data.append(metrics)

                if (idx + 1) % 50 == 0:
                    print(f"  Order books fetched: {idx + 1}/{len(top_markets)}")

                time.sleep(0.1)  # Rate limit
        except Exception as e:
            pass

    print(f"  Order books collected: {len(ob_data)}")

    # Merge order book metrics back to main dataframe
    if ob_data:
        df_ob = pd.DataFrame(ob_data)
        df_filtered = df_filtered.merge(df_ob, on=["market_id", "timestamp"], how="left")

    # Save snapshot with timestamp (allows multiple runs per day)
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    filepath = os.path.join(DATA_DIR, f"snapshot_{date_str}.csv")
    df_filtered.to_csv(filepath, index=False)
    print(f"\nSaved to: {filepath}")

    # Also append to master file
    master_path = os.path.join(DATA_DIR, "all_snapshots.csv")
    if os.path.exists(master_path):
        df_filtered.to_csv(master_path, mode='a', header=False, index=False)
        print(f"Appended to: {master_path}")
    else:
        df_filtered.to_csv(master_path, index=False)
        print(f"Created: {master_path}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Markets saved: {len(df_filtered)}")
    print(f"Columns: {len(df_filtered.columns)}")
    print(f"Markets with order book data: {df_filtered['ob_imbalance'].notna().sum() if 'ob_imbalance' in df_filtered.columns else 0}")
    print(f"Markets with category: {df_filtered['category'].notna().sum()}")

    # Category breakdown
    if 'category' in df_filtered.columns:
        print(f"\nCategory breakdown:")
        cat_counts = df_filtered['category'].value_counts()
        for cat, count in cat_counts.head(10).items():
            vol = df_filtered[df_filtered['category'] == cat]['volume'].sum()
            print(f"  {cat}: {count} markets, ${vol:,.0f} volume")

    print(f"\nTop 5 by volume:")
    top5 = df_filtered.nlargest(5, 'volume')[['question', 'yes_price', 'spread', 'volume']]
    for _, row in top5.iterrows():
        q = (row['question'] or '')[:45]
        q_safe = q.encode('ascii', 'ignore').decode()
        spread = row['spread'] if row['spread'] else 0
        print(f"  ${row['volume']:>10,.0f} | {row['yes_price']:.2f} | spread:{spread:.3f} | {q_safe}")

    print(f"\nTop 5 by 24h price movement:")
    if 'price_change_24h' in df_filtered.columns:
        movers = df_filtered.dropna(subset=['price_change_24h'])
        movers = movers.reindex(movers['price_change_24h'].abs().sort_values(ascending=False).index)
        for _, row in movers.head(5).iterrows():
            q = (row['question'] or '')[:45]
            q_safe = q.encode('ascii', 'ignore').decode()
            change = row['price_change_24h'] if row['price_change_24h'] else 0
            print(f"  {change:+.3f} | {row['yes_price']:.2f} | {q_safe}")

    # Order book imbalance leaders
    if 'ob_imbalance' in df_filtered.columns:
        print(f"\nTop 5 by order book imbalance (buy pressure):")
        ob_markets = df_filtered.dropna(subset=['ob_imbalance'])
        top_buy = ob_markets.nlargest(5, 'ob_imbalance')
        for _, row in top_buy.iterrows():
            q = (row['question'] or '')[:40]
            q_safe = q.encode('ascii', 'ignore').decode()
            imb = row['ob_imbalance'] if row['ob_imbalance'] else 0
            print(f"  {imb:+.3f} | {row['yes_price']:.2f} | {q_safe}")

        print(f"\nTop 5 by order book imbalance (sell pressure):")
        top_sell = ob_markets.nsmallest(5, 'ob_imbalance')
        for _, row in top_sell.iterrows():
            q = (row['question'] or '')[:40]
            q_safe = q.encode('ascii', 'ignore').decode()
            imb = row['ob_imbalance'] if row['ob_imbalance'] else 0
            print(f"  {imb:+.3f} | {row['yes_price']:.2f} | {q_safe}")

    print(f"\nDone!")

if __name__ == "__main__":
    main()
