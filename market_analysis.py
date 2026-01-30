"""
Polymarket Market Analysis
==========================
Analyzes resolved markets to understand:
1. Market calibration (are prices accurate probabilities?)
2. Edge opportunities by category/volume
3. What data we need going forward
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "historical_data")

def analyze_resolved_markets():
    """Analyze resolved markets for insights"""

    # Load data
    files = [f for f in os.listdir(DATA_DIR) if f.startswith("polymarket_resolved_")]
    if not files:
        print("No resolved market data found!")
        return

    latest = sorted(files)[-1]
    df = pd.read_csv(os.path.join(DATA_DIR, latest))
    print(f"Loaded {len(df)} resolved markets from {latest}")

    # Basic stats
    print("\n" + "="*60)
    print("BASIC STATISTICS")
    print("="*60)
    print(f"Total resolved markets: {len(df)}")
    print(f"YES outcomes: {len(df[df['resolution'] == 'YES'])} ({len(df[df['resolution'] == 'YES'])/len(df)*100:.1f}%)")
    print(f"NO outcomes: {len(df[df['resolution'] == 'NO'])} ({len(df[df['resolution'] == 'NO'])/len(df)*100:.1f}%)")
    print(f"Total volume: ${df['volume'].sum():,.0f}")
    print(f"Average volume per market: ${df['volume'].mean():,.0f}")

    # Volume distribution
    print("\n" + "="*60)
    print("VOLUME DISTRIBUTION")
    print("="*60)
    vol_bins = [0, 10000, 50000, 100000, 500000, 1000000, float('inf')]
    vol_labels = ['5-10k', '10-50k', '50-100k', '100-500k', '500k-1M', '1M+']
    df['vol_bin'] = pd.cut(df['volume'], bins=vol_bins, labels=vol_labels)
    print(df.groupby('vol_bin')['resolution'].value_counts().unstack(fill_value=0))

    # Category analysis
    print("\n" + "="*60)
    print("CATEGORY ANALYSIS")
    print("="*60)
    if 'category' in df.columns:
        cat_stats = df.groupby('category').agg({
            'resolution': 'count',
            'volume': 'sum'
        }).rename(columns={'resolution': 'count'})
        cat_stats = cat_stats.sort_values('volume', ascending=False).head(15)
        print(cat_stats.to_string())

    # Win rate by category
    print("\n" + "="*60)
    print("YES WIN RATE BY CATEGORY (top categories)")
    print("="*60)
    if 'category' in df.columns:
        cat_yes = df.groupby('category').apply(
            lambda x: (x['resolution'] == 'YES').sum() / len(x) * 100
        ).sort_values(ascending=False)
        for cat, rate in cat_yes.head(10).items():
            count = len(df[df['category'] == cat])
            print(f"{cat}: {rate:.1f}% YES (n={count})")

    # Time analysis
    print("\n" + "="*60)
    print("TEMPORAL ANALYSIS")
    print("="*60)
    df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
    df['year'] = df['end_date'].dt.year
    year_stats = df.groupby('year').agg({
        'resolution': 'count',
        'volume': 'sum'
    }).rename(columns={'resolution': 'count'})
    print(year_stats.to_string())

    # Top markets by volume
    print("\n" + "="*60)
    print("TOP 20 MARKETS BY VOLUME")
    print("="*60)
    top = df.nlargest(20, 'volume')[['question', 'volume', 'resolution']]
    for _, row in top.iterrows():
        q = row['question'][:60] if row['question'] else 'N/A'
        q_safe = q.encode('ascii', 'ignore').decode()
        print(f"${row['volume']:>12,.0f} | {row['resolution']:>3} | {q_safe}")

    # Key insight: Base rate
    print("\n" + "="*60)
    print("KEY INSIGHT: BASE RATES")
    print("="*60)
    yes_rate = (df['resolution'] == 'YES').mean()
    print(f"Overall YES rate: {yes_rate*100:.1f}%")
    print(f"Overall NO rate: {(1-yes_rate)*100:.1f}%")
    print()
    print("This means if you bet NO on every market, you'd win ~60% of the time!")
    print("But without knowing the ENTRY PRICE, we can't calculate profitability.")

    # What we need for real backtesting
    print("\n" + "="*60)
    print("WHAT WE NEED FOR REAL BACKTESTING")
    print("="*60)
    print("""
    PROBLEM: Historical price data is not available for old markets.
    The 'yes_price' we have is the FINAL price (0 or 1), not entry price.

    SOLUTION: Start collecting data NOW for future backtesting:

    1. DAILY SNAPSHOTS of active markets:
       - Market ID, question, current price, volume, liquidity
       - Run daily to build historical price series

    2. TRACK RESOLUTIONS:
       - When markets close, record the outcome
       - Link to historical price series

    3. COLLECT SIGNALS:
       - Google Trends for relevant keywords
       - Wikipedia pageviews
       - News sentiment (GDELT)
       - Other prediction market prices (PredictIt)

    After 1-3 months of data collection, you'll have enough to:
    - Backtest Kelly strategies
    - Train ML models
    - Identify profitable edges
    """)

    return df


def create_data_collection_pipeline():
    """Create a script for ongoing data collection"""

    script = '''"""
Daily Data Collection Pipeline
Run this script daily (e.g., via Windows Task Scheduler)
to build historical data for backtesting
"""

import requests
import pandas as pd
from datetime import datetime
import json
import os
import time

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "daily_snapshots")

def ensure_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def fetch_active_markets():
    """Fetch all active markets with current prices"""
    all_markets = []
    offset = 0

    while True:
        url = "https://gamma-api.polymarket.com/markets"
        params = {"limit": 100, "offset": offset, "active": "true"}

        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code != 200:
                break
            data = resp.json()
            if not data:
                break
            all_markets.extend(data)
            offset += 100
            time.sleep(0.5)
        except Exception as e:
            print(f"Error: {e}")
            break

    return all_markets

def parse_market(m):
    """Extract key fields from market"""
    try:
        prices = json.loads(m.get("outcomePrices", "[]"))
        yes_price = float(prices[0]) if prices else None
    except:
        yes_price = None

    return {
        "timestamp": datetime.now().isoformat(),
        "market_id": m.get("id"),
        "question": m.get("question"),
        "category": m.get("category"),
        "yes_price": yes_price,
        "volume": float(m.get("volume", 0) or 0),
        "volume_24h": float(m.get("volume24hr", 0) or 0),
        "liquidity": float(m.get("liquidity", 0) or 0),
        "end_date": m.get("endDate"),
    }

def main():
    ensure_dir()

    print(f"Fetching active markets at {datetime.now()}")
    markets = fetch_active_markets()
    print(f"Found {len(markets)} active markets")

    parsed = [parse_market(m) for m in markets]
    df = pd.DataFrame(parsed)

    # Save daily snapshot
    date_str = datetime.now().strftime("%Y%m%d")
    filepath = os.path.join(DATA_DIR, f"snapshot_{date_str}.csv")
    df.to_csv(filepath, index=False)
    print(f"Saved to {filepath}")

    # Also append to master file for easy analysis
    master_path = os.path.join(DATA_DIR, "all_snapshots.csv")
    if os.path.exists(master_path):
        df.to_csv(master_path, mode='a', header=False, index=False)
    else:
        df.to_csv(master_path, index=False)
    print(f"Appended to {master_path}")

if __name__ == "__main__":
    main()
'''

    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "daily_collector.py")
    with open(filepath, 'w') as f:
        f.write(script)
    print(f"\nCreated daily collection script: {filepath}")
    print("Schedule this to run daily to build historical data!")


if __name__ == "__main__":
    df = analyze_resolved_markets()
    create_data_collection_pipeline()
