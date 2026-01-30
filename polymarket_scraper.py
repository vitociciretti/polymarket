"""
Polymarket Historical Data Scraper & Backtesting Framework
===========================================================
This script:
1. Scrapes ALL markets from Polymarket (2020-present)
2. Gets price history for each market
3. Saves to CSV for analysis
4. Provides a backtesting framework with Kelly criterion
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
import os
from typing import List, Dict, Optional, Tuple

# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "historical_data")
RATE_LIMIT_DELAY = 0.5  # seconds between API calls

def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    return DATA_DIR

# =============================================================================
# POLYMARKET API FUNCTIONS
# =============================================================================

def fetch_all_markets(limit_per_page: int = 100, max_pages: int = 100) -> List[Dict]:
    """
    Fetch ALL markets from Polymarket (both active and closed)
    Uses pagination to get everything
    """
    all_markets = []

    # Fetch closed markets
    print("Fetching closed markets...")
    offset = 0
    page = 0
    while page < max_pages:
        url = "https://gamma-api.polymarket.com/markets"
        params = {
            "limit": limit_per_page,
            "offset": offset,
            "closed": "true"
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code != 200:
                print(f"  Error: {resp.status_code}")
                break

            data = resp.json()
            if not data:
                break

            all_markets.extend(data)
            print(f"  Page {page + 1}: Got {len(data)} markets (total: {len(all_markets)})")

            if len(data) < limit_per_page:
                break

            offset += limit_per_page
            page += 1
            time.sleep(RATE_LIMIT_DELAY)

        except Exception as e:
            print(f"  Error fetching page {page}: {e}")
            break

    # Fetch active markets
    print("\nFetching active markets...")
    offset = 0
    page = 0
    while page < max_pages:
        url = "https://gamma-api.polymarket.com/markets"
        params = {
            "limit": limit_per_page,
            "offset": offset,
            "active": "true"
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code != 200:
                break

            data = resp.json()
            if not data:
                break

            all_markets.extend(data)
            print(f"  Page {page + 1}: Got {len(data)} markets (total: {len(all_markets)})")

            if len(data) < limit_per_page:
                break

            offset += limit_per_page
            page += 1
            time.sleep(RATE_LIMIT_DELAY)

        except Exception as e:
            print(f"  Error fetching page {page}: {e}")
            break

    return all_markets


def get_price_history(token_id: str, fidelity: int = 1440) -> List[Dict]:
    """
    Get price history for a market token
    fidelity: 1440 = daily, 60 = hourly
    """
    url = "https://clob.polymarket.com/prices-history"
    params = {
        "market": token_id,
        "interval": "max",
        "fidelity": fidelity
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("history", [])
    except Exception as e:
        pass

    return []


def parse_market_data(market: Dict) -> Dict:
    """Extract key fields from a market"""

    # Parse outcome prices
    outcome_prices = market.get("outcomePrices", "[]")
    try:
        prices = json.loads(outcome_prices) if isinstance(outcome_prices, str) else outcome_prices
        yes_price = float(prices[0]) if prices and prices[0] else None
        no_price = float(prices[1]) if prices and len(prices) > 1 and prices[1] else None
    except:
        yes_price = None
        no_price = None

    # Determine resolution
    resolution = None
    if yes_price is not None and no_price is not None:
        if yes_price > 0.99 and no_price < 0.01:
            resolution = "YES"
        elif no_price > 0.99 and yes_price < 0.01:
            resolution = "NO"

    # Parse token IDs
    token_ids = market.get("clobTokenIds", "[]")
    try:
        tokens = json.loads(token_ids) if isinstance(token_ids, str) else token_ids
        yes_token = tokens[0] if tokens else None
        no_token = tokens[1] if len(tokens) > 1 else None
    except:
        yes_token = None
        no_token = None

    return {
        "id": market.get("id"),
        "question": market.get("question"),
        "slug": market.get("slug"),
        "category": market.get("category"),
        "end_date": market.get("endDate"),
        "created_at": market.get("createdAt"),
        "volume": float(market.get("volume", 0) or 0),
        "liquidity": float(market.get("liquidity", 0) or 0),
        "yes_price": yes_price,
        "no_price": no_price,
        "resolution": resolution,
        "active": market.get("active"),
        "closed": market.get("closed"),
        "yes_token": yes_token,
        "no_token": no_token,
    }


# =============================================================================
# DATA COLLECTION
# =============================================================================

def scrape_all_polymarket_data(min_volume: float = 1000, get_history: bool = True):
    """
    Main function to scrape all Polymarket data
    """
    ensure_data_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("="*60)
    print("POLYMARKET HISTORICAL DATA SCRAPER")
    print("="*60)

    # Step 1: Fetch all markets
    print("\n[1/3] Fetching all markets...")
    raw_markets = fetch_all_markets()
    print(f"Total markets fetched: {len(raw_markets)}")

    # Step 2: Parse and filter
    print("\n[2/3] Parsing market data...")
    markets = []
    for m in raw_markets:
        parsed = parse_market_data(m)
        if parsed["volume"] >= min_volume:
            markets.append(parsed)

    print(f"Markets with volume >= ${min_volume:,.0f}: {len(markets)}")

    # Categorize
    resolved = [m for m in markets if m["resolution"] is not None]
    active = [m for m in markets if m["active"] and not m["closed"]]
    print(f"  - Resolved markets: {len(resolved)}")
    print(f"  - Active markets: {len(active)}")

    # Step 3: Get price history for resolved markets
    price_histories = []
    if get_history and resolved:
        print(f"\n[3/3] Fetching price history for {len(resolved)} resolved markets...")

        for i, m in enumerate(resolved):
            if m["yes_token"]:
                history = get_price_history(m["yes_token"])
                if history:
                    for h in history:
                        price_histories.append({
                            "market_id": m["id"],
                            "question": m["question"][:50] if m["question"] else "",
                            "timestamp": h["t"],
                            "date": datetime.fromtimestamp(h["t"]).strftime("%Y-%m-%d"),
                            "yes_price": h["p"],
                            "resolution": m["resolution"],
                            "volume": m["volume"]
                        })
                    q_safe = (m['question'] or '')[:40].encode('ascii', 'ignore').decode()
                    print(f"  [{i+1}/{len(resolved)}] {q_safe}... -> {len(history)} points")
                else:
                    q_safe = (m['question'] or '')[:40].encode('ascii', 'ignore').decode()
                    print(f"  [{i+1}/{len(resolved)}] {q_safe}... -> No history")

                time.sleep(RATE_LIMIT_DELAY)

            # Progress update
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(resolved)} markets processed")

    # Save to CSV
    print("\nSaving data...")

    # Markets summary
    markets_df = pd.DataFrame(markets)
    markets_path = os.path.join(DATA_DIR, f"polymarket_all_markets_{timestamp}.csv")
    markets_df.to_csv(markets_path, index=False)
    print(f"  Saved {len(markets)} markets to {os.path.basename(markets_path)}")

    # Resolved markets
    resolved_df = pd.DataFrame(resolved)
    resolved_path = os.path.join(DATA_DIR, f"polymarket_resolved_{timestamp}.csv")
    resolved_df.to_csv(resolved_path, index=False)
    print(f"  Saved {len(resolved)} resolved markets to {os.path.basename(resolved_path)}")

    # Price histories
    if price_histories:
        history_df = pd.DataFrame(price_histories)
        history_path = os.path.join(DATA_DIR, f"polymarket_price_history_{timestamp}.csv")
        history_df.to_csv(history_path, index=False)
        print(f"  Saved {len(price_histories)} price points to {os.path.basename(history_path)}")

    return {
        "markets": markets,
        "resolved": resolved,
        "price_histories": price_histories
    }


# =============================================================================
# BACKTESTING FRAMEWORK
# =============================================================================

class KellyBacktester:
    """
    Backtesting framework using Kelly Criterion
    """

    def __init__(self, initial_bankroll: float = 10000, kelly_fraction: float = 0.25):
        """
        Args:
            initial_bankroll: Starting capital
            kelly_fraction: Fraction of Kelly to use (0.25 = quarter Kelly for safety)
        """
        self.initial_bankroll = initial_bankroll
        self.kelly_fraction = kelly_fraction
        self.trades = []
        self.bankroll_history = []

    def kelly_bet_size(self, our_prob: float, market_prob: float, bankroll: float) -> Tuple[float, str]:
        """
        Calculate Kelly bet size

        Args:
            our_prob: Our estimated probability of YES
            market_prob: Market's implied probability (YES price)
            bankroll: Current bankroll

        Returns:
            (bet_size, direction) - positive = bet YES, negative = bet NO
        """
        # Betting on YES
        if our_prob > market_prob:
            # b = odds received = (1 - market_prob) / market_prob
            # For binary market: if YES costs 0.4, you get 1/0.4 = 2.5x return
            b = (1 - market_prob) / market_prob if market_prob > 0 else 0
            p = our_prob
            q = 1 - p

            if b > 0:
                kelly = (b * p - q) / b
                kelly = max(0, kelly)  # No negative bets
                bet_size = bankroll * kelly * self.kelly_fraction
                return bet_size, "YES"

        # Betting on NO
        elif our_prob < market_prob:
            # Flip perspective: NO costs (1 - market_prob)
            no_price = 1 - market_prob
            b = market_prob / no_price if no_price > 0 else 0
            p = 1 - our_prob  # Prob of NO
            q = 1 - p

            if b > 0:
                kelly = (b * p - q) / b
                kelly = max(0, kelly)
                bet_size = bankroll * kelly * self.kelly_fraction
                return bet_size, "NO"

        return 0, "NONE"

    def simulate_trade(self, bet_size: float, direction: str, market_prob: float,
                       resolution: str, bankroll: float) -> float:
        """
        Simulate a single trade

        Returns: PnL from this trade
        """
        if direction == "NONE" or bet_size <= 0:
            return 0

        # Calculate shares bought
        if direction == "YES":
            price = market_prob
            shares = bet_size / price if price > 0 else 0
            # If resolved YES, each share pays $1
            pnl = shares * 1.0 - bet_size if resolution == "YES" else -bet_size
        else:  # NO
            price = 1 - market_prob
            shares = bet_size / price if price > 0 else 0
            pnl = shares * 1.0 - bet_size if resolution == "NO" else -bet_size

        return pnl

    def backtest(self, markets_df: pd.DataFrame,
                 signal_func,
                 entry_days_before_close: int = 30) -> pd.DataFrame:
        """
        Run backtest on historical resolved markets

        Args:
            markets_df: DataFrame with columns: question, yes_price, resolution, volume, end_date
            signal_func: Function that takes market data and returns our_probability
            entry_days_before_close: How many days before close to enter (for price history)

        Returns:
            DataFrame with trade results
        """
        bankroll = self.initial_bankroll
        self.bankroll_history = [bankroll]
        self.trades = []

        for _, market in markets_df.iterrows():
            if market["resolution"] not in ["YES", "NO"]:
                continue
            if market["yes_price"] is None:
                continue

            # Get our signal/probability estimate
            our_prob = signal_func(market)
            if our_prob is None:
                continue

            market_prob = market["yes_price"]

            # Skip if market already resolved (price is 0 or 1)
            if market_prob < 0.01 or market_prob > 0.99:
                continue

            # Calculate bet size
            bet_size, direction = self.kelly_bet_size(our_prob, market_prob, bankroll)

            # Cap bet size at 10% of bankroll
            max_bet = bankroll * 0.10
            bet_size = min(bet_size, max_bet)

            if bet_size < 1:  # Minimum bet
                continue

            # Simulate trade
            pnl = self.simulate_trade(bet_size, direction, market_prob,
                                     market["resolution"], bankroll)

            bankroll += pnl
            self.bankroll_history.append(bankroll)

            self.trades.append({
                "question": market["question"][:50],
                "our_prob": our_prob,
                "market_prob": market_prob,
                "direction": direction,
                "bet_size": bet_size,
                "resolution": market["resolution"],
                "pnl": pnl,
                "bankroll": bankroll,
                "volume": market["volume"]
            })

            # Stop if bankrupt
            if bankroll < 100:
                print("BANKRUPT!")
                break

        return pd.DataFrame(self.trades)

    def get_stats(self) -> Dict:
        """Calculate backtest statistics"""
        if not self.trades:
            return {}

        trades_df = pd.DataFrame(self.trades)

        winning_trades = trades_df[trades_df["pnl"] > 0]
        losing_trades = trades_df[trades_df["pnl"] < 0]

        total_pnl = trades_df["pnl"].sum()
        win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0

        # Calculate max drawdown
        peak = self.initial_bankroll
        max_dd = 0
        for b in self.bankroll_history:
            if b > peak:
                peak = b
            dd = (peak - b) / peak
            max_dd = max(max_dd, dd)

        return {
            "total_trades": len(trades_df),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "return_pct": (self.bankroll_history[-1] / self.initial_bankroll - 1) * 100,
            "max_drawdown_pct": max_dd * 100,
            "final_bankroll": self.bankroll_history[-1],
            "avg_trade_pnl": trades_df["pnl"].mean(),
            "sharpe_approx": trades_df["pnl"].mean() / trades_df["pnl"].std() if trades_df["pnl"].std() > 0 else 0
        }


# =============================================================================
# EXAMPLE STRATEGIES
# =============================================================================

def strategy_follow_the_crowd(market: pd.Series) -> float:
    """
    Simple strategy: If market is very confident (>80%), agree with it
    This tests if high-confidence markets resolve as expected
    """
    market_prob = market["yes_price"]

    if market_prob > 0.80:
        return market_prob + 0.05  # Slightly more confident than market
    elif market_prob < 0.20:
        return market_prob - 0.05  # Slightly more confident in NO
    else:
        return None  # No trade in uncertain markets


def strategy_contrarian(market: pd.Series) -> float:
    """
    Contrarian strategy: Bet against extreme prices
    Theory: Markets sometimes overshoot
    """
    market_prob = market["yes_price"]

    if market_prob > 0.90:
        return 0.85  # Think it's overpriced
    elif market_prob < 0.10:
        return 0.15  # Think NO is overpriced
    else:
        return None


def strategy_volume_momentum(market: pd.Series) -> float:
    """
    Volume-based strategy: High volume markets are more efficient
    Bet with the market on high-volume, against on low-volume
    """
    market_prob = market["yes_price"]
    volume = market["volume"]

    if volume > 100000:  # High volume = efficient market
        # Agree with market
        if market_prob > 0.6:
            return market_prob + 0.02
        elif market_prob < 0.4:
            return market_prob - 0.02
    else:  # Low volume = potentially mispriced
        # Slight contrarian
        if market_prob > 0.7:
            return market_prob - 0.05
        elif market_prob < 0.3:
            return market_prob + 0.05

    return None


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Polymarket Scraper & Backtester")
    parser.add_argument("--scrape", action="store_true", help="Scrape all Polymarket data")
    parser.add_argument("--backtest", action="store_true", help="Run backtest on existing data")
    parser.add_argument("--min-volume", type=float, default=1000, help="Minimum volume filter")
    parser.add_argument("--no-history", action="store_true", help="Skip fetching price history")

    args = parser.parse_args()

    if args.scrape:
        print("\n" + "="*60)
        print("SCRAPING POLYMARKET DATA")
        print("="*60)
        data = scrape_all_polymarket_data(
            min_volume=args.min_volume,
            get_history=not args.no_history
        )
        print("\nScraping complete!")
        print(f"Data saved to: {DATA_DIR}")

    elif args.backtest:
        print("\n" + "="*60)
        print("RUNNING BACKTEST")
        print("="*60)

        # Load the most recent resolved markets file
        ensure_data_dir()
        files = [f for f in os.listdir(DATA_DIR) if f.startswith("polymarket_resolved_")]

        if not files:
            print("No data found! Run with --scrape first.")
        else:
            latest_file = sorted(files)[-1]
            filepath = os.path.join(DATA_DIR, latest_file)
            print(f"Loading: {latest_file}")

            markets_df = pd.read_csv(filepath)
            print(f"Loaded {len(markets_df)} resolved markets")

            # Run backtest with different strategies
            strategies = [
                ("Follow the Crowd", strategy_follow_the_crowd),
                ("Contrarian", strategy_contrarian),
                ("Volume Momentum", strategy_volume_momentum),
            ]

            for name, strategy in strategies:
                print(f"\n--- Strategy: {name} ---")
                backtester = KellyBacktester(initial_bankroll=10000, kelly_fraction=0.25)
                trades_df = backtester.backtest(markets_df, strategy)

                if len(trades_df) > 0:
                    stats = backtester.get_stats()
                    print(f"Total Trades: {stats['total_trades']}")
                    print(f"Win Rate: {stats['win_rate']:.1%}")
                    print(f"Total PnL: ${stats['total_pnl']:,.2f}")
                    print(f"Return: {stats['return_pct']:.1f}%")
                    print(f"Max Drawdown: {stats['max_drawdown_pct']:.1f}%")
                    print(f"Final Bankroll: ${stats['final_bankroll']:,.2f}")
                else:
                    print("No trades executed")

    else:
        # Default: scrape and then backtest
        print("\n" + "="*60)
        print("POLYMARKET SCRAPER & BACKTESTER")
        print("="*60)
        print("\nUsage:")
        print("  python polymarket_scraper.py --scrape          # Scrape all data")
        print("  python polymarket_scraper.py --backtest        # Run backtest")
        print("  python polymarket_scraper.py --scrape --backtest  # Both")
        print("\nOptions:")
        print("  --min-volume 1000    # Filter by minimum volume")
        print("  --no-history         # Skip price history (faster)")
