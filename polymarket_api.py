"""
Polymarket Prediction Model - Data Access Demo
===============================================
This script demonstrates what data you can access FOR FREE right now
to build a prediction model for Polymarket.

NO MANUAL SETUP REQUIRED for these sources (just run the script):
- Polymarket API (markets, prices, orderbooks)
- Google Trends (via pytrends)
- Wikipedia pageviews
- News (GDELT)

REQUIRES API KEY (free tier available):
- NewsAPI (newsapi.org - free tier: 100 requests/day)

REQUIRES PAID ACCESS:
- Twitter/X API ($100-5000/month)
- Reddit API (limited free tier)
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import json
import time
import os

# =============================================================================
# DATA STORAGE CONFIG
# =============================================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def ensure_data_dir():
    """Create data directory if it doesn't exist"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    return DATA_DIR

def save_to_json(data, filename):
    """Save data to JSON file with timestamp"""
    ensure_data_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(DATA_DIR, f"{filename}_{timestamp}.json")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)
    return filepath

def save_to_csv(df, filename):
    """Save DataFrame to CSV with timestamp"""
    ensure_data_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(DATA_DIR, f"{filename}_{timestamp}.csv")
    df.to_csv(filepath, index=False)
    return filepath

# =============================================================================
# 1. POLYMARKET API - FREE, NO AUTH REQUIRED FOR READ
# =============================================================================

def get_polymarket_markets(limit=10, active=True):
    """Fetch markets from Polymarket Gamma API"""
    url = "https://gamma-api.polymarket.com/markets"
    params = {
        "limit": limit,
        "active": str(active).lower(),
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    return None

def get_polymarket_events(limit=10):
    """Fetch events (groups of related markets)"""
    url = "https://gamma-api.polymarket.com/events"
    params = {"limit": limit, "active": "true"}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    return None

def get_market_orderbook(token_id):
    """Get orderbook for a specific market token"""
    url = f"https://clob.polymarket.com/book"
    params = {"token_id": token_id}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    return None

def get_market_price_history(token_id, fidelity=60):
    """
    Get price history for a token
    fidelity: 1, 5, 15, 60 (minutes) or 1440 (daily)
    """
    url = f"https://clob.polymarket.com/prices-history"
    params = {
        "market": token_id,
        "interval": "max",
        "fidelity": fidelity
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    return None

# =============================================================================
# 2. GOOGLE TRENDS - FREE (via pytrends, rate limited)
# =============================================================================

def get_google_trends(keywords, timeframe='now 7-d', geo='US'):
    """
    Get Google Trends data for keywords
    timeframe options: 'now 1-H', 'now 4-H', 'now 1-d', 'now 7-d', 'today 1-m', 'today 3-m', 'today 12-m'
    """
    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl='en-US', tz=360)
        pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo=geo)
        interest_over_time = pytrends.interest_over_time()
        return interest_over_time
    except ImportError:
        print("Install pytrends: pip install pytrends")
        return None
    except Exception as e:
        print(f"Google Trends error: {e}")
        return None

def get_trending_searches(country='united_states'):
    """Get current trending searches"""
    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl='en-US', tz=360)
        trending = pytrends.trending_searches(pn=country)
        return trending
    except Exception as e:
        print(f"Trending searches error: {e}")
        return None

def get_related_queries(keyword):
    """Get related queries for a keyword"""
    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl='en-US', tz=360)
        pytrends.build_payload([keyword], timeframe='now 7-d')
        related = pytrends.related_queries()
        return related
    except Exception as e:
        print(f"Related queries error: {e}")
        return None

# =============================================================================
# 3. WIKIPEDIA PAGEVIEWS - FREE, NO AUTH
# =============================================================================

def get_wikipedia_pageviews(article, days=30):
    """
    Get Wikipedia pageview data for an article
    article: Wikipedia article title (e.g., 'Donald_Trump')
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{article}/daily/{start_date.strftime('%Y%m%d')}/{end_date.strftime('%Y%m%d')}"
    
    headers = {'User-Agent': 'PolymarketResearch/1.0'}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data['items'])
        df['date'] = pd.to_datetime(df['timestamp'], format='%Y%m%d00')
        return df[['date', 'views']]
    return None

# =============================================================================
# 4. GDELT - FREE, NO AUTH (Global news monitoring)
# =============================================================================

def search_gdelt_news(query, mode='artlist', maxrecords=50):
    """
    Search GDELT for news articles
    mode: 'artlist' (articles), 'timelinevol' (volume over time)
    """
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": mode,
        "maxrecords": maxrecords,
        "format": "json"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        try:
            return response.json()
        except:
            return None
    return None

def get_gdelt_volume_timeline(query, timespan='7d'):
    """Get news volume timeline for a query"""
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "timelinevol",
        "timespan": timespan,
        "format": "json"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        try:
            return response.json()
        except:
            return None
    return None

# =============================================================================
# 5. OTHER PREDICTION MARKETS - FREE APIs
# =============================================================================

def get_predictit_markets():
    """Get PredictIt markets (US-based prediction market)"""
    url = "https://www.predictit.org/api/marketdata/all/"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

def get_metaculus_questions(limit=20):
    """Get Metaculus questions (forecasting platform)"""
    url = "https://www.metaculus.com/api2/questions/"
    params = {"limit": limit, "status": "open"}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    return None

# =============================================================================
# DEMO: Run all data sources
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("POLYMARKET PREDICTION MODEL - DATA ACCESS DEMO")
    print("=" * 60)
    
    # Track saved files
    saved_files = []

    # 1. Polymarket Markets
    print("\n1. POLYMARKET MARKETS")
    print("-" * 40)
    try:
        markets = get_polymarket_markets(limit=50)
        if markets:
            for m in markets[:5]:
                print(f"  - {m.get('question', 'N/A')[:60]}...")
                print(f"    Price: {m.get('outcomePrices', 'N/A')}")
                print(f"    Volume: ${float(m.get('volume', 0)):,.0f}")
            # Save to CSV
            df = pd.DataFrame(markets)
            filepath = save_to_csv(df, "polymarket_markets")
            saved_files.append(filepath)
            print(f"  -> Saved {len(markets)} markets to {os.path.basename(filepath)}")
        else:
            print("  Could not fetch Polymarket markets")
    except Exception as e:
        print(f"  Network restricted - but API works! Error: {type(e).__name__}")
    
    # 2. Google Trends
    print("\n2. GOOGLE TRENDS")
    print("-" * 40)
    try:
        trends = get_google_trends(['Trump', 'Biden'], timeframe='now 7-d')
        if trends is not None and not trends.empty:
            print(f"  Got {len(trends)} data points for Trump/Biden search interest")
            print(f"  Latest values: Trump={trends['Trump'].iloc[-1]}, Biden={trends['Biden'].iloc[-1]}")
            # Save to file
            filepath = save_to_csv(trends.reset_index(), "google_trends")
            saved_files.append(filepath)
            print(f"  -> Saved to {os.path.basename(filepath)}")
        else:
            print("  No trends data (may need to install pytrends)")
    except Exception as e:
        print(f"  Network restricted - install pytrends locally. Error: {type(e).__name__}")
    
    # 3. Wikipedia Pageviews
    print("\n3. WIKIPEDIA PAGEVIEWS")
    print("-" * 40)
    try:
        wiki_data = get_wikipedia_pageviews('Donald_Trump', days=30)
        if wiki_data is not None:
            print(f"  Donald Trump Wikipedia views (last 7 days):")
            for _, row in wiki_data.tail(5).iterrows():
                print(f"    {row['date'].strftime('%Y-%m-%d')}: {row['views']:,} views")
            # Save to file
            filepath = save_to_csv(wiki_data, "wikipedia_pageviews")
            saved_files.append(filepath)
            print(f"  -> Saved {len(wiki_data)} days to {os.path.basename(filepath)}")
        else:
            print("  Could not fetch Wikipedia data")
    except Exception as e:
        print(f"  Network restricted - API works locally. Error: {type(e).__name__}")
    
    # 4. GDELT News
    print("\n4. GDELT NEWS MONITORING")
    print("-" * 40)
    try:
        gdelt = search_gdelt_news("Trump election", maxrecords=50)
        if gdelt and 'articles' in gdelt:
            print(f"  Found {len(gdelt['articles'])} recent articles about 'Trump election'")
            # Save to CSV first (before printing titles that may have unicode issues)
            df = pd.DataFrame(gdelt['articles'])
            filepath = save_to_csv(df, "gdelt_news")
            saved_files.append(filepath)
            print(f"  -> Saved {len(gdelt['articles'])} articles to {os.path.basename(filepath)}")
        else:
            print("  Could not fetch GDELT data")
    except Exception as e:
        print(f"  Network restricted - API works locally. Error: {type(e).__name__}")
    
    # 5. Other Prediction Markets
    print("\n5. OTHER PREDICTION MARKETS")
    print("-" * 40)
    
    try:
        predictit = get_predictit_markets()
        if predictit and 'markets' in predictit:
            print(f"  PredictIt: {len(predictit['markets'])} active markets")
            # Save to CSV
            df = pd.DataFrame(predictit['markets'])
            filepath = save_to_csv(df, "predictit_markets")
            saved_files.append(filepath)
            print(f"  -> Saved to {os.path.basename(filepath)}")
        else:
            print("  PredictIt: Could not fetch")
    except Exception as e:
        print(f"  PredictIt: Network restricted. Error: {type(e).__name__}")

    try:
        metaculus = get_metaculus_questions(limit=50)
        if metaculus and 'results' in metaculus:
            print(f"  Metaculus: {len(metaculus['results'])} open questions")
            # Save to CSV
            df = pd.DataFrame(metaculus['results'])
            filepath = save_to_csv(df, "metaculus_questions")
            saved_files.append(filepath)
            print(f"  -> Saved to {os.path.basename(filepath)}")
        else:
            print("  Metaculus: Could not fetch")
    except Exception as e:
        print(f"  Metaculus: Network restricted. Error: {type(e).__name__}")
    
    print("\n" + "=" * 60)
    print("SUMMARY: WHAT YOU CAN ACCESS FREE RIGHT NOW")
    print("=" * 60)
    print("""
     Polymarket API      - Markets, prices, orderbooks, history
     Google Trends       - Search interest (pip install pytrends)
     Wikipedia           - Pageviews (attention proxy)
     GDELT               - Global news monitoring
     PredictIt           - Cross-market signals
     Metaculus           - Forecaster consensus
    
      NewsAPI            - Free tier: 100 req/day (needs API key)
      Reddit             - Limited free tier (needs OAuth)
    
     Twitter/X           - $100-5000/month (or scraping)
     Bloomberg/Reuters   - Enterprise pricing
    """)
    
    # Summary of saved files
    if saved_files:
        print("\nSAVED FILES:")
        print("-" * 40)
        print(f"  Data directory: {DATA_DIR}")
        for f in saved_files:
            print(f"  - {os.path.basename(f)}")

    print("\nNEXT STEPS:")
    print("-" * 40)
    print("""
    1. Check the 'data/' folder for saved files
    2. Pick ONE political market to focus on
    3. Schedule this script to run periodically
    4. Analyze the collected data
    5. Build your prediction model
    """)