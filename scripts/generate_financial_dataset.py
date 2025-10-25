#!/usr/bin/env python3
"""Generate synthetic financial/stock market dataset for benchmarking."""
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_financial_data(n_rows: int, n_tickers: int = 50, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic stock market data."""
    np.random.seed(seed)
    
    # Generate ticker symbols
    tickers = [f"TICK{i:03d}" for i in range(n_tickers)]
    
    # Base date
    start_date = datetime(2024, 1, 1)
    
    data = []
    for i in range(n_rows):
        ticker = np.random.choice(tickers)
        date = start_date + timedelta(days=i // n_tickers)
        
        # Generate realistic price patterns
        base_price = np.random.uniform(10, 500)
        volatility = np.random.uniform(0.01, 0.05)
        
        open_price = base_price * (1 + np.random.normal(0, volatility))
        high_price = open_price * (1 + abs(np.random.normal(0, volatility)))
        low_price = open_price * (1 - abs(np.random.normal(0, volatility)))
        close_price = np.random.uniform(low_price, high_price)
        
        volume = int(np.random.lognormal(15, 1.5))
        
        # Additional financial metrics
        market_cap = close_price * volume * np.random.uniform(50, 1000)
        pe_ratio = np.random.uniform(5, 50)
        dividend_yield = np.random.uniform(0, 0.08)
        
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'ticker': ticker,
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': volume,
            'market_cap': round(market_cap, 2),
            'pe_ratio': round(pe_ratio, 2),
            'dividend_yield': round(dividend_yield, 4)
        })
    
    df = pd.DataFrame(data)
    return df

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic financial dataset')
    parser.add_argument('--rows', type=int, default=50000, help='Number of rows')
    parser.add_argument('--tickers', type=int, default=50, help='Number of unique tickers')
    parser.add_argument('--out', type=str, default='data/financial_50000.csv', help='Output CSV path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print(f"Generating {args.rows} rows of financial data...")
    df = generate_financial_data(args.rows, args.tickers, args.seed)
    
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSample:")
    print(df.head())
    print(f"\nFile size: {len(open(args.out, 'rb').read()) / 1024 / 1024:.2f} MB")

if __name__ == '__main__':
    main()
