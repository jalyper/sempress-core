#!/usr/bin/env python3
"""Generate synthetic ML feature store dataset for benchmarking."""
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_ml_features(n_rows: int, n_users: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic ML feature store with continuous numeric features."""
    np.random.seed(seed)
    
    # User IDs and timestamps
    user_ids = [f"user_{i:06d}" for i in range(n_users)]
    start_date = datetime(2024, 1, 1)
    
    data = []
    for i in range(n_rows):
        user_id = np.random.choice(user_ids)
        timestamp = (start_date + timedelta(hours=i // 100)).isoformat()
        
        # Generate realistic ML features (continuous numeric)
        session_duration_min = np.random.exponential(15)
        pages_viewed = int(np.random.poisson(8))
        click_rate = np.random.beta(2, 5)
        scroll_depth_pct = np.random.uniform(0.2, 1.0)
        time_since_last_visit_hours = np.random.exponential(48)
        total_sessions = int(np.random.lognormal(3, 1))
        avg_session_duration = np.random.exponential(12)
        cart_value = np.random.gamma(2, 50)
        conversion_probability = 1 / (1 + np.exp(-np.random.normal(0, 1.5)))
        lifetime_value = np.random.gamma(3, 200)
        device_age_days = int(np.random.exponential(365))
        screen_resolution_area = np.random.choice([1920*1080, 1366*768, 2560*1440, 3840*2160])
        engagement_score = session_duration_min * 0.3 + pages_viewed * 2 + click_rate * 10 + scroll_depth_pct * 5
        
        data.append({
            'timestamp': timestamp,
            'user_id': user_id,
            'session_duration_min': round(session_duration_min, 2),
            'pages_viewed': pages_viewed,
            'click_rate': round(click_rate, 4),
            'scroll_depth_pct': round(scroll_depth_pct, 4),
            'time_since_last_visit_hours': round(time_since_last_visit_hours, 2),
            'total_sessions': total_sessions,
            'avg_session_duration': round(avg_session_duration, 2),
            'cart_value': round(cart_value, 2),
            'conversion_probability': round(conversion_probability, 4),
            'lifetime_value': round(lifetime_value, 2),
            'device_age_days': device_age_days,
            'screen_resolution_area': screen_resolution_area,
            'engagement_score': round(engagement_score, 2)
        })
    
    return pd.DataFrame(data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic ML feature store dataset')
    parser.add_argument('--rows', type=int, default=100000)
    parser.add_argument('--users', type=int, default=1000)
    parser.add_argument('--out', type=str, default='data/ml_features_100000.csv')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    print(f"Generating {args.rows} rows of ML feature data...")
    df = generate_ml_features(args.rows, args.users, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")
    print(f"Shape: {df.shape}")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"Numeric columns: {len(numeric_cols)}/{len(df.columns)} ({len(numeric_cols)/len(df.columns)*100:.1f}%)")
    print(f"\nFile size: {len(open(args.out, 'rb').read()) / 1024 / 1024:.2f} MB")
