#!/usr/bin/env python3
"""Unified synthetic dataset generator for Sempress benchmarks and tests.

Produces deterministic synthetic datasets that exercise the compressor's
strengths (low-cardinality lock columns + smooth numeric residuals) and
weaknesses (high-entropy financial data).

Usage:
    python scripts/generate_datasets.py telemetry --rows 100000 --out data/telemetry_100000.csv
    python scripts/generate_datasets.py retail --rows 100000 --out data/retail_100000.csv
    python scripts/generate_datasets.py financial --rows 50000 --out data/financial_50000.csv
    python scripts/generate_datasets.py sensor_physics --rows 100000 --out data/sensor_physics_100000.csv
    python scripts/generate_datasets.py ml_features --rows 100000 --out data/ml_features_100000.csv
    python scripts/generate_datasets.py all --rows 100000

All generators accept --rows, --seed, and --out. The `all` subcommand
produces the full benchmark suite into data/ with sensible defaults.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Individual generators
# ---------------------------------------------------------------------------

def gen_telemetry(n_rows: int, seed: int = 42) -> pd.DataFrame:
    """IoT device telemetry. Low-cardinality device IDs, smooth sensor readings."""
    rng = np.random.default_rng(seed)
    n_devices = 200
    device_ids = np.array([f"dev-{i:04d}" for i in range(n_devices)])

    start = datetime(2024, 1, 1)
    timestamps = [(start + timedelta(seconds=int(i * 5))).isoformat(timespec="seconds")
                  for i in range(n_rows)]

    device_idx = rng.integers(0, n_devices, size=n_rows)

    # Smooth per-device temperature baseline + diurnal wobble
    device_baselines = rng.normal(22.0, 2.0, size=n_devices)
    time_sec = np.arange(n_rows) * 5.0
    diurnal = 1.5 * np.sin(2 * np.pi * time_sec / 86400.0)
    temp_c = device_baselines[device_idx] + diurnal + rng.normal(0, 0.2, size=n_rows)

    # Power draw correlated with temperature
    power_kw = 1.2 + 0.04 * (temp_c - 20.0) + rng.normal(0, 0.05, size=n_rows)

    humidity_pct = np.clip(55 + 10 * np.sin(2 * np.pi * time_sec / 43200.0)
                            + rng.normal(0, 2.0, size=n_rows), 0, 100)
    pressure_hpa = 1013.0 + rng.normal(0, 1.5, size=n_rows)
    voltage = 12.0 + rng.normal(0, 0.1, size=n_rows)
    signal_dbm = -65.0 + rng.normal(0, 3.0, size=n_rows)

    return pd.DataFrame({
        "device_id": device_ids[device_idx],
        "timestamp": timestamps,
        "temp_c": np.round(temp_c, 3),
        "power_kw": np.round(power_kw, 3),
        "humidity_pct": np.round(humidity_pct, 2),
        "pressure_hpa": np.round(pressure_hpa, 2),
        "voltage": np.round(voltage, 3),
        "signal_dbm": np.round(signal_dbm, 2),
    })


def gen_retail(n_rows: int, seed: int = 42) -> pd.DataFrame:
    """Retail transaction data. String-heavy categorical columns plus numeric amounts."""
    rng = np.random.default_rng(seed)

    regions = np.array(["north", "south", "east", "west", "central"])
    categories = np.array([
        "electronics", "apparel", "grocery", "home", "toys",
        "books", "sports", "beauty", "garden", "automotive"
    ])
    n_customers = 5000
    customer_ids = np.array([f"cust-{i:06d}" for i in range(n_customers)])

    start = datetime(2024, 1, 1)

    order_ids = np.array([f"ord-{i:08d}" for i in range(n_rows)])
    timestamps = [(start + timedelta(minutes=int(i * 3))).isoformat(timespec="minutes")
                  for i in range(n_rows)]

    region = regions[rng.integers(0, len(regions), size=n_rows)]
    category = categories[rng.integers(0, len(categories), size=n_rows)]
    customer_id = customer_ids[rng.integers(0, n_customers, size=n_rows)]

    # Price distributions per category (smooth, clusterable)
    base_price_by_cat = {
        "electronics": 150.0, "apparel": 45.0, "grocery": 12.0, "home": 60.0,
        "toys": 25.0, "books": 18.0, "sports": 80.0, "beauty": 30.0,
        "garden": 35.0, "automotive": 120.0,
    }
    base = np.array([base_price_by_cat[c] for c in category])
    unit_price = base * rng.lognormal(0, 0.3, size=n_rows)

    quantity = rng.integers(1, 6, size=n_rows)
    discount_pct = np.clip(rng.beta(2, 20, size=n_rows), 0, 0.5)
    shipping_cost = np.clip(5.0 + rng.exponential(2.0, size=n_rows), 0, None)

    return pd.DataFrame({
        "order_id": order_ids,
        "timestamp": timestamps,
        "customer_id": customer_id,
        "region": region,
        "category": category,
        "quantity": quantity.astype(int),
        "unit_price": np.round(unit_price, 2),
        "discount_pct": np.round(discount_pct, 4),
        "shipping_cost": np.round(shipping_cost, 2),
    })


def gen_financial(n_rows: int, seed: int = 42) -> pd.DataFrame:
    """OHLC stock data. Noisy by nature — sempress's hardest case."""
    rng = np.random.default_rng(seed)
    n_tickers = 100
    tickers = np.array([f"TICK{i:03d}" for i in range(n_tickers)])
    start = datetime(2024, 1, 1)

    ticker_idx = rng.integers(0, n_tickers, size=n_rows)
    dates = [(start + timedelta(days=int(i // n_tickers))).strftime("%Y-%m-%d")
             for i in range(n_rows)]

    # Per-ticker random walk
    ticker_bases = rng.uniform(20, 400, size=n_tickers)
    volatilities = rng.uniform(0.01, 0.04, size=n_tickers)

    open_prices = ticker_bases[ticker_idx] * (1 + rng.normal(0, volatilities[ticker_idx]))
    spread = np.abs(rng.normal(0, volatilities[ticker_idx] * open_prices))
    high = open_prices + spread
    low = open_prices - spread * rng.uniform(0.5, 1.5, size=n_rows)
    low = np.maximum(low, 0.01)
    close = rng.uniform(low, high)

    volume = rng.lognormal(15, 1.2, size=n_rows).astype(int)
    market_cap = close * volume * rng.uniform(50, 1000, size=n_rows)
    pe_ratio = rng.uniform(5, 50, size=n_rows)
    dividend_yield = rng.uniform(0, 0.08, size=n_rows)

    return pd.DataFrame({
        "date": dates,
        "ticker": tickers[ticker_idx],
        "open": np.round(open_prices, 2),
        "high": np.round(high, 2),
        "low": np.round(low, 2),
        "close": np.round(close, 2),
        "volume": volume,
        "market_cap": np.round(market_cap, 2),
        "pe_ratio": np.round(pe_ratio, 2),
        "dividend_yield": np.round(dividend_yield, 4),
    })


def gen_sensor_physics(n_rows: int, seed: int = 42) -> pd.DataFrame:
    """Physics sensor array. All-numeric, smooth continuous measurements."""
    rng = np.random.default_rng(seed)
    n_sensors = 100
    sensor_ids = np.array([f"SENSOR_{i:04d}" for i in range(n_sensors)])
    start = datetime(2024, 1, 1)

    sensor_idx = rng.integers(0, n_sensors, size=n_rows)
    timestamps = [(start + timedelta(seconds=int(i * 10))).isoformat(timespec="seconds")
                  for i in range(n_rows)]

    temperature_c = rng.normal(22, 3, size=n_rows)
    humidity_pct = np.clip(rng.normal(60, 15, size=n_rows), 0, 100)
    pressure_hpa = rng.normal(1013, 10, size=n_rows)
    ax = rng.normal(0, 0.5, size=n_rows)
    ay = rng.normal(0, 0.5, size=n_rows)
    az = rng.normal(9.81, 0.3, size=n_rows)
    mx = rng.normal(0, 50, size=n_rows)
    my = rng.normal(0, 50, size=n_rows)
    mz = rng.normal(0, 50, size=n_rows)
    light_lux = rng.lognormal(5, 2, size=n_rows)
    voltage = rng.normal(3.3, 0.1, size=n_rows)
    current_ma = rng.exponential(50, size=n_rows)
    power_mw = voltage * current_ma

    return pd.DataFrame({
        "timestamp": timestamps,
        "sensor_id": sensor_ids[sensor_idx],
        "temperature_c": np.round(temperature_c, 3),
        "humidity_pct": np.round(humidity_pct, 2),
        "pressure_hpa": np.round(pressure_hpa, 2),
        "acceleration_x": np.round(ax, 4),
        "acceleration_y": np.round(ay, 4),
        "acceleration_z": np.round(az, 4),
        "magnetic_field_x": np.round(mx, 2),
        "magnetic_field_y": np.round(my, 2),
        "magnetic_field_z": np.round(mz, 2),
        "light_lux": np.round(light_lux, 2),
        "voltage": np.round(voltage, 3),
        "current_ma": np.round(current_ma, 2),
        "power_mw": np.round(power_mw, 2),
    })


def gen_ml_features(n_rows: int, seed: int = 42) -> pd.DataFrame:
    """User behavior feature store. Mix of integers, floats, categorical."""
    rng = np.random.default_rng(seed)
    n_users = 2000
    user_ids = np.array([f"user_{i:06d}" for i in range(n_users)])
    start = datetime(2024, 1, 1)

    user_idx = rng.integers(0, n_users, size=n_rows)
    timestamps = [(start + timedelta(hours=int(i // 100))).isoformat(timespec="minutes")
                  for i in range(n_rows)]

    session_duration_min = rng.exponential(15, size=n_rows)
    pages_viewed = rng.poisson(8, size=n_rows).astype(int)
    click_rate = rng.beta(2, 5, size=n_rows)
    scroll_depth_pct = rng.uniform(0.2, 1.0, size=n_rows)
    time_since_last_visit_hours = rng.exponential(48, size=n_rows)
    total_sessions = rng.lognormal(3, 1, size=n_rows).astype(int)
    avg_session_duration = rng.exponential(12, size=n_rows)
    cart_value = rng.gamma(2, 50, size=n_rows)
    conversion_probability = 1 / (1 + np.exp(-rng.normal(0, 1.5, size=n_rows)))
    lifetime_value = rng.gamma(3, 200, size=n_rows)
    device_age_days = rng.exponential(365, size=n_rows).astype(int)
    engagement_score = (session_duration_min * 0.3 + pages_viewed * 2
                         + click_rate * 10 + scroll_depth_pct * 5)

    return pd.DataFrame({
        "timestamp": timestamps,
        "user_id": user_ids[user_idx],
        "session_duration_min": np.round(session_duration_min, 2),
        "pages_viewed": pages_viewed,
        "click_rate": np.round(click_rate, 4),
        "scroll_depth_pct": np.round(scroll_depth_pct, 4),
        "time_since_last_visit_hours": np.round(time_since_last_visit_hours, 2),
        "total_sessions": total_sessions,
        "avg_session_duration": np.round(avg_session_duration, 2),
        "cart_value": np.round(cart_value, 2),
        "conversion_probability": np.round(conversion_probability, 4),
        "lifetime_value": np.round(lifetime_value, 2),
        "device_age_days": device_age_days,
        "engagement_score": np.round(engagement_score, 2),
    })


# ---------------------------------------------------------------------------
# Registry & CLI
# ---------------------------------------------------------------------------

GENERATORS = {
    "telemetry": gen_telemetry,
    "retail": gen_retail,
    "financial": gen_financial,
    "sensor_physics": gen_sensor_physics,
    "ml_features": gen_ml_features,
}

DEFAULT_OUT = {
    "telemetry": "data/telemetry_100000.csv",
    "retail": "data/retail_100000.csv",
    "financial": "data/financial_50000.csv",
    "sensor_physics": "data/sensor_physics_100000.csv",
    "ml_features": "data/ml_features_100000.csv",
}

DEFAULT_ROWS = {
    "telemetry": 100_000,
    "retail": 100_000,
    "financial": 50_000,
    "sensor_physics": 100_000,
    "ml_features": 100_000,
}


def _write(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  wrote {out_path}  ({len(df):,} rows, {len(df.columns)} cols, {size_mb:.2f} MB)")


def _generate_one(name: str, rows: int, seed: int, out: Path) -> None:
    print(f"generating {name}: {rows:,} rows (seed={seed})")
    df = GENERATORS[name](rows, seed)
    _write(df, out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified synthetic dataset generator for Sempress",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="dataset", required=True)

    for name in list(GENERATORS.keys()) + ["all"]:
        sp = sub.add_parser(name, help=f"generate {name} dataset")
        sp.add_argument("--rows", type=int, default=None)
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--out", type=str, default=None)

    args = parser.parse_args()

    if args.dataset == "all":
        for name in GENERATORS:
            rows = args.rows or DEFAULT_ROWS[name]
            out = Path(args.out) / DEFAULT_OUT[name] if args.out else Path(DEFAULT_OUT[name])
            _generate_one(name, rows, args.seed, out)
    else:
        name = args.dataset
        rows = args.rows or DEFAULT_ROWS[name]
        out = Path(args.out) if args.out else Path(DEFAULT_OUT[name])
        _generate_one(name, rows, args.seed, out)


if __name__ == "__main__":
    main()
