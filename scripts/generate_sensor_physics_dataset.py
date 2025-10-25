#!/usr/bin/env python3
"""Generate synthetic sensor/physics measurement dataset."""
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sensor_data(n_rows: int, n_sensors: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic physics sensor measurements (numeric-heavy)."""
    np.random.seed(seed)
    
    sensor_ids = [f"SENSOR_{i:04d}" for i in range(n_sensors)]
    start_time = datetime(2024, 1, 1)
    
    data = []
    for i in range(n_rows):
        sensor_id = np.random.choice(sensor_ids)
        timestamp = (start_time + timedelta(seconds=i * 10)).isoformat()
        
        # Physics measurements (all numeric, continuous)
        temperature_c = np.random.normal(22, 3)
        humidity_pct = np.clip(np.random.normal(60, 15), 0, 100)
        pressure_hpa = np.random.normal(1013, 10)
        acceleration_x = np.random.normal(0, 0.5)
        acceleration_y = np.random.normal(0, 0.5)
        acceleration_z = np.random.normal(9.81, 0.3)
        magnetic_field_x = np.random.normal(0, 50)
        magnetic_field_y = np.random.normal(0, 50)
        magnetic_field_z = np.random.normal(0, 50)
        light_lux = np.random.lognormal(5, 2)
        voltage = np.random.normal(3.3, 0.1)
        current_ma = np.random.exponential(50)
        power_mw = voltage * current_ma
        
        data.append({
            'timestamp': timestamp,
            'sensor_id': sensor_id,
            'temperature_c': round(temperature_c, 3),
            'humidity_pct': round(humidity_pct, 2),
            'pressure_hpa': round(pressure_hpa, 2),
            'acceleration_x': round(acceleration_x, 4),
            'acceleration_y': round(acceleration_y, 4),
            'acceleration_z': round(acceleration_z, 4),
            'magnetic_field_x': round(magnetic_field_x, 2),
            'magnetic_field_y': round(magnetic_field_y, 2),
            'magnetic_field_z': round(magnetic_field_z, 2),
            'light_lux': round(light_lux, 2),
            'voltage': round(voltage, 3),
            'current_ma': round(current_ma, 2),
            'power_mw': round(power_mw, 2)
        })
    
    return pd.DataFrame(data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, default=100000)
    parser.add_argument('--sensors', type=int, default=100)
    parser.add_argument('--out', type=str, default='data/sensor_physics_100000.csv')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    print(f"Generating {args.rows} rows of sensor physics data...")
    df = generate_sensor_data(args.rows, args.sensors, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")
    print(f"Shape: {df.shape}")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"Numeric columns: {len(numeric_cols)}/{len(df.columns)} ({len(numeric_cols)/len(df.columns)*100:.1f}%)")
    print(f"\nFile size: {len(open(args.out, 'rb').read()) / 1024 / 1024:.2f} MB")
