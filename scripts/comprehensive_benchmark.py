#!/usr/bin/env python3
"""Comprehensive benchmark comparing Sempress against all baselines."""
import argparse
import gzip
import bz2
import lzma
import time
import json
import subprocess
from pathlib import Path
import pandas as pd
import zstandard as zstd

from sempress import encode_csv, decode_to_csv
from sempress.table_encoder import EncodeConfig

def measure_compression(input_path: Path, config: EncodeConfig) -> dict:
    """Measure Sempress compression metrics."""
    start = time.time()
    blob = encode_csv(str(input_path), config)
    encode_time = time.time() - start
    
    output_path = input_path.parent / f"{input_path.stem}_recon.csv"
    smp_path = input_path.parent / f"{input_path.stem}.smp"
    smp_path.write_bytes(blob)
    
    start = time.time()
    decode_to_csv(blob, str(output_path))
    decode_time = time.time() - start
    
    original_size = input_path.stat().st_size
    compressed_size = len(blob)
    ratio = original_size / compressed_size
    
    return {
        'compressed_size': compressed_size,
        'ratio': round(ratio, 3),
        'encode_time': round(encode_time, 3),
        'decode_time': round(decode_time, 3)
    }

def measure_baseline(input_path: Path, method: str) -> dict:
    """Measure baseline compression (gzip, bz2, lzma, zstd)."""
    data = input_path.read_bytes()
    original_size = len(data)
    
    if method == 'gzip':
        start = time.time()
        compressed = gzip.compress(data, compresslevel=6)
        compress_time = time.time() - start
        
        start = time.time()
        gzip.decompress(compressed)
        decompress_time = time.time() - start
        
    elif method == 'bz2':
        start = time.time()
        compressed = bz2.compress(data, compresslevel=9)
        compress_time = time.time() - start
        
        start = time.time()
        bz2.decompress(compressed)
        decompress_time = time.time() - start
        
    elif method == 'lzma':
        start = time.time()
        compressed = lzma.compress(data, preset=6)
        compress_time = time.time() - start
        
        start = time.time()
        lzma.decompress(compressed)
        decompress_time = time.time() - start
        
    elif method == 'zstd':
        compressor = zstd.ZstdCompressor(level=3)
        start = time.time()
        compressed = compressor.compress(data)
        compress_time = time.time() - start
        
        decompressor = zstd.ZstdDecompressor()
        start = time.time()
        decompressor.decompress(compressed)
        decompress_time = time.time() - start
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    compressed_size = len(compressed)
    ratio = original_size / compressed_size
    
    return {
        'compressed_size': compressed_size,
        'ratio': round(ratio, 3),
        'compress_time': round(compress_time, 3),
        'decompress_time': round(decompress_time, 3)
    }

def benchmark_dataset(input_path: Path, lock_cols: list, residual_cols: list, k: int = 64) -> dict:
    """Run comprehensive benchmark on a single dataset."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {input_path.name}")
    print(f"{'='*60}")
    
    # Count rows
    df = pd.read_csv(input_path)
    n_rows = len(df)
    n_cols = len(df.columns)
    original_size = input_path.stat().st_size
    
    print(f"Rows: {n_rows:,} | Columns: {n_cols} | Size: {original_size/1024/1024:.2f} MB")
    
    # Sempress
    print("\n[1/5] Running Sempress...")
    config = EncodeConfig(
        lock_cols=lock_cols,
        residual_cols=residual_cols,
        k=k,
        uncertainty_thresh=0.2,
        random_state=42
    )
    sempress_result = measure_compression(input_path, config)
    print(f"  Ratio: {sempress_result['ratio']:.2f}x | "
          f"Encode: {sempress_result['encode_time']:.3f}s | "
          f"Decode: {sempress_result['decode_time']:.3f}s")
    
    # Baselines
    baselines = {}
    for method in ['gzip', 'bz2', 'lzma', 'zstd']:
        print(f"\n[{list(['gzip', 'bz2', 'lzma', 'zstd']).index(method) + 2}/5] Running {method}...")
        result = measure_baseline(input_path, method)
        baselines[method] = result
        print(f"  Ratio: {result['ratio']:.2f}x | "
              f"Compress: {result['compress_time']:.3f}s | "
              f"Decompress: {result['decompress_time']:.3f}s")
    
    # Find best baseline
    best_baseline_method = max(baselines.items(), key=lambda x: x[1]['ratio'])[0]
    best_baseline_ratio = baselines[best_baseline_method]['ratio']
    improvement = ((sempress_result['ratio'] - best_baseline_ratio) / best_baseline_ratio) * 100
    
    print(f"\n{'─'*60}")
    print(f"SUMMARY:")
    print(f"  Sempress: {sempress_result['ratio']:.2f}x")
    print(f"  Best baseline ({best_baseline_method}): {best_baseline_ratio:.2f}x")
    print(f"  Improvement: {improvement:+.1f}%")
    print(f"{'─'*60}")
    
    return {
        'dataset': input_path.stem,
        'rows': n_rows,
        'cols': n_cols,
        'original_size': original_size,
        'sempress': sempress_result,
        'baselines': baselines,
        'best_baseline_method': best_baseline_method,
        'best_baseline_ratio': best_baseline_ratio,
        'improvement_pct': round(improvement, 1)
    }

def main():
    parser = argparse.ArgumentParser(description='Comprehensive compression benchmark')
    parser.add_argument('--out', type=str, default='logs/comprehensive_benchmark.json',
                      help='Output JSON path')
    args = parser.parse_args()
    
    # Datasets documented in the README benchmark table.
    # `lock_cols` = lossless (strings, IDs, timestamps, categoricals).
    # `residual_cols` = columns where lossy quantization is NOT acceptable
    #                    and exact reconstruction is needed. These store a
    #                    per-row float32 delta (4 bytes/row) so they cost a
    #                    lot of space — keep them minimal.
    # Every other numeric column is quantized to a 64-entry K-Means codebook
    # (2 bytes/row + tiny codebook), which is where the compression wins come
    # from.
    datasets = [
        {
            'label': 'IoT Telemetry',
            'path': Path('data/telemetry_100000.csv'),
            'lock_cols': ['device_id', 'timestamp'],
            'residual_cols': [],
        },
        {
            'label': 'Sensor Physics',
            'path': Path('data/sensor_physics_100000.csv'),
            'lock_cols': ['timestamp', 'sensor_id'],
            'residual_cols': [],
        },
        {
            'label': 'ML Features',
            'path': Path('data/ml_features_100000.csv'),
            'lock_cols': ['timestamp', 'user_id'],
            'residual_cols': [],
        },
        {
            'label': 'Financial Data',
            'path': Path('data/financial_50000.csv'),
            'lock_cols': ['date', 'ticker'],
            'residual_cols': [],
        },
    ]
    
    results = []
    for dataset_config in datasets:
        path = dataset_config['path']
        if not path.exists():
            print(f"\nSkipping {path.name} (not found)")
            continue
        
        result = benchmark_dataset(
            path,
            dataset_config['lock_cols'],
            dataset_config['residual_cols']
        )
        results.append(result)
    
    # Save results
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to {output_path}")
    print(f"{'='*60}")
    
    # Print summary table
    print("\n\nCOMPRESSION RATIO SUMMARY")
    print("─" * 80)
    print(f"{'Dataset':<20} {'Rows':>10} {'Sempress':>10} {'Gzip':>10} {'Bzip2':>10} {'Zstd':>10} {'Best':>10} {'Lift':>8}")
    print("─" * 80)
    
    for result in results:
        print(f"{result['dataset']:<20} "
              f"{result['rows']:>10,} "
              f"{result['sempress']['ratio']:>10.2f}× "
              f"{result['baselines']['gzip']['ratio']:>10.2f}× "
              f"{result['baselines']['bz2']['ratio']:>10.2f}× "
              f"{result['baselines']['zstd']['ratio']:>10.2f}× "
              f"{result['best_baseline_ratio']:>10.2f}× "
              f"{result['improvement_pct']:>+7.1f}%")
    
    print("─" * 80)

if __name__ == '__main__':
    main()
