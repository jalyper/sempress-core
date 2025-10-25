from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import gzip

import pandas as pd

from sempress.metrics import eval_table
from sempress.table_decoder import decode_to_csv
from sempress.table_encoder import EncodeConfig, encode_csv

ROOT_DIR = Path(__file__).resolve().parents[1]


def _run_generator(dataset: str, out_path: Path, rows: int, seed: int = 42) -> None:
    script_path = ROOT_DIR / "scripts" / "generate_datasets.py"
    cmd = [
        sys.executable,
        str(script_path),
        dataset,
        "--rows",
        str(rows),
        "--seed",
        str(seed),
        "--out",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def test_telemetry_roundtrip(tmp_path):
    csv_path = tmp_path / "telemetry.csv"
    recon_path = tmp_path / "telemetry_recon.csv"
    blob_path = tmp_path / "telemetry.smp"

    _run_generator("telemetry", csv_path, rows=8000, seed=11)

    cfg = EncodeConfig(
        lock_cols=["device_id", "timestamp"],
        residual_cols=["temp_c"],
        k=64,
        uncertainty_thresh=0.15,
    )
    blob = encode_csv(str(csv_path), cfg)
    blob_path.write_bytes(blob)

    meta = decode_to_csv(str(blob_path), str(recon_path))

    orig = pd.read_csv(csv_path)
    recon = pd.read_csv(recon_path)

    assert orig.shape == recon.shape
    assert set(orig.columns) == set(recon.columns)

    # Locked columns should match exactly
    for col in cfg.lock_cols:
        assert orig[col].equals(recon[col])

    # Compression should beat every baseline codec comfortably
    raw_bytes = csv_path.read_bytes()
    ratio = len(raw_bytes) / max(len(blob), 1)
    assert ratio > 5.0

    metrics = eval_table(orig, recon, cfg.lock_cols)
    assert metrics["temp_c"]["rmse"] < 0.5
    assert metrics["power_kw"]["rmse"] < 2.0

    uncertain = meta.get("uncertain", {})
    total_uncertain = sum(len(indices) for indices in uncertain.values())
    total_cells = recon.shape[0] * (recon.shape[1] - len(cfg.lock_cols))
    if total_cells:
        assert (total_uncertain / total_cells) < 0.02


def test_retail_roundtrip(tmp_path):
    csv_path = tmp_path / "retail.csv"
    recon_path = tmp_path / "retail_recon.csv"
    blob_path = tmp_path / "retail.smp"

    _run_generator("retail", csv_path, rows=6000, seed=5)

    cfg = EncodeConfig(
        lock_cols=["order_id", "timestamp", "customer_id", "region", "category"],
        residual_cols=["unit_price"],
        k=48,
        uncertainty_thresh=0.2,
    )
    blob = encode_csv(str(csv_path), cfg)
    blob_path.write_bytes(blob)

    decode_to_csv(str(blob_path), str(recon_path))

    orig = pd.read_csv(csv_path)
    recon = pd.read_csv(recon_path)

    # Lossless lock columns
    for col in cfg.lock_cols:
        assert orig[col].equals(recon[col])

    # Numeric fidelity
    metrics = eval_table(orig, recon, cfg.lock_cols)
    assert metrics["quantity"]["rmse"] < 0.3
    assert metrics["unit_price"]["rmse"] < 3.0

    # Compression sanity
    raw_bytes = csv_path.read_bytes()
    ratio = len(raw_bytes) / max(len(blob), 1)
    assert ratio > 4.0


def test_lock_all_columns_vs_gzip(tmp_path):
    csv_path = tmp_path / "lock_all.csv"
    recon_path = tmp_path / "lock_all_recon.csv"
    blob_path = tmp_path / "lock_all.smp"

    _run_generator("retail", csv_path, rows=3000, seed=9)
    raw_df = pd.read_csv(csv_path)
    lock_cols = list(raw_df.columns)

    cfg = EncodeConfig(
        lock_cols=lock_cols,
        residual_cols=[],
        k=16,
        uncertainty_thresh=0.0,
    )
    blob = encode_csv(str(csv_path), cfg)
    blob_path.write_bytes(blob)

    decode_to_csv(str(blob_path), str(recon_path))
    recon_df = pd.read_csv(recon_path)

    pd.testing.assert_frame_equal(raw_df, recon_df)

    sempress_size = len(blob)
    gzip_size = len(gzip.compress(csv_path.read_bytes()))

    assert sempress_size <= gzip_size * 1.3
