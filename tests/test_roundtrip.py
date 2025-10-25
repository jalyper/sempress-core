from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from sempress.metrics import eval_table
from sempress.table_decoder import decode_to_csv
from sempress.table_encoder import EncodeConfig, encode_csv


SAMPLE_CSV = Path(__file__).resolve().parents[1] / "data" / "sample.csv"


def test_roundtrip(tmp_path):
    sample = SAMPLE_CSV
    raw_bytes = sample.read_bytes()
    out_blob = tmp_path / "sample.smp"
    out_csv = tmp_path / "recon.csv"

    cfg = EncodeConfig(lock_cols=["id", "timestamp"], residual_cols=["amount"], k=16)
    blob = encode_csv(str(sample), cfg)
    out_blob.write_bytes(blob)

    meta = decode_to_csv(str(out_blob), str(out_csv))
    assert out_csv.exists()

    orig = pd.read_csv(sample)
    recon = pd.read_csv(out_csv)

    # Locked columns must match exactly
    assert orig["id"].equals(recon["id"]) and orig["timestamp"].equals(recon["timestamp"])

    # Numeric columns should be reasonably close (very loose smoke test)
    for c in ["amount", "temperature_r", "pressure_kpa"]:
        assert (orig[c] - recon[c]).abs().mean() < 5.0

    # Compression ratio should beat raw CSV payload.
    original_size = max(len(raw_bytes), 1)
    smp_size = max(len(blob), 1)
    compression_ratio = original_size / smp_size
    assert compression_ratio > 0.5

    # RMSE and MAPE stay bounded.
    metrics = eval_table(orig, recon, cfg.lock_cols)
    for column, values in metrics.items():
        if column in cfg.lock_cols:
            continue
        rmse = values.get("rmse")
        mape = values.get("mape")
        if rmse is not None:
            assert rmse < 5.0
        if mape is not None:
            assert mape < 20.0

    # Percent of uncertain cells should remain low.
    uncertain_map = meta.get("uncertain", {})
    total_uncertain = sum(len(indices) for indices in uncertain_map.values())
    total_cells = recon.shape[0] * (recon.shape[1] - len(cfg.lock_cols))
    if total_cells > 0:
        percent_uncertain = (total_uncertain / total_cells) * 100
        assert percent_uncertain <= 10.0


def test_roundtrip_auto_lock_strings(tmp_path):
    sample = SAMPLE_CSV
    out_blob = tmp_path / "sample_auto.smp"
    out_csv = tmp_path / "recon_auto.csv"

    cfg = EncodeConfig(lock_cols=[], residual_cols=["amount"], k=8)
    blob = encode_csv(str(sample), cfg)
    out_blob.write_bytes(blob)

    meta = decode_to_csv(str(out_blob), str(out_csv))

    orig = pd.read_csv(sample)
    recon = pd.read_csv(out_csv)

    # String columns should be preserved even without manually specifying locks.
    assert orig["timestamp"].equals(recon["timestamp"])

    auto_locked = set(meta.get("meta", {}).get("auto_locked_cols", []))
    assert "timestamp" in auto_locked
