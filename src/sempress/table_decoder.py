from __future__ import annotations
from pathlib import Path
from typing import Tuple, Union
import numpy as np
import pandas as pd
from .container import unpack_container
from .utils import from_bytes

BlobSource = Union[str, Path, bytes, bytearray]

def _bytes_to_float32_array(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)

def _bytes_to_uint16_array(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.uint16)

def _load_blob(source: BlobSource) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    path = Path(source)
    with path.open("rb") as handle:
        return handle.read()

def _decode_payload(payload: dict) -> Tuple[pd.DataFrame, dict]:
    assert payload.get("domain") == "table"
    n = payload["n_rows"]
    cols = payload["columns"]
    lock_cols = payload["locked_cols"]

    locked_df = from_bytes(payload["locked_blob"]) if lock_cols else pd.DataFrame()

    recon = {}
    for c in cols:
        if c in lock_cols:
            recon[c] = locked_df[c]
        else:
            cb = _bytes_to_float32_array(payload["codebooks"][c]) if c in payload["codebooks"] else None
            if cb is None:
                recon[c] = pd.Series([None] * n)
                continue
            idx = _bytes_to_uint16_array(payload["idx_streams"][c])
            vals = cb[idx]
            if c in payload["residuals"]:
                delta = np.frombuffer(payload["residuals"][c], dtype=np.float32)
                vals = (vals.astype(np.float32) + delta).astype(np.float32)
            recon[c] = pd.Series(vals)

    df = pd.DataFrame(recon)[cols]
    meta = {
        "uncertain": payload.get("uncertain", {}),
        "meta": payload.get("meta", {}),
    }
    return df, meta

def decode_to_dataframe(source: BlobSource):
    blob = _load_blob(source)
    payload = unpack_container(blob)
    return _decode_payload(payload)

def decode_to_csv(blob_source: BlobSource, out_path: str):
    df, meta = decode_to_dataframe(blob_source)
    df.to_csv(out_path, index=False)
    return meta
