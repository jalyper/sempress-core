from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from .utils import infer_schema, split_columns, numeric_columns, to_bytes, quant_error
from .container import pack_container

@dataclass
class EncodeConfig:
    lock_cols: List[str]
    residual_cols: List[str]
    k: int = 64
    uncertainty_thresh: float = 0.2
    random_state: int = 42

def _fit_codebook(col: np.ndarray, k: int, rs: int) -> np.ndarray:
    x = col.astype(float).reshape(-1, 1)
    k = min(k, max(2, len(np.unique(x))))
    model = KMeans(n_clusters=k, n_init=5, random_state=rs)
    model.fit(x)
    return model.cluster_centers_.reshape(-1)

def _encode_column(col: np.ndarray, codebook: np.ndarray) -> dict:
    x = col.astype(float)
    # nearest centroid
    idx = np.argmin(np.abs(x[:, None] - codebook[None, :]), axis=1).astype(np.uint16)
    recon = codebook[idx]
    qerr = quant_error(x, recon)
    return {
        "indices": idx.astype(np.uint16).tobytes(),
        "recon": recon,  # only for uncertainty; not stored in payload
        "qerr": qerr,
    }

def encode_csv(path: str, cfg: EncodeConfig) -> bytes:
    df = pd.read_csv(path)
    schema = infer_schema(df)
    # Always preserve string/categorical columns losslessly; they cannot be reconstructed
    # from numeric codebooks. Keep any user supplied locked columns first to preserve intent.
    manual_locks = [c for c in cfg.lock_cols if c in df.columns]
    auto_locks = [c for c, meta in schema.items() if meta.get("dtype") != "numeric" and c not in manual_locks]
    locked_cols = manual_locks + auto_locks

    locked_df, rest = split_columns(df, locked_cols)

    num_cols = [c for c in numeric_columns(rest)]
    codebooks: Dict[str, np.ndarray] = {}
    idx_streams: Dict[str, bytes] = {}
    uncertain_cells: Dict[str, List[int]] = {}
    residuals: Dict[str, bytes] = {}

    for c in num_cols:
        codebook = _fit_codebook(rest[c].to_numpy(), cfg.k, cfg.random_state)
        codebooks[c] = codebook
        encoded = _encode_column(rest[c].to_numpy(), codebook)
        idx_streams[c] = encoded["indices"]
        # Uncertainty mask (indices where relative error exceeds threshold)
        uncertain = np.where(encoded["qerr"] > cfg.uncertainty_thresh)[0].astype(np.int32)
        if len(uncertain) > 0:
            uncertain_cells[c] = uncertain.tolist()
        # Optional residuals for protected columns
        if c in cfg.residual_cols:
            delta = (rest[c].to_numpy().astype(np.float32) - encoded["recon"].astype(np.float32))
            residuals[c] = delta.tobytes()  # small; compressed by container

    payload = {
        "domain": "table",
        "schema": schema,
        "n_rows": len(df),
        "columns": list(df.columns),
        "locked_cols": locked_cols,
        "residual_cols": cfg.residual_cols,
        "locked_blob": to_bytes(locked_df) if len(locked_cols) else b"",
        "codebooks": {c: cb.astype(np.float32).tobytes() for c, cb in codebooks.items()},
        "k": cfg.k,
        "idx_streams": idx_streams,  # per-col raw bytes (uint16)
        "uncertain": uncertain_cells,
        "residuals": residuals,
        "model": {"type": "kmeans-per-column", "random_state": cfg.random_state},
        "meta": {
            "uncertainty_thresh": cfg.uncertainty_thresh,
            "auto_locked_cols": auto_locks,
            "manual_locked_cols": manual_locks,
        },
    }

    return pack_container(payload)
