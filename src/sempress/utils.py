from __future__ import annotations
import pandas as pd
import numpy as np

NUMERIC_KINDS = set(list("ifu"))  # int/float/unsigned

def infer_schema(df: pd.DataFrame) -> dict:
    schema = {}
    for col in df.columns:
        dt = df[col].dtype
        if dt.kind in NUMERIC_KINDS:
            schema[col] = {"dtype": "numeric"}
        else:
            schema[col] = {"dtype": "string"}
    return schema

def split_columns(df: pd.DataFrame, lock_cols: list[str]):
    lock_cols = [c for c in lock_cols if c in df.columns]
    locked = df[lock_cols].copy()
    rest = df.drop(columns=lock_cols).copy()
    return locked, rest

def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if df[c].dtype.kind in NUMERIC_KINDS]

def to_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def from_bytes(b: bytes) -> pd.DataFrame:
    from io import BytesIO
    return pd.read_csv(BytesIO(b))

def quant_error(x: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
    err = np.abs(x - x_hat)
    denom = np.maximum(np.abs(x), 1e-8)
    return err / denom
