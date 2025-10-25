from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def mape(y, yhat):
    y = np.asarray(y); yhat = np.asarray(yhat)
    denom = np.maximum(np.abs(y), 1e-8)
    return float(np.mean(np.abs(y - yhat) / denom))

def rmse(y, yhat):
    y = np.asarray(y); yhat = np.asarray(yhat)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))

def ks_distance(y, yhat):
    y = np.asarray(y).ravel(); yhat = np.asarray(yhat).ravel()
    try:
        return float(ks_2samp(y, yhat).statistic)
    except Exception:
        return float("nan")

def eval_table(original: pd.DataFrame, recon: pd.DataFrame, lock_cols: list[str]):
    metrics = {}
    common = [c for c in original.columns if c in recon.columns]
    for c in common:
        if c in lock_cols:
            metrics[c] = {"locked_exact_match": bool(original[c].equals(recon[c]))}
        elif np.issubdtype(original[c].dtype, np.number):
            metrics[c] = {
                "mape": mape(original[c], recon[c]),
                "rmse": rmse(original[c], recon[c]),
            }
    return metrics
