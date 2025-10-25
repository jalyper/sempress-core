from __future__ import annotations
import json
import click
from .table_encoder import encode_csv, EncodeConfig
from .table_decoder import decode_to_csv
from .metrics import eval_table
import pandas as pd

@click.group()
def main():
    """Sempress: semantic compression MVP for tables"""
    pass

@main.command()
@click.option("--in", "in_path", required=True, type=click.Path(exists=True))
@click.option("--out", "out_path", required=True, type=click.Path())
@click.option("--lock-cols", default="", help="Comma-separated columns to store losslessly")
@click.option("--residual-cols", default="", help="Comma-separated columns to store residuals for")
@click.option("--k", default=64, show_default=True, type=int, help="Codebook size per numeric column")
@click.option("--uncert-thresh", default=0.2, show_default=True, type=float, help="Relative error threshold for uncertainty mask")
@click.option("--seed", default=42, show_default=True, type=int)
def encode(in_path, out_path, lock_cols, residual_cols, k, uncert_thresh, seed):
    cfg = EncodeConfig(
        lock_cols=[c for c in lock_cols.split(",") if c.strip()],
        residual_cols=[c for c in residual_cols.split(",") if c.strip()],
        k=k,
        uncertainty_thresh=uncert_thresh,
        random_state=seed,
    )
    blob = encode_csv(in_path, cfg)
    with open(out_path, "wb") as f:
        f.write(blob)
    click.echo(f"Wrote {out_path}")

@main.command()
@click.option("--in", "in_path", required=True, type=click.Path(exists=True))
@click.option("--out", "out_path", required=True, type=click.Path())
def decode(in_path, out_path):
    meta = decode_to_csv(in_path, out_path)
    click.echo(f"Reconstructed → {out_path}")
    click.echo("Uncertainty summary:" )
    click.echo(json.dumps(meta.get("uncertain", {}), indent=2))

@main.command()
@click.option("--original", required=True, type=click.Path(exists=True))
@click.option("--recon", required=True, type=click.Path(exists=True))
@click.option("--lock-cols", default="", help="Comma-separated locked columns")
def eval(original, recon, lock_cols):
    df = pd.read_csv(original)
    dfh = pd.read_csv(recon)
    locks = [c for c in lock_cols.split(",") if c.strip()]
    metrics = eval_table(df, dfh, locks)
    click.echo(json.dumps(metrics, indent=2))
