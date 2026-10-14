"""FRED constant-maturity Treasury yields. Sealed behind Gate E.

Every route to historical data goes through `load_monthly_yields`, which refuses to download
or read anything until gates A-D have a recorded PASS. Used for three things only: calibration
diagnostics, evaluation of policies trained on simulated curves, and legibility in the writeup.
Never for training.
"""

from __future__ import annotations

import io
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from duralloc import config as _config
from duralloc.gates import require_gates

URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"


def tenor_of(series_id: str) -> float:
    m = re.fullmatch(r"DGS(\d+)(MO)?", series_id)
    if not m:
        raise ValueError(f"not a constant-maturity series: {series_id}")
    return int(m.group(1)) / (12 if m.group(2) else 1)


def _download(series_id: str, cache: Path) -> Path:
    path = cache / f"{series_id}.csv"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(URL.format(series_id), headers={"User-Agent": "duralloc/0.1"})
        with urllib.request.urlopen(req, timeout=60) as r:
            path.write_bytes(r.read())
    return path


def _read(path: Path) -> pd.Series:
    df = pd.read_csv(io.StringIO(path.read_text()))
    date_col = df.columns[0]
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce")  # FRED marks gaps with "." or blanks
    s.index = pd.to_datetime(df[date_col])
    return s / 100.0


def load_monthly_yields(cfg, cache: Path | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Month-end yields (decimals), columns ordered by tenor. Gate E: A-D must have passed."""
    require_gates("A", "B", "C", "D", stage="historical data loading (Gate E)")
    cache = cache or _config.DATA_CACHE
    cols = {}
    for sid in cfg.historical.series:
        cols[sid] = _read(_download(sid, cache))
    daily = pd.DataFrame(cols)
    monthly = daily.resample("ME").last()  # last available observation in each month
    monthly = monthly[monthly.index >= pd.Timestamp(cfg.historical.start)]
    monthly = monthly.dropna(how="all")
    tenors = np.array([tenor_of(c) for c in monthly.columns])
    order = np.argsort(tenors)
    return monthly.iloc[:, order], tenors[order]
