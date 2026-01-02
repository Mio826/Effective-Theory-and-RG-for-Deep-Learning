# dlphys/core/scan_runs.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from dlphys.utils.io import load_json
from dlphys.utils.jsonl import read_jsonl, flatten_dict

PathLike = Union[str, Path]


def _safe_read_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return load_json(path)
    except Exception:
        return {}


def _safe_read_metrics(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return read_jsonl(path)
    except Exception:
        return []


def _last_record(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    # We prefer the last row that has "metrics" key (a training record).
    # Fallback: last row.
    if not records:
        return {}
    for r in reversed(records):
        if isinstance(r, dict) and ("metrics" in r):
            return r
    return records[-1] if isinstance(records[-1], dict) else {}


def scan_runs(
    runs_dir: PathLike = "runs",
    *,
    glob_pattern: str = "*",
    flatten_sep: str = ".",
):
    """
    Scan runs directory and return a pandas DataFrame where each row is one run.

    Expected run layout:
      runs/<run_name>/config.json
      runs/<run_name>/metrics.jsonl

    Output columns (typical):
      run_name, run_dir, mtime,
      (flattened config fields),
      last.step, last.epoch, last.t, last.metrics.loss, ...
      num_records
    """
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(runs_dir)

    rows: List[Dict[str, Any]] = []

    for run_dir in sorted([p for p in runs_dir.glob(glob_pattern) if p.is_dir()]):
        cfg_path = run_dir / "config.json"
        met_path = run_dir / "metrics.jsonl"

        cfg = _safe_read_config(cfg_path)
        records = _safe_read_metrics(met_path)
        last = _last_record(records)

        row: Dict[str, Any] = {
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "mtime": run_dir.stat().st_mtime,
            "num_records": len(records),
            "has_config": cfg_path.exists(),
            "has_metrics": met_path.exists(),
        }

        # flatten config into cfg.*
        if cfg:
            fcfg = flatten_dict(cfg, prefix="cfg", sep=flatten_sep)
            row.update(fcfg)

        # flatten last record into last.*
        if last:
            flast = flatten_dict(last, prefix="last", sep=flatten_sep)
            row.update(flast)

        rows.append(row)

    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("pandas is required for scan_runs. Please install pandas.") from e

    df = pd.DataFrame(rows)

    # Nice: sort by mtime (latest first) if present
    if "mtime" in df.columns and len(df) > 0:
        df = df.sort_values("mtime", ascending=False).reset_index(drop=True)

    return df
