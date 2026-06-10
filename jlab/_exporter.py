"""
File writers for the two output formats 
  - Blackrock_YYYY-MM-DD_expmeta.txt  (key: value per line)
  - Blackrock_YYYY-MM-DD_trials.csv   (flattened trial data)
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from ._constants import COORD_FIELDS
from ._features import DERIVED_FIELDS
from ._parser import _make_empty_trial


def _build_column_order() -> list[str]:
    """Derive CSV column order from the trial template + derived feature names."""
    # Explicit 0-based sequential row index (pandas-friendly: read_csv(index_col='index')).
    # Kept separate from Trial_number, which holds the real (resetting) trial number.
    columns: list[str] = ["index"]
    for field in _make_empty_trial():
        if field in ("undefined", "duplicates"):
            continue
        if field in COORD_FIELDS:
            columns += [field + "_x", field + "_y"]
        else:
            columns.append(field)
    columns += DERIVED_FIELDS
    return columns


_COLUMN_ORDER = _build_column_order()


# ── Helpers ────────────────────────────────────────────────────────────────

def _mat2str(val) -> str:
    """
    Reproduce MATLAB mat2str() output for scalar/array/string values.
      NaN / None      → 'NaN'
      [a, b] list     → '[a b]'   (space-separated, no commas)
      whole float     → integer string  (30000.0 → '30000')
      string          → as-is
    """
    if val is None:
        return "NaN"
    if isinstance(val, float) and math.isnan(val):
        return "NaN"
    if isinstance(val, (list, tuple)):
        parts = []
        for v in val:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                parts.append("NaN")
            elif isinstance(v, float) and v.is_integer():
                parts.append(str(int(v)))
            else:
                parts.append(str(v))
        return "[" + " ".join(parts) + "]"
    if isinstance(val, float) and val.is_integer():
        return str(int(val))
    return str(val)


def _csv_value(val) -> str:
    """Serialise a single trial field value for CSV output. NaN → empty cell."""
    if val is None:
        return ""
    if isinstance(val, float) and math.isnan(val):
        return ""
    if isinstance(val, float) and val.is_integer():
        return str(int(val))
    if isinstance(val, float):
        return f"{val:.15g}"  # match MATLAB writetable precision (15 sig figs)
    return str(val)


# ── Public writers ─────────────────────────────────────────────────────────

def write_expmeta(experiment: dict, path: Path) -> None:
    """
    Write experiment metadata to a .txt file.
    Format mirrors MATLAB:  fprintf(fid, '%s: %s\\n', field, mat2str(val))
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fid:
        for field, val in experiment.items():
            fid.write(f"{field}: {_mat2str(val)}\n")


def write_trials_csv(trials: list[dict], path: Path) -> None:
    """
    Write trial data to a .csv file.
    Mirrors MATLAB:
      - removes 'undefined' and 'duplicates' columns (rmfield)
      - expands 2D coord fields into _x / _y columns
      - NaN / None → empty cell
    """
    if not trials:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    # Flatten each trial: expand coord fields, drop internal lists
    rows = []
    for i, trial in enumerate(trials):
        row: dict[str, str] = {"index": str(i)}
        for field, val in trial.items():
            if field in ("undefined", "duplicates"):
                continue
            if field in COORD_FIELDS:
                xy = val if isinstance(val, (list, tuple)) else [math.nan, math.nan]
                row[field + "_x"] = _csv_value(xy[0])
                row[field + "_y"] = _csv_value(xy[1])
            else:
                row[field] = _csv_value(val)
        rows.append(row)

    # Build ordered column list: template order first, then any extras
    known = set(_COLUMN_ORDER)
    extra = [c for c in rows[0].keys() if c not in known]
    columns = [c for c in _COLUMN_ORDER if c in rows[0]] + extra

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
