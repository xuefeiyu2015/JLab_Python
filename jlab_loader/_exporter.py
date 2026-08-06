"""
File writers for the two text output formats
  - Blackrock_YYYY-MM-DD_expmeta.txt  (key: value per line)
  - Blackrock_YYYY-MM-DD_trials.csv   (flattened trial data)

Both reproduce what MATLAB's prepareExport + export produce, byte for byte.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from ._constants import COORD_FIELDS
from ._features import DERIVED_FIELDS
from ._parser import _make_empty_trial


def _build_column_order() -> list[str]:
    """Derive CSV column order from the trial template + derived feature names.

    MATLAB builds this table in three steps (BlackrockLoader.m:1300-1318):
      1. struct2table over the template order, minus undefined/duplicates,
         with the 7 derived fields already appended by addDerivedTrialFeatures;
      2. an explicit 0-based `index` column inserted at the front;
      3. every 2-column numeric field split into <name>_x / <name>_y.

    Step 3 uses dot-assignment, which APPENDS the new variables to the end of
    the table and then deletes the original — so the six coordinate columns land
    after the derived features, not in the template position of the field they
    came from. That ordering is reproduced here.
    """
    # Explicit 0-based sequential row index (pandas-friendly: read_csv(index_col='index')).
    # Kept separate from Trial_number, which holds the real (resetting) trial number.
    columns: list[str] = ["index"]
    coord_seen: list[str] = []
    for field in _make_empty_trial():
        if field in ("undefined", "duplicates"):
            continue
        if field in COORD_FIELDS:
            coord_seen.append(field)  # deferred to the end, as MATLAB does
        else:
            columns.append(field)
    columns += DERIVED_FIELDS
    for field in coord_seen:
        columns += [field + "_x", field + "_y"]
    return columns


_COLUMN_ORDER = _build_column_order()


# ── Helpers ────────────────────────────────────────────────────────────────

def _fmt_float(v: float) -> str:
    """MATLAB's default 15-significant-digit float formatting."""
    if v.is_integer():
        return str(int(v))
    return f"{v:.15g}"


def _mat2str(val) -> str:
    """
    Reproduce MATLAB mat2str() output for scalar/array/string values.
      NaN / None      → 'NaN'
      [a, b] list     → '[a b]'   (space-separated, no commas)
      whole float     → integer string  (30000.0 → '30000')
      other float     → 15 significant digits (mat2str's default precision)
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
            elif isinstance(v, float):
                parts.append(_fmt_float(v))
            else:
                parts.append(str(v))
        return "[" + " ".join(parts) + "]"
    if isinstance(val, float):
        return _fmt_float(val)
    return str(val)


def _csv_value(val, text_column: bool) -> str:
    """Serialise one trial field for CSV output.

    MATLAB's writetable prints a numeric NaN as the literal text "NaN". Text
    columns are different: prepareExport converts any cell column holding text
    to a string array with `missing` for the NaN placeholders, and writetable
    prints missing as an empty field. So the encoding of "no value" depends on
    whether the column carries text anywhere in the session.
    """
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "" if text_column else "NaN"
    if isinstance(val, float):
        return _fmt_float(val)
    return str(val)


def _text_columns(rows: list[dict]) -> set[str]:
    """Columns holding a string in at least one trial.

    Mirrors MATLAB's `is_text = cellfun(@(x) ischar(x) || isstring(x), v)` test:
    a column becomes a string array (empty for missing) only if some trial
    actually wrote text into it; otherwise it stays double and prints NaN.
    """
    text: set[str] = set()
    for row in rows:
        for k, v in row.items():
            if isinstance(v, str):
                text.add(k)
    return text


# ── Public writers ─────────────────────────────────────────────────────────

def write_expmeta(experiments: list[dict], path: Path) -> None:
    """
    Write per-session experiment metadata to a .txt file.

    One section per session, separated by a "Session N:" header and a blank line.
    Format mirrors MATLAB (BlackrockLoader.m:1285-1298):
        fprintf(fid, 'Session %d:\\n', s)
        fprintf(fid, '%s: %s\\n', field, mat2str(val))
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fid:
        for s, experiment in enumerate(experiments, start=1):
            fid.write(f"Session {s}:\n")
            for field, val in experiment.items():
                fid.write(f"{field}: {_mat2str(val)}\n")
            fid.write("\n")


def write_trials_csv(trials: list[dict], path: Path) -> None:
    """
    Write trial data to a .csv file.
    Mirrors MATLAB:
      - removes 'undefined' and 'duplicates' columns (rmfield)
      - prepends a 0-based 'index' column
      - expands 2D coord fields into _x / _y columns, appended at the end
      - numeric missing → literal 'NaN'; text missing → empty cell
    """
    if not trials:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    # Flatten each trial: expand coord fields, drop internal lists. Values are
    # kept raw here so _text_columns can tell text from numeric placeholders.
    raw_rows: list[dict] = []
    for i, trial in enumerate(trials):
        row: dict = {"index": i}
        for field, val in trial.items():
            if field in ("undefined", "duplicates"):
                continue
            if field in COORD_FIELDS:
                xy = val if isinstance(val, (list, tuple)) else [math.nan, math.nan]
                row[field + "_x"] = xy[0]
                row[field + "_y"] = xy[1]
            else:
                row[field] = val
        raw_rows.append(row)

    text_cols = _text_columns(raw_rows)

    # Build ordered column list: template order first, then any extras
    known = set(_COLUMN_ORDER)
    extra = [c for c in raw_rows[0] if c not in known]
    columns = [c for c in _COLUMN_ORDER if c in raw_rows[0]] + extra

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in raw_rows:
            writer.writerow(
                [_csv_value(row.get(c), c in text_cols) for c in columns]
            )
