# Python Package:  jlab

Python package for loading and exporting Blackrock NEV/NSx data from JLab experiments.
Replicates the output of `BackRockFileLoader.m`[Matlab Version](https://github.com/xuefeiyu2015/JLab) — produces identical `.txt` and `.csv` files.

Last update: Xuefei Yu, 06-19-2026


## What it does

- Reads a Blackrock `.nev` file file and parses behavioral trial event comments into trial based structure
- Reads a Blackrock `.ns2` file file and parses eye data into trial based structure (under construction)
- Exports `Blackrock_YYYY-MM-DD_expmeta.txt` — experiment metadata (one key: value per line)
- Exports `Blackrock_YYYY-MM-DD_trials.csv` — per-trial behavioral data with derived features
- Optionally loads a `.ns2` analog file (eye tracking) and exports it to `Blackrock_YYYY-MM-DD_analog.csv`(under construction)

## Requirements

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Installation

**1. Install uv** (if you don't have it)

macOS / Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or with pip if you already have Python:
```bash
pip install uv
```

Verify the install:
```bash
uv --version
```

**2. Clone the repository**

```bash
git clone https://github.com/xuefeiyu2015/JLab_Python.git
cd JLab_Python
```

**3. Create a virtual environment and install dependencies**

```bash
uv venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

uv pip install -e .
```

That's it. `brpylib` is bundled in this repo — no separate install needed.

## Setting up your data path

Loading is **folder-based** — you don't pass individual filenames. The loader
mirrors the MATLAB workflow: point it at a *session folder* and a *date*, and it
auto-detects the raw files and writes the exports.

Your data must be laid out like this:

```
<Basic_Path>/
└── <Monkey>/
    └── <Location>/                 ← this is the "session path"
        ├── raw_data/
        │   └── <Date>/             ← e.g. 2026-06-17
        │       ├── *.nev           ← behavioral events (largest is auto-picked)
        │       └── *.ns2           ← optional analog / eye tracking
        └── export_data/            ← created automatically
            └── <Date>/
                ├── Blackrock_<Date>_expmeta.txt
                ├── Blackrock_<Date>_trials.csv
                └── Blackrock_<Date>_analog.csv
```

Build the **session path** from your own folder structure, then pass it with the
date:

```python
from pathlib import Path
from jlab import BlackRockLoader

# ── Edit these for your machine ─────────────────────────────────────────────
Basic_Path = "/path/to/your/Data"   # root data folder
Monkey     = "Monkey Porthos"
Location   = "in_lab"
Date       = "2026-06-17"           # the raw_data/<Date> sub-folder
# ────────────────────────────────────────────────────────────────────────────

Session_Path = Path(Basic_Path) / Monkey / Location

# The loader reads from  Session_Path/raw_data/<Date>
# and writes to          Session_Path/export_data/<Date>
```

Defaults you can override in the constructor:

| Parameter | Default | Purpose |
|---|---|---|
| `data_type` | `"raw_data"` | name of the raw-data sub-folder |
| `output_folder` | `"export_data"` | name of the export sub-folder |
| `ns_marker` | `"ns2"` | NSx analog extension (use `"ns6"` etc. for other rates) |
| `nev_filename` | auto | pick a specific `.nev` instead of the largest |
| `ns_filename` | auto | pick a specific NSx file instead of the largest |

## Quick start

```python
from pathlib import Path
from jlab import BlackRockLoader

Session_Path = Path("/path/to/your/Data") / "Monkey Porthos" / "in_lab"
Date = "2026-06-17"

# Peek at the .nev files in the folder (largest first) — optional
BlackRockLoader.list_nev_files(Session_Path, Date)

loader = BlackRockLoader(
    Session_Path, Date,
    load_analog=True,        # also auto-detect the NSx analog (eye-tracking) file
    # nev_filename="...",    # optional: pick a specific .nev (else the largest)
    # ns_marker="ns2",       # NSx extension to load; default "ns2", e.g. "ns6"
    # ns_filename="...",     # optional: pick a specific NSx file
)

output_dir, files = loader.run()

print(f"Output folder: {output_dir}")
for f in files:
    print(f"  {f}")
```

Output:
```
NEV file      : Hub1-Porthos_20260617.nev
Analog file   : Hub1-Porthos_20260617.ns2
Output dir    : /path/to/your/Data/Monkey Porthos/in_lab/export_data/2026-06-17
Loading NEV: Hub1-Porthos_20260617.nev
  210 events found
  68 trials parsed (68 complete)
Loading analog: Hub1-Porthos_20260617.ns2
  3 channels, 1 segment(s), 279280 samples, 1000.0 Hz

Output folder: /path/to/your/Data/Monkey Porthos/in_lab/export_data/2026-06-17
  Blackrock_2026-06-17_expmeta.txt
  Blackrock_2026-06-17_trials.csv
  Blackrock_2026-06-17_analog.csv
```

## Batch loading (multiple dates)

To process more than one session at once, use the `run_batch` **classmethod**. Pass
a **list of dates**, or omit `dates` to auto-discover every `YYYY-MM-DD` folder under
`raw_data/`. It builds one single-date `BlackRockLoader` per date, runs it through the
same load + export pipeline, and returns a summary list.

```python
from pathlib import Path
from jlab import BlackRockLoader

Session_Path = Path("/path/to/your/Data") / "Monkey Porthos" / "in_lab"

# Specific dates
report = BlackRockLoader.run_batch(Session_Path, dates=["2026-06-17", "2026-06-18"])

# Every date folder under raw_data/, also exporting analog, skipping already-done dates
report = BlackRockLoader.run_batch(Session_Path, load_analog=True, skip_existing=True)

for r in report:
    print(r["date"], r["status"])           # "ok" | "skipped" | "failed"
    if r["status"] == "ok":
        print(" ", r["output_dir"], r["files"])
    elif r["status"] == "failed":
        print(" ", r["error"])
```

**Continue-on-error:** a date that fails (e.g. missing `.nev`, parse error) is
recorded and skipped so the batch always finishes — the failure shows up in the
report with an `"error"` message instead of raising.

```
Batch: 3 date folder(s)
[1/3] 2026-06-17  ... OK
[2/3] 2026-06-18  ... FAILED (No *.nev file found in: .../raw_data/2026-06-18)
[3/3] 2026-06-19  ... OK

Done: 2 ok, 0 skipped, 1 failed
```

**Report structure** — one dict per date:

| Key | Always present | Description |
|---|---|---|
| `date` | yes | The `YYYY-MM-DD` string |
| `status` | yes | `"ok"`, `"skipped"`, or `"failed"` |
| `output_dir` | on `"ok"` | `Path` to the export folder |
| `files` | on `"ok"` | List of exported filenames |
| `error` | on `"failed"` | Error message string |

**`run_batch` options:**

| Parameter | Default | Purpose |
|---|---|---|
| `dates` | `None` | List of dates, or `None`/empty to auto-discover every `YYYY-MM-DD` folder |
| `skip_existing` | `False` | Skip dates whose `_expmeta.txt` **and** `_trials.csv` already exist |
| `load_analog` | `False` | Also load + export the NSx analog file for each date |
| `ns_marker` | `"ns2"` | NSx extension to load when `load_analog=True` |
| `data_type` | `"raw_data"` | Raw-data sub-folder name |
| `output_folder` | `"export_data"` | Export sub-folder name |
| `verbose` | `True` | Print per-date progress |

> **Note:** batch mode always auto-detects the **largest** `.nev` (and NSx) file in
> each date folder — there's no per-date `nev_filename`/`ns_filename`. If a folder has
> several recordings and you need a specific one, load that date with a single
> `BlackRockLoader` instead. The same applies to per-session inspection like
> `print_summary()`.

## Notebooks

Open JupyterLab and run the notebooks in `notebooks/`:

```bash
jupyter lab
```

| Notebook | Description |
|---|---|
| `01_load_and_export.ipynb` | Load a `.nev` file and export `.txt` / `.csv` / analog |
| `02_psychometric_analysis.ipynb` | Load exported CSV, fit logistic function, plot psychometric curve |

Edit the path variables at the top of each notebook (marked with `# ── Edit these variables`).

## Exported files

### `_expmeta.txt`
One key: value per line. Example:
```
git_commit: abc123
viewing_distance: 57
screen_size: [53 30]
screen_resolution: [1920 1080]
FPS: 60
eyetracker_rate: 1000
eye_tracked: right
start: 1772321423.977
end: 1772321687.412
```

### `_trials.csv`
One row per trial. Key columns:

| Column | Description |
|---|---|
| `Trial_number` | Trial index |
| `Task` | Task name (e.g. `time_delay_experiment`) |
| `Trialoutcome` | Behavioral outcome |
| `Save_complete` | 1 if trial data is complete |
| `Target_1_angle`, `Target_2_angle` | Target polar angle (degrees, 0° = up, clockwise) |
| `Target_1_ecc`, `Target_2_ecc` | Target eccentricity (degrees) |
| `Stimulus_direction` | +1 or −1 |
| `Choose_target` | Which target was chosen (1 or 2) |
| `Choose_leftright` | 1 = rightward, −1 = leftward |

### `_analog.csv` (under construction)
One row per sample, columns named `channel_1`, `channel_2`, etc. (electrode IDs from the NSx file).

## Step-by-step API (if you want to do some basic analysis directly after exporting)

```python
from pathlib import Path
from jlab import BlackRockLoader

Session_Path = Path("/path/to/your/Data") / "Monkey Porthos" / "in_lab"
loader = BlackRockLoader(Session_Path, "2026-06-17", load_analog=True)

# Step 1 — load and parse the NEV file
loader.load_nev()

# Step 2 — export .txt and .csv
loader.export()

# Step 3 — load and export analog (optional)
# self.ns_path was auto-detected because load_analog=True was passed
loader.load_analog(loader.ns_path)
loader.export_analog()

# Access parsed data directly
loader.experiments  # list of experiment-metadata dicts (one per session)
loader.trials       # list of per-trial dicts
loader.analog       # dict with 'data', 'elec_ids', 'samp_per_s'
```

## Project structure

```
JLab_Python/
├── jlab/
│   ├── loader.py        # BlackRockLoader — main user-facing class
│   ├── _parser.py       # event comment parsing
│   ├── _features.py     # derived feature computation (angles, eccentricity)
│   ├── _exporter.py     # writes .txt and .csv files
│   └── _constants.py    # event maps and regex patterns
├── brpylib/             # bundled Blackrock file reader
├── notebooks/
│   ├── 01_load_and_export.ipynb
│   └── 02_psychometric_analysis.ipynb
└── pyproject.toml
```
