# Python Package:  jlab

Python package for loading and exporting Blackrock NEV/NSx data from JLab experiments.
Replicates the output of `BackRockFileLoader.m` / `BlackrockLoader.m` [Matlab Version](https://github.com/xuefeiyu2015/JLab) — same parsing schema, segmentation, and output products.

Last update: Xuefei Yu, 06-28-2026


## What it does

A session is split across files by **filename prefix**, and the loader picks each
file by its role (mirroring the MATLAB `BlackrockLoader`):

- `NSP-*.nev` → behavioral trial event comments, parsed into a trial-based structure
- `NSP-*.ns2` → analog (eye) data, segmented per trial
- `HUB-*.nev` → online spike timing, rasterized per trial (also the **legacy fallback** for comments)

Outputs (written to `export_data/<Date>/`):

- `Blackrock_YYYY-MM-DD_expmeta.txt` — experiment metadata (one key: value per line)
- `Blackrock_YYYY-MM-DD_trials.csv` — per-trial behavioral data with derived features
- `Blackrock_YYYY-MM-DD_analog.mat` — analog segmented into trials (`load_analog=True`)
- `Blackrock_YYYY-MM-DD_spikes.mat` — online spikes rasterized into trials (`load_spikes=True`)

The two segmented `.mat` files hold a struct with `data` (channels/units × trials ×
samples/bins, NaN-padded), `timeseq` (`alignedrawtime`, `aligned_marker`,
`relative_time`), and `info` (`samplingrate`, `Session`, `Trial_number`, plus
`Channel_Number`/`Unit_No` for spikes) — byte-for-byte aligned with the MATLAB output.

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
        │       ├── NSP-*.nev       ← behavioral events (largest NSP-* is auto-picked)
        │       ├── NSP-*.ns2       ← optional analog / eye tracking
        │       └── HUB-*.nev       ← optional online spikes (legacy comment fallback)
        └── export_data/            ← created automatically
            └── <Date>/
                ├── Blackrock_<Date>_expmeta.txt
                ├── Blackrock_<Date>_trials.csv
                ├── Blackrock_<Date>_analog.mat   ← if load_analog=True
                └── Blackrock_<Date>_spikes.mat   ← if load_spikes=True
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
| `load_analog` | `False` | load + segment the `NSP-*.ns2` analog file |
| `load_spikes` | `False` | load + rasterize online spikes from the `HUB-*.nev` file |
| `ns_marker` | `"ns2"` | NSx analog extension (use `"ns6"` etc. for other rates) |
| `nev_filename` | auto | pick a specific comment `.nev` instead of the `NSP-*` auto-pick |
| `ns_filename` | auto | pick a specific NSx file instead of the `NSP-*` auto-pick |
| `spike_nev_filename` | auto | pick a specific spike `.nev` instead of the `HUB-*` auto-pick |
| `pre_ms`, `post_ms` | `500` | trial-segmentation buffers (ms): window = `[Start - pre, End + post]` |
| `bin_ms` | `1` | spike-raster bin width (ms) |

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
    load_analog=True,        # segment the NSP-*.ns2 analog (eye-tracking) file
    load_spikes=True,        # rasterize online spikes from the HUB-*.nev file
    # nev_filename="...",    # optional: pick a specific comment .nev (else largest NSP-*)
    # ns_marker="ns2",       # NSx extension to load; default "ns2", e.g. "ns6"
    # ns_filename="...",     # optional: pick a specific NSx file
    # pre_ms=500, post_ms=500, bin_ms=1,   # segmentation buffers / raster bin width
)

output_dir, files = loader.run()

print(f"Output folder: {output_dir}")
for f in files:
    print(f"  {f}")
```

Output:
```
Comment NEV   : NSP-Porthos_20260617.nev
Analog file   : NSP-Porthos_20260617.ns2
Spike NEV     : HUB-Porthos_20260617.nev
Output dir    : /path/to/your/Data/Monkey Porthos/in_lab/export_data/2026-06-17
Loading NEV: NSP-Porthos_20260617.nev  (210 events)
  68 trials parsed (68 complete)
Loading analog: NSP-Porthos_20260617.ns2
  3 channels, 279280 samples, 1000.0 Hz
  Analog segmented (68 trials) -> Blackrock_2026-06-17_analog.mat
Loading spikes: HUB-Porthos_20260617.nev
  15234 spikes loaded
  Spikes rasterized (24 units x 68 trials) -> Blackrock_2026-06-17_spikes.mat

Output folder: /path/to/your/Data/Monkey Porthos/in_lab/export_data/2026-06-17
  Blackrock_2026-06-17_expmeta.txt
  Blackrock_2026-06-17_trials.csv
  Blackrock_2026-06-17_analog.mat
  Blackrock_2026-06-17_spikes.mat
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

# Every date folder under raw_data/, also exporting analog + spikes, skipping already-done dates
report = BlackRockLoader.run_batch(
    Session_Path, load_analog=True, load_spikes=True, skip_existing=True
)

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
| `load_analog` | `False` | Also segment + export the `NSP-*.ns2` analog file for each date |
| `load_spikes` | `False` | Also rasterize + export online spikes from the `HUB-*.nev` file |
| `ns_marker` | `"ns2"` | NSx extension to load when `load_analog=True` |
| `pre_ms`, `post_ms` | `500` | Trial-segmentation buffers (ms) |
| `bin_ms` | `1` | Spike-raster bin width (ms) |
| `data_type` | `"raw_data"` | Raw-data sub-folder name |
| `output_folder` | `"export_data"` | Export sub-folder name |
| `verbose` | `True` | Print per-date progress |

> **Note:** batch mode auto-detects the **largest** matching file per role prefix in
> each date folder (`NSP-*` for comments/analog, `HUB-*` for spikes) — there's no
> per-date `nev_filename`/`ns_filename`. If a folder has several recordings and you need
> a specific one, load that date with a single `BlackRockLoader` instead. The same
> applies to per-session inspection like `print_summary()`.

## Notebooks

Open JupyterLab and run the notebooks in `notebooks/`:

```bash
jupyter lab
```

| Notebook | Description |
|---|---|
| `01_load_and_export.ipynb` | Load NSP/HUB files and export `.txt` / `.csv` / segmented analog + spike `.mat` |
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

### `_analog.mat`
Analog stream segmented into trials. A MATLAB struct `analog` with:

| Field | Shape | Description |
|---|---|---|
| `data` | channels × trials × samples | continuous analog (µV), NaN-padded to the longest trial |
| `timeseq.alignedrawtime` | trials | absolute time (s) of each trial's `Start` marker |
| `timeseq.aligned_marker` | — | `"Start"` (event that `relative_time` = 0 aligns to) |
| `timeseq.relative_time` | samples | seconds from the `Start` marker (negative through the pre-buffer) |
| `info.samplingrate` | — | analog sampling rate (Hz) |
| `info.Session`, `info.Trial_number` | trials | per-trial session index / trial number |

Load it in Python with `scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)`.

### `_spikes.mat`
Online spikes rasterized into per-trial bins. A MATLAB struct `online_spike` with the
same `timeseq` layout as above, plus:

| Field | Shape | Description |
|---|---|---|
| `data` | units × trials × bins | binary raster (0/1), NaN-padded; one row per `(channel, unit)` pair |
| `info.samplingrate` | — | bin rate (Hz), 1000 for 1 ms bins |
| `info.Channel_Number`, `info.Unit_No` | units | electrode / unit id per raster row |

## Step-by-step API (if you want to do some basic analysis directly after exporting)

```python
from pathlib import Path
from jlab import BlackRockLoader

Session_Path = Path("/path/to/your/Data") / "Monkey Porthos" / "in_lab"
loader = BlackRockLoader(Session_Path, "2026-06-17", load_analog=True, load_spikes=True)

# Step 1 — load and parse the comment NEV file
loader.load_nev()

# Step 2 — export .txt and .csv
loader.export()

# Step 3 — segment + export analog (optional; needs load_analog=True)
loader.load_analog()
loader.export_analog()

# Step 4 — rasterize + export online spikes (optional; needs load_spikes=True)
loader.load_spikes()
loader.export_spikes()

# Access parsed data directly
loader.experiments   # list of experiment-metadata dicts (one per session)
loader.trials        # list of per-trial dicts
loader.analog        # raw brpylib dict ('data', 'elec_ids', 'samp_per_s', ...)
loader.spike_times   # spike timestamps (s); also spike_channel / spike_unit
```

## Project structure

```
JLab_Python/
├── jlab/
│   ├── loader.py        # BlackRockLoader — main user-facing class
│   ├── _parser.py       # event comment parsing
│   ├── _features.py     # derived feature computation (angles, eccentricity)
│   ├── _exporter.py     # writes .txt and .csv files
│   ├── _analog.py       # segment_analog — analog stream → per-trial 3D array
│   ├── _spikes.py       # segment_spikes — online spikes → per-trial raster
│   └── _constants.py    # file schema, event maps, regex patterns
├── brpylib/             # bundled Blackrock file reader
├── notebooks/
│   ├── 01_load_and_export.ipynb
│   └── 02_psychometric_analysis.ipynb
└── pyproject.toml
```
