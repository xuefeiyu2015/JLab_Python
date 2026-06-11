# Python Package:  jlab

Python package for loading and exporting Blackrock NEV/NSx data from JLab experiments.
Replicates the output of `BackRockFileLoader.m`[Matlab Version](https://github.com/xuefeiyu2015/JLab) — produces identical `.txt` and `.csv` files.

Last update: Xuefei Yu, 06-09-2026


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

## Quick start

```python
from pathlib import Path
from jlab import BlackRockLoader

loader = BlackRockLoader(
    nev_path="path/to/file.nev",
    output_dir="path/to/output/",   # optional — defaults to same folder as .nev
    ns_path="path/to/file.ns2",     # optional — omit if no eye tracking
)

output_dir, files = loader.run()

print(f"Output folder: {output_dir}")
for f in files:
    print(f"  {f}")
```

Output:
```
Loading NEV: file.nev
  210 events found
  68 trials parsed (68 complete)
Loading analog: file.ns2
  3 channels, 1 segment(s), 279280 samples, 1000.0 Hz

Output folder: path/to/output
  Blackrock_2026-03-24_expmeta.txt
  Blackrock_2026-03-24_trials.csv
  Blackrock_2026-03-24_analog.csv
```

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
from jlab import BlackRockLoader

loader = BlackRockLoader("file.nev", output_dir="output/", ns_path="file.ns2")

# Step 1 — load and parse the NEV file
loader.load_nev()

# Step 2 — export .txt and .csv
loader.export()

# Step 3 — load and export analog (optional)
loader.load_analog("file.ns2")
loader.export_analog()

# Access parsed data directly
loader.experiment   # dict of experiment metadata
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
