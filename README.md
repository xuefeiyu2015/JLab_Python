# Python Package:  jlab_loader

Python package for loading and exporting Blackrock NEV/NSx data from JLab experiments.
Replicates the output of `BackRockFileLoader.m` / `BlackrockLoader.m` [Matlab Version](https://github.com/xuefeiyu2015/JLab) — same parsing schema, segmentation, and output products.

Last update: Xuefei Yu, 08-05-2026


## What it does

A session is split across files by **filename prefix**, and the loader picks each
file by its role (mirroring the MATLAB `BlackrockLoader`):

- `NSP-*.nev` → behavioral trial event comments, parsed into a trial-based structure
- `NSP-*.ns2` → eye data (channels 1-3) and the photodiode (channels 4-6), segmented per trial
- `Hub-*.ns2` → local field potential, segmented per trial
- `HUB-*.nev` → online spike timing, rasterized per trial (also the **legacy fallback** for comments)
- `HUB-*.nev` → online spike **waveforms**, segmented per trial (opt-in `load_online_wave=True`)

Outputs (written to `export_data/<Date>/`) — the same contents as the MATLAB loader, with
the same filenames minus the `_matlab` suffix:

| File | MATLAB variable | Gate |
|---|---|---|
| `Blackrock_<date>_expmeta.txt` | — | always |
| `Blackrock_<date>_trials.csv` | — | always |
| `Blackrock_<date>_eye.mat` | `eye` | `load_eye=True` |
| `Blackrock_<date>_photodiode.mat` | `photodiode` | `load_photodiode=True` |
| `Blackrock_<date>_lfp.mat` | `lfp` | `load_lfp=True` |
| `Blackrock_<date>_spikes.mat` | `online_spike` | `load_online_spikes=True` |
| `Blackrock_<date>_spikes_waveform.mat` | `online_spike_waveform` | `load_online_wave=True` |

The photodiode normally rides on channels 4-6 of the **same** `NSP-*.ns2` as the eye, so it
costs no extra read: that stream is segmented once and the result split by row. It falls
back to a dedicated `NSP-*.ns4` when the eye stream has fewer channels.

The eye/LFP/photodiode/spike `.mat` files hold a struct with `data` (channels or units ×
trials × samples or bins, NaN-padded), `timeseq` (`alignedrawtime`, `aligned_marker`,
`relative_time`), and `info` (`samplingrate`, `Session`, `Trial_number`, plus
`Channel_Number`, `Unit_No`, `ViolationRate`, `MeanWaveform` and `MeanWaveformUnit` for
spikes). The waveform `.mat` mirrors the MATLAB `segmentSpikeWaveforms` struct instead:
`waveform` (units × trials × spikes × samples, µV, NaN-padded — spike waveforms really
are µV, they use a different header factor), `waveform_time`,
`waveform_nsamp`, `waveform_unit`, plus `timeseq` and `info`. `load_online_wave` reads the
same `HUB-*.nev` as spikes, so it implies reading spikes; it is off by default because the
dense array is large.

**Precision** follows the MATLAB class's rule — voltages `single`, times always `double`.
Sample values, spike waveforms and the 0/1 raster are `float32`; every timestamp is
`float64` seconds on the recording clock.

### A note on units: not everything is µV

Each NSx channel is stored as raw ADC codes plus a scale factor from its extended header,
`MaxAnalogValue / MaxDigitalValue`. That factor produces the unit named in the header's
`Units` field — but NPMK's `openNSx` ignores that field and labels every scaled result µV
(`openNSx.m:1350`), and the MATLAB loader inherits the label. On a typical rig:

| Stream | header `Units` | factor | what the numbers really are |
|---|---|---|---|
| LFP (`Hub1-chan*`) | `uV` | 8191/32764 = 0.25 | genuinely **µV** |
| eye / photodiode (`ainp*`) | `mV` | 5000/32767 = 0.152592547 | **mV** — MATLAB calls them µV |

So an eye trace reading `-3118.8` is −3118.8 **mV**, not µV. This package keeps MATLAB's
numbers (that is the whole point of the port) but removes the ambiguity by recording the
real unit in `info.Unit` — `"milliVolts"` for eye/photodiode, `"microVolts"` for LFP. That
field is the one deliberate addition to MATLAB's struct schema.

If you would rather have everything in true µV, pass `to_microvolts=True`. The header unit is
then folded into the scale factor, so every product really is µV and `info.Unit` reads
`"microVolts"` throughout:

```python
loader = BlackRockLoader(Session_Path, Date, load_eye=True, to_microvolts=True)
# eye  data  ×1000  ->  -3118839.0  µV      info.Unit == "microVolts"
# lfp  data  ×1     ->      -11.75  µV      (already µV, unchanged)
```

Be aware that this makes eye/photodiode differ from the MATLAB export by exactly 1000×, so
only use it if your downstream analysis is Python-only or you make the matching change on
the MATLAB side.

All `.mat` products are written as MATLAB **v7.3** (HDF5, via `hdf5storage`), matching the
MATLAB loader's `-v7.3` save: these dense per-trial arrays can exceed scipy's 4 GB MAT-v5
limit, and they are mostly NaN padding, which gzips well. Note that `scipy.io.loadmat`
**cannot** read v7.3 — use `jlab_loader.load_product` (see below) or `h5py`.

## Parity with the MATLAB loader

Verified field-by-field against a full MATLAB export (`Monkey Athos`, 2026-07-24: 4575
trials, 3 sessions, 16 LFP channels, 5 sorted units, 7.5 M spikes). **Every product matches
exactly** — the only difference is the deliberately added `info.Unit`:

| Product | Result |
|---|---|
| `_expmeta.txt` | byte-identical |
| `_trials.csv` | all 4575 rows × 68 columns identical, including column order and the numeric-`NaN` / text-empty split |
| `_eye.mat`, `_photodiode.mat` | `data` identical; plus the added `info.Unit` |
| `_lfp.mat` | all 2.77 GB of `data` identical |
| `_spikes.mat` | all 216 M raster cells identical, and `MeanWaveform` / `ViolationRate` / `Channel_Number` / `Unit_No` |
| `_spikes_waveform.mat` | all 4.37 GB of `waveform` and 0.18 GB of `waveform_time` identical |

Reproduce with `tests/` for the unit level, or by exporting the same session from both
pipelines and diffing the `.mat` files field by field.

### Exact integer-tick window arithmetic

Getting to exact parity required a matching change on **both** sides (`jlab_loader` and
`BlackrockLoader.m`). Blackrock stamps everything against a PTP wall clock, so a trial start
is a number like `1521797100.2291598` s — and at that magnitude a `double` resolves only
**238.4 ns**, while a spike bin is 1 ms. Computing `floor((t_spike − t_start) / binSec)` in
seconds therefore inherits ~238 ns of rounding from *each* operand, so a spike within that
band of a bin edge can land on either side depending on the last bit. The same applies to
`ceil((Start − pre − t_ref) × fs)` for a continuous window edge.

Both loaders now keep the raw integer clock ticks of each trial's `Start` and `End` markers
and do the whole window in integer arithmetic:

```
bin = (spike_tick − (start_tick − pre_ticks)) ÷ bin_ticks        # exact
i0  = ceil_div((start_tick − pre_ticks) − ref_tick, ticks_per_sample)
i1  = floor_div((end_tick + post_ticks) − ref_tick, ticks_per_sample)
```

This is the same technique MATLAB already used for `waveform_time`; it is now applied to the
spike raster and the continuous window too. Both implementations fall back to the original
double path when the ticks are unavailable, when the two files' clocks tick at different
rates, or when the clock does not divide evenly into samples/bins and the ms buffers — so
older recordings keep working unchanged.

Before this change the two pipelines disagreed on 785 of 6.64 M spikes (0.012 %, every one
within 238 ns of a bin edge) and on 1 trial of 4575 whose window edge sat 402 ns from a
sample boundary. Both are now zero.

**NPMK compatibility note.** NPMK's `openNSx` repairs PTP clock drift by duplicating or
dropping samples at evenly spaced positions, and treats isolated single-sample gaps as part
of one continuous segment. `brpylib` does neither — it splits at every such gap and silently
drops the file's last packet. `jlab_loader` reads the PTP packets directly and applies the
same drift correction, so both readers hand the segmenter the same sample stream; without
this every trial after the first correction point would be off by one sample.

## Reading the products back

```python
from jlab_loader import load_product

eye = load_product("Blackrock_2026-07-24_eye.mat")          # or ..., "eye"
eye["data"].shape                    # (nChan, nTrials, nSamp)
eye["info"]["Unit"]                  # "milliVolts" for eye, "microVolts" for LFP
eye["timeseq"]["relative_time"]      # seconds, 0 at the Start marker

spk = load_product("Blackrock_2026-07-24_spikes.mat", "online_spike")
spk["info"]["ViolationRate"]         # per-unit ISI-violation fraction
```

`load_product` also reads MATLAB's own `*_matlab.mat` exports unchanged, since both sides
write v7.3.

## Requirements

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- `hdf5storage` (+ `h5py`) — pulled in automatically; used for every `.mat` export and by
  `load_product`

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
        │       ├── NSP-*.ns2       ← optional eye (ch 1-3) + photodiode (ch 4-6)
        │       ├── Hub-*.ns2       ← optional LFP
        │       └── HUB-*.nev       ← optional online spikes (legacy comment fallback)
        └── export_data/            ← created automatically
            └── <Date>/
                ├── Blackrock_<Date>_expmeta.txt
                ├── Blackrock_<Date>_trials.csv
                ├── Blackrock_<Date>_eye.mat             ← if load_eye=True
                ├── Blackrock_<Date>_photodiode.mat      ← if load_photodiode=True
                ├── Blackrock_<Date>_lfp.mat             ← if load_lfp=True
                ├── Blackrock_<Date>_spikes.mat          ← if load_online_spikes=True
                └── Blackrock_<Date>_spikes_waveform.mat ← if load_online_wave=True
```

Build the **session path** from your own folder structure, then pass it with the
date:

```python
from pathlib import Path
from jlab_loader import BlackRockLoader

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
| `load_eye` | `False` | load + segment the eye channels of the `NSP-*.ns2` file |
| `load_photodiode` | `False` | produce the photodiode product (channels 4-6 of the eye `.ns2`, else a dedicated `NSP-*.ns4`) |
| `load_lfp` | `False` | load + segment the `Hub-*.ns2` LFP file |
| `load_online_spikes` | `False` | load + rasterize online spikes from the `HUB-*.nev` file |
| `load_online_wave` | `False` | also segment per-spike waveforms (µV) from the `HUB-*.nev` file; implies reading spikes |
| `include_unsorted` | `False` | keep unsorted (unit 0) + noise (255) online spikes. Default `False` → only sorted units 1–5 are exported (applies to **both** spikes and waveforms) |
| `eye_marker` | `"ns2"` | eye NSx extension (use `"ns6"` etc. for other rates) |
| `eye_channels` | `(1, 2, 3)` | 1-based eye rows in the eye `.ns2` (MATLAB convention) |
| `photodiode_channels` | `(4, 5, 6)` | 1-based photodiode rows in the same file |
| `photodiode_use_separate_file` | `False` | skip the channel split and read `NSP-*.ns4` directly |
| `to_microvolts` | `False` | load the continuous streams in true µV (see "A note on units"); default keeps MATLAB's numbers and records the real unit in `info.Unit` |
| `nev_filename` | auto | pick a specific comment `.nev` instead of the `NSP-*` auto-pick |
| `eye_filename` | auto | pick a specific eye NSx file instead of the `NSP-*` auto-pick |
| `spike_nev_filename` | auto | pick a specific spike `.nev` instead of the `HUB-*` auto-pick |
| `pre_ms`, `post_ms` | `500` | trial-segmentation buffers (ms): window = `[Start - pre, End + post]` |
| `bin_ms` | `1` | spike-raster bin width (ms) |
| `isi_violation_ms` | `1` | refractory window (ms) for `info.ViolationRate` |

## Quick start

```python
from pathlib import Path
from jlab_loader import BlackRockLoader

Session_Path = Path("/path/to/your/Data") / "Monkey Porthos" / "in_lab"
Date = "2026-06-17"

# Peek at the .nev files in the folder (largest first) — optional
BlackRockLoader.list_nev_files(Session_Path, Date)

loader = BlackRockLoader(
    Session_Path, Date,
    load_eye=True,             # segment the NSP-*.ns2 eye channels
    load_photodiode=True,      # ch 4-6 of the same NSP-*.ns2
    load_lfp=True,             # segment the Hub-*.ns2 LFP file
    load_online_spikes=True,   # rasterize online spikes from the HUB-*.nev file
    load_online_wave=True,     # also segment per-spike waveforms (µV) from the HUB-*.nev file
    # nev_filename="...",      # optional: pick a specific comment .nev (else largest NSP-*)
    # eye_marker="ns2",        # eye NSx extension; default "ns2", e.g. "ns6"
    # eye_filename="...",      # optional: pick a specific eye NSx file
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
Eye file      : NSP-Porthos_20260617.ns2
LFP file      : Hub1-Porthos_20260617.ns2
Spike NEV     : HUB-Porthos_20260617.nev (spikes+waveforms)
Output dir    : /path/to/your/Data/Monkey Porthos/in_lab/export_data/2026-06-17
Loading NEV: NSP-Porthos_20260617.nev  (210 events)
  68 trials parsed (68 complete), 1 session(s)
Loading eye: NSP-Porthos_20260617.ns2
  6 channels, 279280 samples, 1000.0 Hz
  Photodiode: channels [4, 5, 6] of the eye ns2
  Eye segmented (68 trials) -> Blackrock_2026-06-17_eye.mat
  Photodiode segmented (68 trials) -> Blackrock_2026-06-17_photodiode.mat
Loading lfp: Hub1-Porthos_20260617.ns2
  16 channels, 279280 samples, 1000.0 Hz
  LFP segmented (68 trials) -> Blackrock_2026-06-17_lfp.mat
Loading spikes: HUB-Porthos_20260617.nev
  15234 spikes loaded
  Spikes rasterized (24 units x 68 trials) -> Blackrock_2026-06-17_spikes.mat

Output folder: /path/to/your/Data/Monkey Porthos/in_lab/export_data/2026-06-17
  Blackrock_2026-06-17_expmeta.txt
  Blackrock_2026-06-17_trials.csv
  Blackrock_2026-06-17_eye.mat
  Blackrock_2026-06-17_photodiode.mat
  Blackrock_2026-06-17_lfp.mat
  Blackrock_2026-06-17_spikes.mat
```

## Batch loading (multiple dates)

To process more than one session at once, use the `run_batch` **classmethod**. Pass
a **list of dates**, or omit `dates` to auto-discover every `YYYY-MM-DD` folder under
`raw_data/`. It builds one single-date `BlackRockLoader` per date, runs it through the
same load + export pipeline, and returns a summary list.

```python
from pathlib import Path
from jlab_loader import BlackRockLoader

Session_Path = Path("/path/to/your/Data") / "Monkey Porthos" / "in_lab"

# Specific dates
report = BlackRockLoader.run_batch(Session_Path, dates=["2026-06-17", "2026-06-18"])

# Every date folder under raw_data/, also exporting eye + spikes, skipping already-done dates
report = BlackRockLoader.run_batch(
    Session_Path, load_eye=True, load_online_spikes=True, skip_existing=True
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
| `load_eye` | `False` | Also segment + export each date's `NSP-*.ns2` eye channels |
| `load_photodiode` | `False` | Also export the photodiode (ch 4-6 of the eye `.ns2`) |
| `load_lfp` | `False` | Also segment + export each date's `Hub-*.ns2` LFP |
| `load_online_spikes` | `False` | Also rasterize + export online spikes from the `HUB-*.nev` file |
| `load_online_wave` | `False` | Also segment + export per-spike waveforms from the `HUB-*.nev` file |
| `include_unsorted` | `False` | Keep unsorted (unit 0) + noise (255) spikes; default `False` → sorted units only |
| `eye_marker` | `"ns2"` | eye NSx extension to load when `load_eye=True` |
| `pre_ms`, `post_ms` | `500` | Trial-segmentation buffers (ms) |
| `bin_ms` | `1` | Spike-raster bin width (ms) |
| `data_type` | `"raw_data"` | Raw-data sub-folder name |
| `output_folder` | `"export_data"` | Export sub-folder name |
| `verbose` | `True` | Print per-date progress |

> **Note:** batch mode auto-detects the **largest** matching file per role prefix in
> each date folder (`NSP-*` for comments/eye, `Hub-*` for LFP, `HUB-*` for spikes) — there's
> no per-date `nev_filename`/`eye_filename`. If a folder has several recordings and you need
> a specific one, load that date with a single `BlackRockLoader` instead. The same
> applies to per-session inspection like `print_summary()`.

## Notebooks

Open JupyterLab and run the notebooks in `notebooks/`:

```bash
jupyter lab
```

| Notebook | Description |
|---|---|
| `01_load_and_export.ipynb` | Load NSP/Hub/HUB files and export `.txt` / `.csv` / segmented eye, photodiode, LFP + spike `.mat` |
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

### `_eye.mat`, `_photodiode.mat`, `_lfp.mat`
A continuous stream segmented into trials. MATLAB structs `eye`, `photodiode` and `lfp`,
all with the same layout:

| Field | Shape | Description |
|---|---|---|
| `data` | channels × trials × samples | continuous signal (`float32`), NaN-padded to the longest trial; unit given by `info.Unit` |
| `timeseq.alignedrawtime` | trials | absolute time (s) of each trial's `Start` marker |
| `timeseq.aligned_marker` | — | `"Start"` (event that `relative_time` = 0 aligns to) |
| `timeseq.relative_time` | samples | seconds from the `Start` marker (negative through the pre-buffer) |
| `info.samplingrate` | — | sampling rate (Hz) |
| `info.Session`, `info.Trial_number` | trials | per-trial session index / trial number |
| `info.Unit` | — | the samples' real physical unit, e.g. `"milliVolts"` (eye/photodiode) or `"microVolts"` (LFP). Not present in MATLAB's struct — see "A note on units" |

`eye` rows are `eye_channels` of the source `.ns2` (1-3 by default) and `photodiode` rows
are `photodiode_channels` (4-6); as in MATLAB, the row subset itself is not recorded in the
file, so you need to know the layout out of band.

Load it in Python with `jlab_loader.load_product(path)` — these are v7.3 files, which
`scipy.io.loadmat` cannot read.

### `_spikes.mat`
Online spikes rasterized into per-trial bins. A MATLAB struct `online_spike` with the
same `timeseq` layout as above, plus:

| Field | Shape | Description |
|---|---|---|
| `data` | units × trials × bins | binary raster (0/1), NaN-padded; one row per `(channel, unit)` pair |
| `info.samplingrate` | — | bin rate (Hz), 1000 for 1 ms bins |
| `info.Channel_Number`, `info.Unit_No` | units | electrode / unit id per raster row |
| `info.ViolationRate` | units | fraction of the unit's ISIs shorter than `isi_violation_ms`, over its **full** train (not per trial); NaN for a unit with < 2 spikes |
| `info.MeanWaveform` | units × samples | mean waveform (µV) over **all** of the unit's spikes; empty unless `load_online_wave=True` |
| `info.MeanWaveformUnit` | — | `"microVolts"` |

By default rows are restricted to **sorted units** (unit 0 unsorted + unit 255 noise are
dropped); pass `include_unsorted=True` to keep all units.

### `_waveforms.mat`
Per-spike waveforms segmented into trials. A MATLAB struct `online_spike_waveform`
mirroring `segmentSpikeWaveforms`. Rows are the same `(channel, unit)` order as
`_spikes.mat`, so waveform rows line up 1:1 with the raster rows. The raw int16 waveforms
are scaled to µV per electrode (`DigitizationFactor / 1000`), like the MATLAB loader.
Keeping only sorted units (the `include_unsorted=False` default) is what keeps this dense
array small / the v7.3 file from getting huge; `include_unsorted=True` restores all units.

| Field | Shape | Description |
|---|---|---|
| `waveform` | units × trials × spikes × samples | per-spike waveform (µV), NaN-padded to the busiest `(unit, trial)` spike count |
| `waveform_time` | units × trials × spikes | each spike's time (s) relative to the `Start` marker |
| `waveform_nsamp` | — | samples per waveform |
| `waveform_unit` | — | `"microVolts"` |
| `timeseq.alignedrawtime` | trials | absolute time (s) of each trial's `Start` marker |
| `timeseq.aligned_marker` | — | `"Start"` (waveform_time = 0 at the Start marker) |
| `info.Channel_Number`, `info.Unit_No` | units | electrode / unit id per row |
| `info.maxSpikes` | — | spike-dimension length (busiest unit-trial in-window count) |

## Step-by-step API (if you want to do some basic analysis directly after exporting)

```python
from pathlib import Path
from jlab_loader import BlackRockLoader

Session_Path = Path("/path/to/your/Data") / "Monkey Porthos" / "in_lab"
loader = BlackRockLoader(Session_Path, "2026-06-17", load_eye=True,
                         load_photodiode=True, load_lfp=True,
                         load_online_spikes=True, load_online_wave=True)

# Step 1 — load and parse the comment NEV file
loader.load_nev()

# Step 2 — export .txt and .csv
loader.export()

# Step 3 — segment + export the continuous streams (each needs its own gate).
# Resolve the photodiode BEFORE exporting the eye: by default both come out of a
# single segmentation pass over the eye stream, which releases the raw data after.
loader.load_eye()
loader.load_photodiode()
loader.export_eye()
loader.export_photodiode()

loader.load_lfp()
loader.export_lfp()

# Step 4 — rasterize + export online spikes (optional; needs load_online_spikes=True)
# load_online_spikes() also reads per-spike waveforms when load_online_wave=True.
loader.load_online_spikes()
loader.export_online_spikes()

# Step 5 — segment + export per-spike waveforms (optional; needs load_online_wave=True)
loader.export_online_wave()

# Access parsed data directly
loader.experiments   # list of experiment-metadata dicts (one per session)
loader.trials        # list of per-trial dicts
loader.eye           # segmented eye product (dict: 'data', 'timeseq', 'info')
loader.photodiode    # segmented photodiode product
loader.lfp           # segmented LFP product
loader.raw           # raw streams still held, keyed by role ('eye'/'lfp'/'photodiode')
loader.spike_times   # spike timestamps (s); also spike_channel / spike_unit
loader.spike_waveform # per-spike waveforms (µV, nSpikes × nSamp) if load_online_wave=True
```

## Project structure

```
JLab_Python/
├── jlab_loader/
│   ├── loader.py        # BlackRockLoader — main user-facing class
│   ├── _parser.py       # event comment parsing
│   ├── _features.py     # derived feature computation (angles, eccentricity)
│   ├── _exporter.py     # writes .txt and .csv files
│   ├── _continuous.py   # segment_continuous — eye/LFP/photodiode → per-trial 3D array,
│   │                    #   plus the NPMK PTP clock-drift alignment
│   ├── _spikes.py       # segment_spikes + drop_units — source-agnostic spike align/filter
│   ├── _waveforms.py    # segment_waveforms — per-spike waveforms → per-trial 4D array
│   ├── _matio.py        # v7.3 (HDF5) writer/reader with MATLAB orientations
│   └── _constants.py    # file schema, event maps, regex patterns
├── brpylib/             # bundled Blackrock file reader
├── notebooks/
│   ├── 01_load_and_export.ipynb
│   ├── 02_psychometric_analysis.ipynb
│   └── 03_eye_signal_check.ipynb
└── pyproject.toml
```

## Extending to offline spikes

The spike pipeline is **source-agnostic** so online and offline (sorted) spikes can share it
— only the *reader* differs:

```
reader  →  common per-spike arrays (times, channel, unit[, waveform])  →  segment_*  →  export
```

- **Shared (already source-agnostic):** `segment_spikes` and `segment_waveforms`
  (`jlab_loader/_spikes.py`, `jlab_loader/_waveforms.py`) align loose per-spike arrays to trials with no
  knowledge of where they came from; `drop_units` (`jlab_loader/_spikes.py`) is the shared
  unsorted/noise filter (defaults to ids `0`/`255`, overridable via `drop_ids`).
- **Source-specific (today):** `BlackRockLoader._read_online_spikes` reads the online HUB
  NEV into those arrays.
- **To add offline later:** write a parallel `_read_offline_spikes` (from the sorted-spike
  source) that returns the same `(times, channel, unit[, waveform])`, plus
  `load_offline_spikes` / `export_offline_spikes` methods that reuse the **same**
  `segment_*` + `drop_units` and write `offline_spike` / `offline_spike_waveform` outputs.
  No change to the segmentation code is needed.
