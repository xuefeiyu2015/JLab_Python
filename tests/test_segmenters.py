"""Unit tests for the trial segmenters and the MATLAB-parity details.

These run on synthetic data — no Blackrock files needed — and pin the behaviours
that were verified field-by-field against the MATLAB exports:

* segment_continuous picks the same inclusive sample window a brute-force
  ``(t >= t0) & (t <= t1)`` mask would, using realistic ~1.5e9 s epoch
  timestamps where a double resolves only ~240 ns;
* µV scaling, float32 data and float64 times;
* ViolationRate / MeanWaveform computed over each unit's FULL train;
* waveform_time from exact uint64 tick subtraction, which beats the
  seconds-domain subtraction it falls back to;
* the memory-saccade Task relabel and the MATLAB derived-field order;
* CSV column order and the numeric-NaN / text-empty split.

Run with:  PYTHONPATH=. .venv/bin/python -m pytest tests/ -q
"""
import math

import numpy as np
import pytest

from jlab_loader._continuous import (
    align_ptp_clock_drift,
    clamp_channels,
    segment_continuous,
    subset_channels,
)
from jlab_loader._exporter import _COLUMN_ORDER, _csv_value, _mat2str
from jlab_loader._features import DERIVED_FIELDS, compute_derived_features
from jlab_loader._spikes import drop_units, segment_spikes
from jlab_loader._waveforms import segment_waveforms

T0 = 1521786290.123456  # a real absolute epoch timestamp: the precision trap
TICK = 1e9


@pytest.fixture
def stream():
    rng = np.random.default_rng(0)
    fs = 1000.0
    n = 200_000
    t = T0 + np.arange(n) / fs
    data = rng.integers(-30000, 30000, size=(6, n)).astype(np.int16)
    uv = np.array([0.25, 0.25, 0.25, 1.0, 1.0, 1.0])
    trials = []
    for k in range(200):
        s = T0 + 5 + k * 0.6 + rng.random() * 0.1
        trials.append({"Start": s, "End": s + 0.3 + rng.random() * 0.2,
                       "Session": 1, "Trial_number": k})
    trials.append({"Start": math.nan, "End": math.nan,
                   "Session": 1, "Trial_number": 998})     # missing marker
    trials.append({"Start": T0 - 1000, "End": T0 - 999,
                   "Session": 1, "Trial_number": 999})     # before the recording
    return trials, data, t, fs, uv


def test_continuous_window_matches_bruteforce(stream):
    trials, data, t, fs, uv = stream
    A = segment_continuous(trials, data, t, fs, 500, 500, uv)
    for i, tr in enumerate(trials):
        s, e = tr["Start"], tr["End"]
        if math.isnan(s) or math.isnan(e):
            assert np.all(np.isnan(A["data"][:, i, :]))
            assert np.isnan(A["timeseq"]["alignedrawtime"][i])
            continue
        want = np.flatnonzero((t >= s - 0.5) & (t <= e + 0.5))
        got = int(np.sum(~np.isnan(A["data"][0, i, :])))
        assert got == want.size, f"trial {i}"
        if want.size:
            ref = data[:, want].astype(np.float32) * uv.reshape(-1, 1).astype(np.float32)
            assert np.allclose(A["data"][:, i, : want.size], ref, equal_nan=True)


def test_continuous_dtypes_and_alignment(stream):
    trials, data, t, fs, uv = stream
    A = segment_continuous(trials, data, t, fs, 500, 500, uv)
    assert A["data"].dtype == np.float32                       # voltages: single
    assert A["timeseq"]["relative_time"].dtype == np.float64   # times: double
    assert A["timeseq"]["alignedrawtime"].dtype == np.float64
    assert A["timeseq"]["aligned_marker"] == "Start"
    # relative_time is 0 at the Start marker, negative through the pre-buffer
    assert A["timeseq"]["relative_time"][0] == pytest.approx(-0.5)
    # µV scaling actually applied (channel 0 scaled by 0.25, channel 3 by 1.0)
    assert np.nanmax(np.abs(A["data"][0])) < np.nanmax(np.abs(A["data"][3]))


def test_channel_split(stream):
    trials, data, t, fs, uv = stream
    A = segment_continuous(trials, data, t, fs, 500, 500, uv)
    eye = subset_channels(A, [1, 2, 3])       # 1-based, MATLAB convention
    pd_ = subset_channels(A, [4, 5, 6])
    assert np.array_equal(eye["data"], A["data"][0:3], equal_nan=True)
    assert np.array_equal(pd_["data"], A["data"][3:6], equal_nan=True)
    # info/timeseq are shared, not re-indexed
    assert eye["info"]["samplingrate"] == A["info"]["samplingrate"]


def test_unit_recorded_and_subset(stream):
    """info.Unit is the one deliberate addition to MATLAB's schema."""
    trials, data, t, fs, uv = stream
    # omitted by default -> exactly MATLAB's field set
    assert "Unit" not in segment_continuous(trials, data, t, fs, 500, 500, uv)["info"]

    A = segment_continuous(trials, data, t, fs, 500, 500, uv, "milliVolts")
    assert A["info"]["Unit"] == "milliVolts"
    assert subset_channels(A, [4, 5, 6])["info"]["Unit"] == "milliVolts"

    # a per-channel unit list is sliced alongside the data
    per_ch = ["mV", "mV", "mV", "uV", "uV", "uV"]
    B = segment_continuous(trials, data, t, fs, 500, 500, uv, per_ch)
    assert subset_channels(B, [1, 2, 3])["info"]["Unit"] == ["mV", "mV", "mV"]
    assert subset_channels(B, [4, 5, 6])["info"]["Unit"] == ["uV", "uV", "uV"]
    # and slicing must not mutate the parent
    assert B["info"]["Unit"] == per_ch


def test_unit_conversion_factors():
    """to_microvolts folds the header unit into the scale factor."""
    from jlab_loader._constants import UNIT_LABELS, UNIT_TO_MICROVOLTS

    assert UNIT_TO_MICROVOLTS["uV"] == 1.0
    assert UNIT_TO_MICROVOLTS["mV"] == 1e3
    assert UNIT_TO_MICROVOLTS["V"] == 1e6
    assert UNIT_LABELS["mV"] == "milliVolts"
    assert UNIT_LABELS["uV"] == "microVolts"


def test_clamp_channels_drops_missing():
    with pytest.warns(UserWarning):
        assert clamp_channels([4, 5, 6], 3, "photodiode_channels").tolist() == []
    assert clamp_channels([1, 2, 3], 6, "eye_channels").tolist() == [1, 2, 3]


def test_ptp_alignment_matches_npmk_convention():
    """NPMK measures the segment as ``last - first + 1`` ticks, which spans only
    n-1 sample periods. A recording whose timestamps step by exactly the nominal
    period therefore reads as one sample short and gets one removed — the
    formula is reproduced as-is (openNSx.m:1373-1440), quirk included."""
    fs, tres = 1000.0, 1e9
    n = 30_000
    step = np.uint64(1_000_000)  # 1 ms at 1e9 ticks/s
    base = np.uint64(int(T0 * tres))
    data = np.arange(n, dtype=np.int16).reshape(1, n)

    nominal = np.arange(n, dtype=np.uint64) * step + base
    out, added = align_ptp_clock_drift(data, nominal, fs, tres)
    assert added == -1 and out.shape[1] == n - 1

    # Stretch the span by one full sample period -> the books balance, no edit.
    balanced = nominal.copy()
    balanced[-1] += step - np.uint64(1)
    out, added = align_ptp_clock_drift(data, balanced, fs, tres)
    assert added == 0 and out.shape == data.shape

    # Clock running fast -> samples are added; running slow -> removed. 200 ns
    # per sample over 30000 samples is ~6 sample periods of drift, comfortably
    # clear of the rounding boundary.
    drift = (np.arange(n) * 200).astype(np.int64)
    fast = (nominal.astype(np.int64) + drift).astype(np.uint64)
    out, added = align_ptp_clock_drift(data, fast, fs, tres)
    assert added > 0 and out.shape[1] == n + added
    slow = (nominal.astype(np.int64) - drift).astype(np.uint64)
    out, added = align_ptp_clock_drift(data, slow, fs, tres)
    assert added < 0 and out.shape[1] == n + added

    # Edits are interior only: the first and last samples always survive.
    assert out[0, 0] == data[0, 0] and out[0, -1] == data[0, -1]


def test_ptp_alignment_skipped_for_non_ptp():
    """Older files (TimeRes <= 1e5, or no per-sample stamps) are left alone."""
    data = np.arange(1000, dtype=np.int16).reshape(1, 1000)
    out, added = align_ptp_clock_drift(data, None, 1000.0, 1e9)
    assert added == 0 and out is data
    ts = np.arange(1000, dtype=np.uint64)
    out, added = align_ptp_clock_drift(data, ts, 1000.0, 30000)
    assert added == 0 and out is data


# ── exact integer-tick window arithmetic ────────────────────────────────────

TICK = 1_000_000_000
REF_TICK = 1_521_786_290_123_456_789


def test_exact_tick_grid_guards_non_integral_rates():
    from jlab_loader._continuous import exact_tick_grid

    assert exact_tick_grid(TICK, 1000.0, 500, 500) == (1_000_000, 500_000_000, 500_000_000)
    # a rate that does not divide the clock evenly must refuse the exact path
    assert exact_tick_grid(TICK, 30000 / 7, 500, 500) is None
    assert exact_tick_grid(TICK, 1000.0, 1 / 3, 500) is None   # non-integral buffer
    assert exact_tick_grid(None, 1000.0, 500, 500) is None
    # 0.3 ms IS integral at a 1e9 clock (300000 ticks) — the guard is about
    # divisibility, not about being a whole number of milliseconds.
    assert exact_tick_grid(TICK, 1000.0, 0.3, 500) == (1_000_000, 300_000, 500_000_000)


@pytest.fixture
def tick_trials():
    """Trials whose window edges sit exactly on a sample boundary — the case a
    double cannot resolve at ~1.5e9 s."""
    rng = np.random.default_rng(0)
    per_samp = TICK // 1000
    trials, st, en = [], [], []
    for k in range(120):
        start = REF_TICK + (500 + k * 1200) * per_samp   # exactly on a boundary
        if k % 3:
            start += int(rng.integers(1, per_samp))      # jitter most of them
        end = start + 300 * per_samp
        trials.append({"Start": start / TICK, "End": end / TICK,
                       "Session": 1, "Trial_number": k})
        st.append(start)
        en.append(end)
    return trials, np.array(st, dtype=np.int64), np.array(en, dtype=np.int64)


def test_exact_continuous_window_matches_integer_truth(tick_trials):
    trials, st, en = tick_trials
    per_samp = TICK // 1000
    n = 400_000
    abs_time = (REF_TICK + np.arange(n, dtype=np.int64) * per_samp) / TICK
    data = np.arange(2 * n, dtype=np.int16).reshape(2, n)

    A = segment_continuous(trials, data, abs_time, 1000.0, 500, 500,
                           start_ticks=st, end_ticks=en,
                           ref_tick=REF_TICK, time_res=TICK)
    got = np.sum(~np.isnan(A["data"][0]), axis=1)
    for i in range(len(trials)):
        lo = -(-(int(st[i]) - 500 * per_samp - REF_TICK) // per_samp)   # ceil
        hi = (int(en[i]) + 500 * per_samp - REF_TICK) // per_samp       # floor
        assert got[i] == hi - lo + 1, f"trial {i}"


def test_exact_spike_bins_match_integer_truth(tick_trials):
    trials, st, en = tick_trials
    per_bin = TICK // 1000
    rng = np.random.default_rng(1)
    ns = 20_000
    ticks = np.sort(rng.integers(REF_TICK, REF_TICK + 200 * TICK, size=ns)).astype(np.int64)
    # force a chunk of them exactly onto bin edges
    edge = rng.choice(ns, 2000, replace=False)
    ticks[edge] = (ticks[edge] // per_bin) * per_bin
    ticks = np.sort(ticks)
    chan = rng.choice([1.0, 2.0], size=ns)
    unit = np.ones(ns)

    R = segment_spikes(trials, ticks / TICK, chan, unit, 500, 500, 1,
                       spike_ticks=ticks, start_ticks=st, end_ticks=en,
                       time_res=TICK)
    keys = np.unique(np.column_stack([chan, unit]), axis=0)
    for i in range(len(trials)):
        lo = int(st[i]) - 500 * per_bin
        hi = int(en[i]) + 500 * per_bin
        nb = (hi - lo) // per_bin
        ref = np.zeros((len(keys), nb), dtype=np.float32)
        for idx in np.flatnonzero((ticks >= lo) & (ticks < hi)):
            b = min((int(ticks[idx]) - lo) // per_bin, nb - 1)
            r = int(np.flatnonzero((keys[:, 0] == chan[idx])
                                   & (keys[:, 1] == unit[idx]))[0])
            ref[r, b] = 1.0
        assert np.array_equal(R["data"][:, i, :nb], ref), f"trial {i}"


def test_exact_path_resolves_ties_the_float_path_cannot(tick_trials):
    """The two paths must agree except where the edge is inside the ULP band."""
    trials, st, en = tick_trials
    per_bin = TICK // 1000
    rng = np.random.default_rng(2)
    ns = 20_000
    ticks = np.sort(rng.integers(REF_TICK, REF_TICK + 200 * TICK, size=ns)).astype(np.int64)
    ticks[rng.choice(ns, 2000, replace=False)] //= per_bin
    ticks = np.sort(np.where(ticks < REF_TICK, REF_TICK, ticks))
    chan = np.ones(ns)
    unit = np.ones(ns)

    a = segment_spikes(trials, ticks / TICK, chan, unit, 500, 500, 1)
    b = segment_spikes(trials, ticks / TICK, chan, unit, 500, 500, 1,
                       spike_ticks=ticks, start_ticks=st, end_ticks=en,
                       time_res=TICK)
    # Same geometry and the same NaN padding: n_bins is float-derived in both
    # paths, so only the placement of 1s inside the window can differ.
    assert a["data"].shape == b["data"].shape
    assert np.array_equal(np.isnan(a["data"]), np.isnan(b["data"]))
    # The two agree everywhere except the boundary ties. (Totals are NOT a valid
    # invariant: the raster is binary, so a shifted spike can merge into a bin
    # that is already occupied.)
    in_window = int(np.sum(~np.isnan(a["data"])))
    differing = int(np.sum(~np.isclose(a["data"], b["data"], equal_nan=True)))
    assert differing < 0.005 * in_window, f"{differing} of {in_window} differ"


def test_exact_path_declined_when_ticks_missing(tick_trials):
    """No ticks -> silently use the float path, same as before."""
    trials, st, en = tick_trials
    rng = np.random.default_rng(3)
    ticks = np.sort(rng.integers(REF_TICK, REF_TICK + 200 * TICK, size=500)).astype(np.int64)
    chan = np.ones(500)
    unit = np.ones(500)
    base = segment_spikes(trials, ticks / TICK, chan, unit, 500, 500, 1)
    # start_ticks of the wrong length must not silently mis-align
    other = segment_spikes(trials, ticks / TICK, chan, unit, 500, 500, 1,
                           spike_ticks=ticks, start_ticks=st[:5],
                           end_ticks=en[:5], time_res=TICK)
    assert np.array_equal(base["data"], other["data"], equal_nan=True)


# ── spikes ──────────────────────────────────────────────────────────────────

@pytest.fixture
def spikes():
    rng = np.random.default_rng(1)
    n = 20_000
    ticks = np.sort(rng.integers(int(T0 * TICK), int((T0 + 60) * TICK), size=n)).astype(np.uint64)
    times = ticks.astype(float) / TICK
    chan = rng.choice([1.0, 1.0, 2.0, 7.0], size=n)
    unit = rng.choice([1.0, 2.0], size=n)
    wf = rng.normal(0, 50, size=(n, 48)).astype(np.float32)
    trials, start_ticks = [], []
    for k in range(40):
        st = T0 + 1 + k * 1.4
        trials.append({"Start": st, "End": st + 0.5, "Session": 1, "Trial_number": k})
        start_ticks.append(int(round(st * TICK)))
    trials.append({"Start": math.nan, "End": math.nan, "Session": 1, "Trial_number": 99})
    start_ticks.append(0)
    return trials, times, chan, unit, wf, ticks, np.array(start_ticks, dtype=np.uint64)


def test_raster_matches_bruteforce(spikes):
    trials, times, chan, unit, wf, _, _ = spikes
    R = segment_spikes(trials, times, chan, unit, 500, 500, 1, 1, wf)
    assert R["data"].dtype == np.float32
    keys = np.unique(np.column_stack([chan, unit]), axis=0)
    for i, tr in enumerate(trials):
        if math.isnan(tr["Start"]):
            assert np.all(np.isnan(R["data"][:, i, :]))
            continue
        t0, t1 = tr["Start"] - 0.5, tr["End"] + 0.5
        nb = int(max(round((t1 - t0) / 1e-3), 0))
        ref = np.zeros((len(keys), nb), dtype=np.float32)
        for idx in np.flatnonzero((times >= t0) & (times < t1)):
            b = min(int(np.floor((times[idx] - t0) / 1e-3)), nb - 1)
            r = int(np.flatnonzero((keys[:, 0] == chan[idx]) & (keys[:, 1] == unit[idx]))[0])
            ref[r, b] = 1.0
        assert np.array_equal(R["data"][:, i, :nb], ref), f"trial {i}"
        assert np.all(np.isnan(R["data"][:, i, nb:]))


def test_violation_rate_and_mean_waveform_use_full_train(spikes):
    trials, times, chan, unit, wf, _, _ = spikes
    R = segment_spikes(trials, times, chan, unit, 500, 500, 1, 1, wf)
    info = R["info"]
    assert info["MeanWaveformUnit"] == "microVolts"
    keys = np.unique(np.column_stack([chan, unit]), axis=0)
    for r, (ch, un) in enumerate(keys):
        m = (chan == ch) & (unit == un)
        isi = np.diff(np.sort(times[m]))
        want = np.mean(isi < 1e-3) if isi.size else np.nan
        assert info["ViolationRate"][r] == pytest.approx(want, nan_ok=True)
        assert np.allclose(info["MeanWaveform"][r], wf[m].astype(np.float64).mean(axis=0))


def test_mean_waveform_empty_without_waveforms(spikes):
    trials, times, chan, unit, _, _, _ = spikes
    R = segment_spikes(trials, times, chan, unit)
    assert R["info"]["MeanWaveform"].size == 0    # MATLAB []


def test_drop_units_keeps_arrays_aligned(spikes):
    _, times, chan, unit, wf, ticks, _ = spikes
    u = np.where(np.arange(len(unit)) % 5 == 0, 0.0, unit)   # make some unsorted
    a, b, c, w, tk, n = drop_units(times, chan, u, wf, ticks)
    assert a.size == b.size == c.size == w.shape[0] == tk.size == len(times) - n
    assert not np.isin(c, [0, 255]).any()


# ── waveforms ───────────────────────────────────────────────────────────────

def test_waveform_time_uses_exact_ticks(spikes):
    trials, times, chan, unit, wf, ticks, start_ticks = spikes
    W = segment_waveforms(trials, times, chan, unit, wf, 500, 500,
                          ticks, start_ticks, TICK)
    assert W["waveform"].dtype == np.float32
    assert W["waveform_time"].dtype == np.float64
    # MATLAB stores these scalars as double, not int
    assert isinstance(W["waveform_nsamp"], float)
    assert isinstance(W["info"]["maxSpikes"], float)

    keys = np.unique(np.column_stack([chan, unit]), axis=0)
    for i, tr in enumerate(trials[:6]):
        if math.isnan(tr["Start"]):
            continue
        t0, t1 = tr["Start"] - 0.5, tr["End"] + 0.5
        pos: dict = {}
        for idx in np.flatnonzero((times >= t0) & (times < t1)):
            r = int(np.flatnonzero((keys[:, 0] == chan[idx]) & (keys[:, 1] == unit[idx]))[0])
            p = pos.get(r, 0)
            assert np.allclose(W["waveform"][r, i, p, :], wf[idx])
            # exact reference: subtract in Python ints, then divide
            exact = (int(ticks[idx]) - int(start_ticks[i])) / TICK
            assert W["waveform_time"][r, i, p] == pytest.approx(exact, abs=1e-12)
            pos[r] = p + 1

    # the tick path must actually beat the seconds-domain fallback
    W2 = segment_waveforms(trials, times, chan, unit, wf, 500, 500)
    assert np.nanmax(np.abs(W["waveform_time"] - W2["waveform_time"])) > 0


def test_waveform_rows_match_raster_rows(spikes):
    trials, times, chan, unit, wf, ticks, start_ticks = spikes
    R = segment_spikes(trials, times, chan, unit, 500, 500, 1, 1, wf)
    W = segment_waveforms(trials, times, chan, unit, wf, 500, 500,
                          ticks, start_ticks, TICK)
    assert np.array_equal(R["info"]["Channel_Number"], W["info"]["Channel_Number"])
    assert np.array_equal(R["info"]["Unit_No"], W["info"]["Unit_No"])


# ── trial features and CSV formatting ───────────────────────────────────────

def test_memory_saccade_relabel():
    """Both conditions must hold — MATLAB BlackrockLoader.m:2838-2846."""
    trials = [
        {"Trial_type": "memory", "Task": "visual_saccades_experiment"},   # relabelled
        {"Trial_type": "memory", "Task": "time_delay_experiment"},        # left alone
        {"Trial_type": None, "Task": "visual_saccades_experiment"},       # left alone
        {"Trial_type": "visual", "Task": "visual_saccades_experiment"},   # left alone
    ]
    for t in trials:
        t.setdefault("Target_1_position", [math.nan, math.nan])
        t.setdefault("Target_2_position", [math.nan, math.nan])
    compute_derived_features(trials)
    assert [t["Task"] for t in trials] == [
        "memory_saccades_experiment",
        "time_delay_experiment",
        "visual_saccades_experiment",
        "visual_saccades_experiment",
    ]


def test_derived_field_order_matches_matlab():
    assert DERIVED_FIELDS == [
        "Target_1_angle", "Target_2_angle", "Stimulus_direction",
        "Choose_target", "Choose_leftright",
        "Target_1_eccentricity", "Target_2_eccentricity",
    ]


def test_csv_column_order_puts_coords_last():
    assert _COLUMN_ORDER[0] == "index"
    # the 7 derived fields come before the six split coordinate columns
    assert _COLUMN_ORDER[-6:] == [
        "Fixation_position_x", "Fixation_position_y",
        "Target_1_position_x", "Target_1_position_y",
        "Target_2_position_x", "Target_2_position_y",
    ]
    assert _COLUMN_ORDER[-13:-6] == DERIVED_FIELDS
    for c in ("Fixation_position", "Target_1_position", "Target_2_position"):
        assert c not in _COLUMN_ORDER


def test_csv_nan_encoding_depends_on_column_kind():
    """writetable prints numeric NaN as 'NaN' but a missing string as empty."""
    assert _csv_value(math.nan, text_column=False) == "NaN"
    assert _csv_value(math.nan, text_column=True) == ""
    assert _csv_value(None, text_column=True) == ""
    assert _csv_value(-90.0, text_column=False) == "-90"
    assert _csv_value(7.00035713374682, text_column=False) == "7.00035713374682"


def test_mat2str_uses_15_significant_digits():
    assert _mat2str(1521786290.8498201) == "1521786290.84982"
    assert _mat2str([53.0, 30.0]) == "[53 30]"
    assert _mat2str(math.nan) == "NaN"
    assert _mat2str(None) == "NaN"
    assert _mat2str("right") == "right"
