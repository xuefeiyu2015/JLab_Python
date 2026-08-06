"""
Continuous (NSx) trial segmentation — eye, LFP and photodiode.
Cuts a continuous stream into one slice per trial, mirroring
BlackrockLoader.segmentContinuous in the MATLAB version.

Precision rule, copied from the MATLAB class header: VOLTAGES are single,
TIMES are always double. ``.data`` is float32; every timestamp stays float64.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from ._constants import SEGMENT_POST_MS, SEGMENT_PRE_MS


def _isnan(v) -> bool:
    try:
        return math.isnan(v)
    except (TypeError, ValueError):
        return False


def _trial_bounds(trials: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Start/End (seconds) per trial plus an ``ok`` mask for usable trials.

    A trial missing either marker gets an all-NaN slice, which is what keeps the
    trial dimension index-aligned 1:1 with ``trials``.
    """
    n = len(trials)
    starts = np.full(n, np.nan)
    ends = np.full(n, np.nan)
    for i, tr in enumerate(trials):
        s, e = tr.get("Start"), tr.get("End")
        if not _isnan(s):
            starts[i] = s
        if not _isnan(e):
            ends[i] = e
    ok = ~np.isnan(starts) & ~np.isnan(ends)
    return starts, ends, ok


def exact_tick_grid(time_res, rate, *ms_buffers):
    """Ticks per sample/bin and per buffer, or None when they are not integral.

    The exact-integer window arithmetic only works when the clock divides evenly
    into samples and into each ms buffer — true for the usual 1e9 tick / 1 kHz /
    500 ms combination (1e6 ticks per sample, 5e8 per buffer), but not for an
    arbitrary rate. Returns None to signal "fall back to the float path".
    """
    if not time_res or not rate:
        return None
    per_unit = time_res / rate
    out = [per_unit]
    for ms in ms_buffers:
        out.append(ms * time_res / 1000.0)
    if any(v != int(v) for v in out):
        return None
    return tuple(int(v) for v in out)


def _ceil_div(a: np.ndarray, b: int) -> np.ndarray:
    """Ceiling division for signed integers (numpy // floors toward -inf)."""
    return -np.floor_divide(-a, b)


def _matlab_round(x: float) -> int:
    """MATLAB's round(): half away from zero. numpy rounds half to even."""
    return int(math.floor(abs(x) + 0.5) * (1 if x >= 0 else -1))


def align_ptp_clock_drift(
    data: np.ndarray,
    timestamps: np.ndarray | None,
    samplingrate: float,
    time_res: float,
) -> tuple[np.ndarray, int]:
    """Duplicate or drop samples to correct PTP clock drift, as NPMK does.

    Port of the "samplealign" block in NPMK's openNSx.m (lines 1373-1440), which
    MATLAB applies to every one-sample-per-packet PTP file (``TimeRes > 1e5``).
    brpylib returns the packets verbatim, so without this the two readers hand
    ``segment_continuous`` different sample streams and every trial after the
    first correction point is off by one sample.

    The true sampling rate implied by the PTP timestamps differs slightly from
    the nominal rate, so over a long recording the stream drifts by a whole
    sample or more. NPMK measures that drift, then adds (duplicating the last
    column) or removes (dropping the last column) one sample at each of
    ``abs(added)`` evenly spaced chunk boundaries — never at the very start or
    end of the recording.

    Returns ``(data, n_added)``; ``n_added`` is negative when samples were
    removed and 0 when nothing changed (``data`` is then returned unmodified).
    """
    n = data.shape[1]
    if timestamps is None or n == 0 or time_res <= 1e5:
        return data, 0
    timestamps = np.asarray(timestamps).ravel()
    if timestamps.size != n or n < 2:
        # Not one-sample-per-packet, so there is no per-sample drift to correct.
        return data, 0

    # segmentDurations = timestampLast - timestampFirst + 1  (openNSx.m:918)
    duration = int(timestamps[-1]) - int(timestamps[0]) + 1
    ratio = duration / n / time_res * samplingrate
    added = _matlab_round((ratio - 1) * n)
    if added == 0:
        return data, 0

    gap = _matlab_round(n / (abs(added) + 1))
    if gap >= n:
        half = _matlab_round(n / 2)
        sizes = [half, n - half]
    else:
        sizes = [gap] * abs(added) + [n - gap * abs(added)]

    # Edit every chunk except the last, at its right-hand boundary.
    bounds = np.cumsum(sizes)
    edges = bounds[:-1]
    if added < 0:
        data = np.delete(data, edges - 1, axis=1)  # drop each chunk's last column
    else:
        # duplicate each chunk's last column, i.e. insert a copy just after it
        data = np.insert(data, edges, data[:, edges - 1], axis=1)
    return data, added


def clamp_channels(rows, n_chan: int, name: str) -> np.ndarray:
    """Drop requested (1-based) channel rows the stream does not actually have.

    Port of BlackrockLoader.clampChannels. A file with fewer channels than the
    configured layout degrades to the rows that exist instead of erroring
    mid-pipeline.
    """
    rows = np.asarray(rows, dtype=int).ravel()
    keep = rows <= n_chan
    if not np.all(keep):
        warnings.warn(
            f"{name} requests channel(s) {rows[~keep].tolist()} but the stream "
            f"has only {n_chan}; using {rows[keep].tolist()}.",
            stacklevel=2,
        )
        rows = rows[keep]
    return rows


def subset_channels(product: dict, rows) -> dict:
    """Take a channel subset of a segment_continuous product.

    Port of BlackrockLoader.subsetChannels. Only ``data`` is indexed;
    ``timeseq``/``info`` describe the trials and the clock, which the subset
    shares, so they carry through unchanged. ``rows`` is 1-based (MATLAB
    convention) and converted here.

    The one exception is ``info.Unit`` when it is per-channel (a stream whose
    channels are not all in the same unit): that is sliced alongside ``data``.
    """
    idx = np.asarray(rows, dtype=int).ravel() - 1  # 1-based config -> 0-based
    out = dict(product)
    out["data"] = product["data"][idx, :, :]
    info = dict(product["info"])
    unit = info.get("Unit")
    if isinstance(unit, list):
        info["Unit"] = [unit[i] for i in idx]
    out["info"] = info
    return out


def segment_continuous(
    trials: list[dict],
    nsxdata: np.ndarray,
    nsx_abs_time: np.ndarray,
    nsx_samplingrate: float,
    pre_ms: float = SEGMENT_PRE_MS,
    post_ms: float = SEGMENT_POST_MS,
    uv_scale: np.ndarray | None = None,
    unit: str | list[str] | None = None,
    *,
    start_ticks=None,
    end_ticks=None,
    ref_tick: int | None = None,
    time_res: float | None = None,
) -> dict:
    """
    Cut a continuous stream into one slice per trial.

    Port of BlackrockLoader.segmentContinuous. For each trial the window is
    ``[Start - pre_ms, End + post_ms]`` (ms buffers), matched against
    ``nsx_abs_time`` (seconds, same clock as the event timestamps). Slices are
    left-aligned (each starts at its own window start) and NaN-padded to the
    longest trial, so the result is one ``nChan × nTrials × maxSamples`` array
    that lines up 1:1 with ``trials``. A trial whose Start/End is NaN (or that
    has no samples in range) gets an all-NaN slice.

    ``nsxdata`` may be raw int16 (as the loader now reads it) together with
    ``uv_scale``, a per-channel µV-per-digit vector; each slice is then scaled
    on the way in. Pass ``None`` when the data is already in µV.

    The sample window is computed arithmetically rather than searched:
    ``nsx_abs_time`` is uniform by construction, so the bounds follow from the
    first sample's time and the sampling rate. Only ``nsx_abs_time[0]`` is read,
    which keeps this O(nTrials) instead of scanning the whole time vector once
    per trial — at ~9e6 samples × ~4.5e3 trials the scan is not viable.

    Parameters
    ----------
    trials : list[dict]
        Parsed trials; each needs "Start" and "End" (seconds), plus "Session"
        and "Trial_number".
    nsxdata : np.ndarray
        Continuous data, shape ``(nChan, nSamples)``. Raw int16 or µV.
    nsx_abs_time : np.ndarray
        Absolute time (seconds) of each sample, shape ``(nSamples,)``.
    nsx_samplingrate : float
        Sampling rate in Hz.
    pre_ms, post_ms : float
        Buffer (ms) before Start / after End. Default 500 each.
    uv_scale : np.ndarray or None
        Per-channel scale per digit, shape ``(nChan,)``. None -> already scaled.
    unit : str or list[str] or None
        Physical unit of the scaled samples, recorded as ``info.Unit``. A list
        gives one entry per channel, for a stream whose channels differ. None
        omits the field entirely (exactly MATLAB's schema).

    Returns
    -------
    dict mirroring the MATLAB ``A`` struct:
        "data"     : (nChan, nTrials, maxSamples) float32, NaN-padded
        "timeseq"  : {"alignedrawtime", "aligned_marker", "relative_time"}
        "info"     : {"samplingrate", "Session", "Trial_number"[, "Unit"]}
    """
    pre = pre_ms / 1000.0
    post = post_ms / 1000.0

    nsxdata = np.asarray(nsxdata)
    nsx_abs_time = np.asarray(nsx_abs_time, dtype=float).ravel()
    n_chan = nsxdata.shape[0]
    n_sample = nsxdata.shape[1]
    n_trials = len(trials)

    if uv_scale is not None:
        # (nChan, 1) so it broadcasts down the channel dimension of a slice.
        uv_scale = np.asarray(uv_scale, dtype=float).ravel().reshape(-1, 1)

    starts, ends, ok = _trial_bounds(trials)

    # --- first pass: each trial's sample window, in closed form ---
    # nsx_abs_time[k] = tRef + k/fs, so the first sample at or after t0 is
    # ceil((t0-tRef)*fs) and the last at or before t1 is floor((t1-tRef)*fs)
    # (0-based) -- the same inclusive window a >= t0 & <= t1 mask produces.
    #
    # These are absolute epoch timestamps (~1.5e9 s), where a double resolves
    # only ~2.4e-7 s. At 1 kHz that is ~2.4e-4 of a sample, so the stored grid
    # is not perfectly even and the arithmetic above can land one sample off at
    # a window edge. The candidate is therefore snapped against the actual
    # sample times, still O(1) per trial: only the candidate and its neighbour
    # are read, never the whole vector.
    t_ref = nsx_abs_time[0] if n_sample else 0.0
    t0 = starts - pre
    t1 = ends + post

    grid = None
    if start_ticks is not None and end_ticks is not None and ref_tick is not None:
        st = np.asarray(start_ticks, dtype=np.int64).ravel()
        en = np.asarray(end_ticks, dtype=np.int64).ravel()
        if st.size == n_trials and en.size == n_trials:
            grid = exact_tick_grid(time_res, nsx_samplingrate, pre_ms, post_ms)

    if grid is not None:
        # --- exact path: the whole window in integer ticks ---
        # The sample grid is uniform by construction (t_ref + k/fs), so index k
        # sits at tick ref_tick + k*tps exactly. Solving for the first index at
        # or after t0 and the last at or before t1 is then integer division —
        # no rounding, and the neighbour-snap below becomes unnecessary.
        tps, pre_ticks, post_ticks = grid
        have = ok & (st != 0) & (en != 0)
        i0 = np.zeros(n_trials, dtype=np.int64)
        i1 = np.full(n_trials, -1, dtype=np.int64)
        i0[have] = _ceil_div((st[have] - pre_ticks) - int(ref_tick), tps)
        i1[have] = np.floor_divide((en[have] + post_ticks) - int(ref_tick), tps)
        ok = have
    else:
        with np.errstate(invalid="ignore"):
            i0 = np.ceil((t0 - t_ref) * nsx_samplingrate)
            i1 = np.floor((t1 - t_ref) * nsx_samplingrate)
        i0 = np.where(ok, np.nan_to_num(i0, nan=0.0), 0.0).astype(np.int64)
        i1 = np.where(ok, np.nan_to_num(i1, nan=-1.0), -1.0).astype(np.int64)

        # i0 := smallest index whose time is >= t0
        sel = ok & (i0 >= 0) & (i0 < n_sample)
        adj = np.zeros(n_trials, dtype=bool)
        adj[sel] = nsx_abs_time[i0[sel]] < t0[sel]
        i0[adj] += 1
        sel = ok & (i0 >= 1) & (i0 <= n_sample)
        adj = np.zeros(n_trials, dtype=bool)
        adj[sel] = nsx_abs_time[i0[sel] - 1] >= t0[sel]
        i0[adj] -= 1

        # i1 := largest index whose time is <= t1
        sel = ok & (i1 >= 0) & (i1 < n_sample)
        adj = np.zeros(n_trials, dtype=bool)
        adj[sel] = nsx_abs_time[i1[sel]] > t1[sel]
        i1[adj] -= 1
        sel = ok & (i1 >= -1) & (i1 < n_sample - 1)
        adj = np.zeros(n_trials, dtype=bool)
        adj[sel] = nsx_abs_time[i1[sel] + 1] <= t1[sel]
        i1[adj] += 1

    # Clamp on one side each, so a window lying entirely outside the recording
    # ends up with i1 < i0 and is dropped rather than collapsing onto a bogus
    # single sample.
    i0 = np.maximum(i0, 0)
    i1 = np.minimum(i1, n_sample - 1)

    n = i1 - i0 + 1
    n[~ok] = 0          # missing marker -> all-NaN slice
    n[n < 0] = 0        # window entirely outside the recording
    rawstarttime = starts.copy()
    rawstarttime[n == 0] = np.nan   # abs time of the Start marker (s)

    # --- second pass: stack into NaN-padded 3-D array (left-aligned) ---
    max_samples = int(n.max()) if n.size else 0
    data = np.full((n_chan, n_trials, max_samples), np.nan, dtype=np.float32)
    for i in np.flatnonzero(n > 0):
        sl = nsxdata[:, i0[i] : i1[i] + 1].astype(np.float32)
        if uv_scale is not None:
            sl = sl * uv_scale.astype(np.float32)
        data[:, i, : n[i]] = sl

    # reltime: 0 at the Start marker, negative through the pre-buffer
    reltime = np.arange(max_samples, dtype=float) / nsx_samplingrate - pre

    info = {
        "samplingrate": float(nsx_samplingrate),
        "Session": np.array([tr.get("Session") for tr in trials], dtype=float),
        "Trial_number": np.array(
            [tr.get("Trial_number") for tr in trials], dtype=float
        ),
    }
    # MATLAB records no unit on the continuous products, so the samples are
    # ambiguous unless you know the stream's header out of band. This field is
    # the one deliberate addition to that schema.
    if unit is not None:
        info["Unit"] = unit

    return {
        "data": data,  # nChan x nTrials x maxSamples, NaN-padded, float32
        "timeseq": {
            "alignedrawtime": rawstarttime,  # nTrials, abs time of Start marker (s)
            "aligned_marker": "Start",
            "relative_time": reltime,  # maxSamples, seconds from aligned marker
        },
        "info": info,
    }
