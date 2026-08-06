"""
Spike trial segmentation (source-agnostic).
Rasterizes spikes into one binary slice per trial, mirroring
BlackrockLoader.segmentSpikes in the MATLAB version. ``segment_spikes`` and the
shared ``drop_units`` helper take only loose arrays
(``spike_times/channel/unit``), so they work for any spike source — online
(HUB NEV) today, offline (sorted) later — differing only in the reader that
produces those arrays.
"""

from __future__ import annotations

import numpy as np

from ._constants import (
    SEGMENT_BIN_MS,
    SEGMENT_POST_MS,
    SEGMENT_PRE_MS,
    SPIKE_ISI_VIOLATION_MS,
    UNSORTED_UNIT_IDS,
)
from ._continuous import _trial_bounds, exact_tick_grid


def drop_units(
    spike_times: np.ndarray,
    spike_channel: np.ndarray,
    spike_unit: np.ndarray,
    spike_waveform: np.ndarray | None = None,
    spike_ticks: np.ndarray | None = None,
    *,
    drop_ids=UNSORTED_UNIT_IDS,
):
    """Drop spikes whose unit id is in ``drop_ids`` (default: unsorted 0 + noise 255).

    Source-agnostic filter shared by every spike reader: it applies one keep-mask
    ``~np.isin(spike_unit, drop_ids)`` across the parallel per-spike arrays so
    they stay index-aligned. A future offline reader can reuse this (passing its
    own ``drop_ids`` if its "unsorted/noise" convention differs).

    Parameters
    ----------
    spike_times, spike_channel, spike_unit : np.ndarray
        Parallel per-spike arrays (one entry per spike).
    spike_waveform : np.ndarray or None
        Optional (nSpikes, nSamp) per-spike waveform array, filtered along axis 0.
        Stays ``None`` when not provided.
    spike_ticks : np.ndarray or None
        Optional per-spike raw uint64 clock ticks, filtered alongside so the
        exact-tick path in segment_waveforms stays aligned.
    drop_ids : iterable
        Unit ids to remove. Empty -> drops nothing.

    Returns
    -------
    (spike_times, spike_channel, spike_unit, spike_waveform, spike_ticks, n_dropped)
    """
    spike_unit = np.asarray(spike_unit)
    keep = ~np.isin(spike_unit, np.asarray(drop_ids))
    n_dropped = int((~keep).sum())
    spike_times = np.asarray(spike_times)[keep]
    spike_channel = np.asarray(spike_channel)[keep]
    spike_unit = spike_unit[keep]
    if spike_waveform is not None:
        spike_waveform = np.asarray(spike_waveform)[keep]
    if spike_ticks is not None:
        spike_ticks = np.asarray(spike_ticks)[keep]
    return spike_times, spike_channel, spike_unit, spike_waveform, spike_ticks, n_dropped


def channel_keys(spike_channel: np.ndarray, spike_unit: np.ndarray):
    """One row per (channel, unit) pair, sorted, plus each spike's row index.

    ``np.unique(..., axis=0)`` sorts rows lexicographically (col0 then col1),
    matching MATLAB ``unique([electrode, unit], 'rows')``. Shared by
    segment_spikes and segment_waveforms so their rows line up 1:1.
    """
    if spike_channel.size:
        keys = np.column_stack([spike_channel, spike_unit])
        chan_keys, spike_row = np.unique(keys, axis=0, return_inverse=True)
        spike_row = spike_row.ravel().astype(np.int64)
    else:
        chan_keys = np.empty((0, 2))
        spike_row = np.empty(0, dtype=np.int64)
    return chan_keys, spike_row


def trial_spike_index(
    sorted_times: np.ndarray,
    t_start: np.ndarray,
    t_end: np.ndarray,
    ok: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Every (trial, in-window spike) pair, via two binary searches per trial.

    Port of BlackrockLoader.trialSpikeIndex. The window is half-open
    ``[t_start, t_end)``. Windows may overlap between adjacent trials when the
    buffers exceed the ITI, and a spike is deliberately allowed to belong to two
    trials.

    Works in whatever domain it is handed: float seconds, or exact integer clock
    ticks. Pass ``ok`` explicitly for the integer case, where there is no NaN to
    mark a trial that has no window; otherwise it is derived from NaN bounds.

    Returns ``(spk_idx, trial_of)`` — indices into ``sorted_times`` and the trial
    each pair belongs to.
    """
    if ok is None:
        ok = ~(np.isnan(t_start) | np.isnan(t_end))
    lo = np.zeros(t_start.size, dtype=np.int64)
    hi = np.zeros(t_start.size, dtype=np.int64)
    if sorted_times.size and np.any(ok):
        lo[ok] = np.searchsorted(sorted_times, t_start[ok], side="left")
        hi[ok] = np.searchsorted(sorted_times, t_end[ok], side="left")
    counts = np.maximum(hi - lo, 0)
    counts[~ok] = 0
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    trial_of = np.repeat(np.arange(t_start.size, dtype=np.int64), counts)
    # Offsets 0..counts[i]-1 within each run, without a Python loop.
    run_start = np.repeat(np.cumsum(counts) - counts, counts)
    offset = np.arange(total, dtype=np.int64) - run_start
    spk_idx = np.repeat(lo, counts) + offset
    return spk_idx, trial_of


def _violation_rate(
    spike_row: np.ndarray, spike_times: np.ndarray, n_chan: int, viol_sec: float
) -> np.ndarray:
    """Fraction of each unit's ISIs shorter than ``viol_sec``, over its FULL train.

    Port of the accumarray block in BlackrockLoader.segmentSpikes. A pure timing
    metric, independent of the raster and of the waveform product. NaN for a unit
    with fewer than 2 spikes. Sorting by (unit, time) once makes every unit's
    train a contiguous ascending run.
    """
    rate = np.full(n_chan, np.nan)
    if spike_row.size < 2:
        return rate
    order = np.lexsort((spike_times, spike_row))  # primary: row, secondary: time
    unit_of = spike_row[order]
    isi = np.diff(spike_times[order])
    same_unit = np.diff(unit_of) == 0  # drop the gaps between units
    if not np.any(same_unit):
        return rate
    owner = unit_of[:-1][same_unit]  # the first element of each within-unit pair
    hits = (isi[same_unit] < viol_sec).astype(float)
    counts = np.bincount(owner, minlength=n_chan)
    sums = np.bincount(owner, weights=hits, minlength=n_chan)
    nz = counts > 0
    rate[nz] = sums[nz] / counts[nz]
    return rate


def _mean_waveform(
    spike_row: np.ndarray, spike_waveform: np.ndarray, n_chan: int
) -> np.ndarray:
    """Each unit's average waveform (µV) over its FULL set of spikes, all trials.

    Port of the meanWf block in BlackrockLoader.segmentSpikes. NaN row for a unit
    with no waveforms. Computed in float64 over float32 inputs, matching MATLAB's
    ``mean(double(single_block), 2, 'omitnan')``.
    """
    n_samp = spike_waveform.shape[1]
    mean_wf = np.full((n_chan, n_samp), np.nan)
    if spike_row.size == 0:
        return mean_wf
    order = np.argsort(spike_row, kind="stable")
    rows_sorted = spike_row[order]
    counts = np.bincount(rows_sorted, minlength=n_chan)
    stop = np.cumsum(counts)
    start = stop - counts
    for r in np.flatnonzero(counts > 0):
        block = spike_waveform[order[start[r] : stop[r]], :].astype(np.float64)
        with np.errstate(invalid="ignore"):
            mean_wf[r, :] = np.nanmean(block, axis=0)
    return mean_wf


def segment_spikes(
    trials: list[dict],
    spike_times: np.ndarray,
    spike_channel: np.ndarray,
    spike_unit: np.ndarray,
    pre_ms: float = SEGMENT_PRE_MS,
    post_ms: float = SEGMENT_POST_MS,
    bin_ms: float = SEGMENT_BIN_MS,
    viol_ms: float = SPIKE_ISI_VIOLATION_MS,
    spike_waveform: np.ndarray | None = None,
    *,
    spike_ticks=None,
    start_ticks=None,
    end_ticks=None,
    time_res: float | None = None,
) -> dict:
    """
    Rasterize spikes into one binary slice per trial.

    Source-agnostic: operates purely on the ``spike_times/channel/unit`` arrays,
    so it is the shared "align to trials" stage for any spike source (online HUB
    NEV now, offline sorted spikes later).

    Port of BlackrockLoader.segmentSpikes. For each trial the window is
    ``[Start - pre_ms, End + post_ms]`` (ms buffers), matched against
    ``spike_times`` (seconds, same clock as the event timestamps). Time is
    binned at ``bin_ms`` (default 1 ms); a bin is 1 if any spike of that row
    falls in it, 0 otherwise — binary, so two spikes in one bin are
    indistinguishable from one. Slices are left-aligned (bin 0 at the window
    start) and NaN-padded to the longest trial, giving one
    ``NtotalUnit × nTrials × maxBins`` array that lines up 1:1 with ``trials``
    (same layout as segment_continuous). Rows are one per (channel, unit) pair.
    A trial whose Start/End is NaN gets an all-NaN slice.

    Parameters
    ----------
    trials : list[dict]
        Parsed trials; each needs "Start" and "End" (seconds), plus "Session"
        and "Trial_number".
    spike_times : np.ndarray
        Spike timestamps in seconds.
    spike_channel, spike_unit : np.ndarray
        Electrode/channel and unit id per spike.
    pre_ms, post_ms : float
        Buffer (ms) before Start / after End. Default 500 each.
    bin_ms : float
        Raster bin width (ms). Default 1.
    viol_ms : float
        Refractory window (ms) for info.ViolationRate. Default 1.
    spike_waveform : np.ndarray or None
        Optional (nSpikes, nSamp) µV waveforms aligned 1:1 with the spike
        arrays; supplies info.MeanWaveform. None -> MeanWaveform is empty.

    Returns
    -------
    dict mirroring the MATLAB ``R`` struct:
        "data"     : (NtotalUnit, nTrials, maxBins) float32, 0/1, NaN-padded
        "timeseq"  : {"alignedrawtime", "aligned_marker", "relative_time"}
        "info"     : {"samplingrate", "Session", "Trial_number",
                      "Channel_Number", "Unit_No", "ViolationRate",
                      "MeanWaveform", "MeanWaveformUnit"}
    """
    pre = pre_ms / 1000.0
    post = post_ms / 1000.0
    bin_sec = bin_ms / 1000.0
    viol_sec = viol_ms / 1000.0

    spike_times = np.asarray(spike_times, dtype=float).ravel()
    spike_channel = np.asarray(spike_channel, dtype=float).ravel()
    spike_unit = np.asarray(spike_unit, dtype=float).ravel()

    n_trials = len(trials)

    chan_keys, spike_row = channel_keys(spike_channel, spike_unit)
    electrode = chan_keys[:, 0]
    unit = chan_keys[:, 1]
    n_chan = chan_keys.shape[0]

    # --- overall QC, both over each unit's FULL train (not per trial) ---
    viol_rate = _violation_rate(spike_row, spike_times, n_chan, viol_sec)
    if spike_waveform is not None and len(spike_waveform):
        mean_wf = _mean_waveform(spike_row, np.asarray(spike_waveform), n_chan)
    else:
        mean_wf = np.zeros((0, 0))  # MATLAB []

    # --- first pass: each trial's bin count and window ---
    starts, ends, ok = _trial_bounds(trials)
    t_start = np.where(ok, starts - pre, np.nan)
    t_end = np.where(ok, ends + post, np.nan)
    n_bins = np.zeros(n_trials, dtype=np.int64)
    with np.errstate(invalid="ignore"):
        n_bins[ok] = np.maximum(
            np.round((t_end[ok] - t_start[ok]) / bin_sec).astype(np.int64), 0
        )
    rawstarttime = np.where(ok, starts, np.nan)

    # --- second pass: fill NaN-padded binary raster (left-aligned) ---
    max_bins = int(n_bins.max()) if n_bins.size else 0
    raster = np.full((n_chan, n_trials, max_bins), np.nan, dtype=np.float32)
    for i in np.flatnonzero(n_bins > 0):
        raster[:, i, : n_bins[i]] = 0.0  # within-window bins start at 0

    # Every (trial, in-window spike) pair at once, then a single indexed write.
    # Both membership and the bin index are done in exact integer ticks when
    # they are available: in seconds these are absolute epoch values (~1.5e9 s)
    # where a double resolves only ~238 ns, so a spike within that band of a bin
    # edge would land on either side depending on the last bit.
    grid = None
    if spike_ticks is not None and start_ticks is not None and end_ticks is not None:
        sp_t = np.asarray(spike_ticks, dtype=np.int64).ravel()
        st_t = np.asarray(start_ticks, dtype=np.int64).ravel()
        en_t = np.asarray(end_ticks, dtype=np.int64).ravel()
        if (
            sp_t.size == spike_times.size
            and st_t.size == n_trials
            and en_t.size == n_trials
        ):
            grid = exact_tick_grid(time_res, 1.0 / bin_sec, pre_ms, post_ms)

    if grid is not None:
        bin_ticks, pre_ticks, post_ticks = grid
        order = np.argsort(sp_t, kind="stable")
        sorted_ticks = sp_t[order]
        # A trial needs both markers AND both raw ticks to use the exact path.
        ok_t = ok & (st_t != 0) & (en_t != 0)
        lo = np.where(ok_t, st_t - pre_ticks, 0)
        hi = np.where(ok_t, en_t + post_ticks, 0)
        spk_idx, trial_of = trial_spike_index(sorted_ticks, lo, hi, ok_t)
        if spk_idx.size:
            bins = np.floor_divide(sorted_ticks[spk_idx] - lo[trial_of], bin_ticks)
    else:
        order = np.argsort(spike_times, kind="stable")
        sorted_times = spike_times[order]
        spk_idx, trial_of = trial_spike_index(sorted_times, t_start, t_end)
        if spk_idx.size:
            bins = np.floor(
                (sorted_times[spk_idx] - t_start[trial_of]) / bin_sec
            ).astype(np.int64)

    if spk_idx.size:
        bins = np.minimum(bins, n_bins[trial_of] - 1)  # guard the right edge
        bins = np.maximum(bins, 0)
        rows = spike_row[order[spk_idx]]
        raster[rows, trial_of, bins] = 1.0

    # reltime: 0 at the Start marker, negative through the pre-buffer
    reltime = np.arange(max_bins, dtype=float) * bin_sec - pre

    return {
        "data": raster,  # NtotalUnit x nTrials x maxBins, 0/1, NaN-padded
        "timeseq": {
            "alignedrawtime": rawstarttime,  # nTrials, abs time of Start marker (s)
            "aligned_marker": "Start",
            "relative_time": reltime,  # maxBins, seconds from aligned marker
        },
        "info": {
            "samplingrate": 1.0 / bin_sec,  # bin rate (Hz), 1000 for 1 ms bins
            "Session": np.array([tr.get("Session") for tr in trials], dtype=float),
            "Trial_number": np.array(
                [tr.get("Trial_number") for tr in trials], dtype=float
            ),
            "Channel_Number": electrode,  # NtotalUnit, electrode per row
            "Unit_No": unit,  # NtotalUnit, unit per row
            "ViolationRate": viol_rate,  # NtotalUnit, frac ISIs < viol_ms (full train)
            "MeanWaveform": mean_wf,  # NtotalUnit x nSamp (µV), or empty
            "MeanWaveformUnit": "microVolts",
        },
    }
