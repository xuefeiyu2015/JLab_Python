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

import math

import numpy as np

from ._constants import (
    SEGMENT_BIN_MS,
    SEGMENT_POST_MS,
    SEGMENT_PRE_MS,
    UNSORTED_UNIT_IDS,
)


def _isnan(v) -> bool:
    try:
        return math.isnan(v)
    except (TypeError, ValueError):
        return False


def drop_units(
    spike_times: np.ndarray,
    spike_channel: np.ndarray,
    spike_unit: np.ndarray,
    spike_waveform: np.ndarray | None = None,
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
    drop_ids : iterable
        Unit ids to remove. Empty -> drops nothing.

    Returns
    -------
    (spike_times, spike_channel, spike_unit, spike_waveform, n_dropped)
        Filtered arrays (``spike_waveform`` is ``None`` if it was ``None``) and
        the number of spikes removed.
    """
    spike_unit = np.asarray(spike_unit)
    keep = ~np.isin(spike_unit, np.asarray(drop_ids))
    n_dropped = int((~keep).sum())
    spike_times = np.asarray(spike_times)[keep]
    spike_channel = np.asarray(spike_channel)[keep]
    spike_unit = spike_unit[keep]
    if spike_waveform is not None:
        spike_waveform = np.asarray(spike_waveform)[keep]
    return spike_times, spike_channel, spike_unit, spike_waveform, n_dropped


def segment_spikes(
    trials: list[dict],
    spike_times: np.ndarray,
    spike_channel: np.ndarray,
    spike_unit: np.ndarray,
    pre_ms: float = SEGMENT_PRE_MS,
    post_ms: float = SEGMENT_POST_MS,
    bin_ms: float = SEGMENT_BIN_MS,
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
    falls in it, 0 otherwise. Slices are left-aligned (bin 0 at the window
    start) and NaN-padded to the longest trial, giving one
    ``NtotalUnit × nTrials × maxBins`` array that lines up 1:1 with ``trials``
    (same layout as segment_analog). Rows are one per (channel, unit) pair, so
    NtotalUnit is the total isolated units summed across channels.
    A trial whose Start/End is NaN gets an all-NaN slice.

    Parameters
    ----------
    trials : list[dict]
        Parsed trials; each needs "Start" and "End" (seconds), plus "Session"
        and "Trial_number".
    spike_times : np.ndarray
        Spike timestamps in seconds.
    spike_channel : np.ndarray
        Electrode/channel id per spike.
    spike_unit : np.ndarray
        Unit id per spike.
    pre_ms, post_ms : float
        Buffer (ms) before Start / after End. Default 500 each.
    bin_ms : float
        Raster bin width (ms). Default 1.

    Returns
    -------
    dict mirroring the MATLAB ``R`` struct:
        "data"     : (NtotalUnit, nTrials, maxBins) float, 0/1, NaN-padded
        "timeseq"  : {"alignedrawtime", "aligned_marker", "relative_time"}
        "info"     : {"samplingrate", "Session", "Trial_number",
                      "Channel_Number", "Unit_No"}
    """
    pre = pre_ms / 1000.0
    post = post_ms / 1000.0
    bin_sec = bin_ms / 1000.0

    spike_times = np.asarray(spike_times, dtype=float).ravel()
    spike_channel = np.asarray(spike_channel, dtype=float).ravel()
    spike_unit = np.asarray(spike_unit, dtype=float).ravel()

    n_trials = len(trials)

    # --- channel list: one row per (channel, unit), sorted ---
    # np.unique(..., axis=0) sorts rows lexicographically (col0 then col1),
    # matching MATLAB unique([electrode, unit], 'rows'); inverse maps each spike
    # to its channel row.
    if spike_times.size:
        keys = np.column_stack([spike_channel, spike_unit])
        chan_keys, spike_row = np.unique(keys, axis=0, return_inverse=True)
        spike_row = spike_row.ravel()
    else:
        chan_keys = np.empty((0, 2))
        spike_row = np.empty(0, dtype=int)
    electrode = chan_keys[:, 0]
    unit = chan_keys[:, 1]
    n_chan = chan_keys.shape[0]

    # --- first pass: find each trial's bin count and window ---
    n_bins = np.zeros(n_trials, dtype=int)
    t_start = np.full(n_trials, np.nan)
    t_end = np.full(n_trials, np.nan)
    rawstarttime = np.full(n_trials, np.nan)
    for i, tr in enumerate(trials):
        start, end = tr.get("Start"), tr.get("End")
        if _isnan(start) or _isnan(end):
            continue  # missing marker -> all-NaN slice
        t0 = start - pre
        t1 = end + post
        n_bins[i] = max(int(round((t1 - t0) / bin_sec)), 0)
        t_start[i] = t0
        t_end[i] = t1
        rawstarttime[i] = start  # abs time of the Start marker (s)

    # --- second pass: fill NaN-padded binary raster (left-aligned) ---
    max_bins = int(n_bins.max()) if n_bins.size else 0
    raster = np.full((n_chan, n_trials, max_bins), np.nan)
    for i in range(n_trials):
        if n_bins[i] <= 0:
            continue
        t0 = t_start[i]
        t1 = t_end[i]
        raster[:, i, : n_bins[i]] = 0.0  # within-window bins start at 0
        sel = (spike_times >= t0) & (spike_times < t1)
        if not np.any(sel):
            continue
        bins = np.floor((spike_times[sel] - t0) / bin_sec).astype(int)
        bins = np.minimum(bins, n_bins[i] - 1)  # guard the right edge
        rows = spike_row[sel]
        raster[rows, i, bins] = 1.0  # binary: spike present (clamped)

    # reltime: 0 at the Start marker, negative through the pre-buffer
    reltime = np.arange(max_bins) * bin_sec - pre

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
        },
    }
