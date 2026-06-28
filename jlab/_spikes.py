"""
Online-spike trial segmentation.
Rasterizes online spikes into one binary slice per trial, mirroring
BlackrockLoader.segmentSpikes in the MATLAB version.
"""

from __future__ import annotations

import math

import numpy as np

from ._constants import SEGMENT_BIN_MS, SEGMENT_POST_MS, SEGMENT_PRE_MS


def _isnan(v) -> bool:
    try:
        return math.isnan(v)
    except (TypeError, ValueError):
        return False


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
    Rasterize online spikes into one binary slice per trial.

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
