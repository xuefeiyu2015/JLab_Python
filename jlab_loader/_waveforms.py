"""
Online-spike waveform trial segmentation.
Collects every in-window spike's waveform into one dense, NaN-padded 4-D array
(microVolts), mirroring BlackrockLoader.segmentSpikeWaveforms in the MATLAB
version. Rows match segment_spikes 1:1.
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


def segment_waveforms(
    trials: list[dict],
    spike_times: np.ndarray,
    spike_channel: np.ndarray,
    spike_unit: np.ndarray,
    spike_waveform: np.ndarray,
    pre_ms: float = SEGMENT_PRE_MS,
    post_ms: float = SEGMENT_POST_MS,
) -> dict:
    """
    Collect every in-window spike's waveform into a dense, NaN-padded 4-D array.

    Source-agnostic: operates purely on the ``spike_times/channel/unit`` +
    ``spike_waveform`` arrays, so it is the shared waveform "align to trials"
    stage for any spike source (online HUB NEV now, offline sorted spikes later).

    Port of BlackrockLoader.segmentSpikeWaveforms. For each trial the window is
    ``[Start - pre_ms, End + post_ms]`` (ms buffers), matched against
    ``spike_times`` (seconds, same clock as the event timestamps). Rows are one
    per (channel, unit) pair, sorted lexicographically — the SAME ordering as
    segment_spikes, so waveform rows line up 1:1 with the raster rows. Within a
    (row, trial), spikes are packed in time order into positions 1..k and the
    array is NaN-padded to ``maxSpikes`` (the busiest (unit, trial) count), so
    the result lines up 1:1 with ``trials``. A trial whose Start/End is NaN
    contributes no spikes (all-NaN slice), staying index-aligned with ``trials``.

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
    spike_waveform : np.ndarray
        Per-spike waveforms in microVolts, shape ``(nSpikes, nSamp)`` (row per
        spike, aligned 1:1 with spike_times/channel/unit).
    pre_ms, post_ms : float
        Buffer (ms) before Start / after End. Default 500 each.

    Returns
    -------
    dict mirroring the MATLAB ``W`` struct:
        "waveform"       : (NtotalUnit, nTrials, maxSpikes, nSamp) µV, NaN-padded
        "waveform_time"  : (NtotalUnit, nTrials, maxSpikes) s, relative to Start
        "waveform_nsamp" : int, samples per waveform
        "waveform_unit"  : "microVolts"
        "timeseq"        : {"alignedrawtime", "aligned_marker"}
        "info"           : {"Session", "Trial_number", "Channel_Number",
                            "Unit_No", "maxSpikes"}
    """
    pre = pre_ms / 1000.0
    post = post_ms / 1000.0

    spike_times = np.asarray(spike_times, dtype=float).ravel()
    spike_channel = np.asarray(spike_channel, dtype=float).ravel()
    spike_unit = np.asarray(spike_unit, dtype=float).ravel()
    spike_waveform = np.asarray(spike_waveform, dtype=float)
    if spike_waveform.ndim != 2:
        raise ValueError("spike_waveform must be 2-D (nSpikes, nSamp).")
    n_samp = spike_waveform.shape[1]

    n_trials = len(trials)

    # --- channel list: one row per (channel, unit), sorted (matches segment_spikes) ---
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

    # --- first pass: window per trial + busiest (unit, trial) spike count ---
    t_start = np.full(n_trials, np.nan)
    t_end = np.full(n_trials, np.nan)
    rawstarttime = np.full(n_trials, np.nan)
    max_spk = 0
    for i, tr in enumerate(trials):
        start, end = tr.get("Start"), tr.get("End")
        if _isnan(start) or _isnan(end):
            continue  # missing marker -> all-NaN slice
        t0 = start - pre
        t1 = end + post
        t_start[i] = t0
        t_end[i] = t1
        rawstarttime[i] = start  # abs time of the Start marker (s)
        sel = (spike_times >= t0) & (spike_times < t1)
        if np.any(sel):
            counts = np.bincount(spike_row[sel], minlength=n_chan)
            max_spk = max(max_spk, int(counts.max()))

    # --- allocate (warn first if the dense array is large) ---
    if n_chan * n_trials * max_spk * n_samp * 8 > 2e9:
        warnings.warn(
            "segment_waveforms: waveform array is "
            f"{n_chan * n_trials * max_spk * n_samp * 8 / 1e9:.1f} GB "
            f"({n_chan} units x {n_trials} trials x {max_spk} spikes x "
            f"{n_samp} samples). Consider narrowing the data "
            "(fewer/sorted units or trials).",
            stacklevel=2,
        )
    wf = np.full((n_chan, n_trials, max_spk, n_samp), np.nan)  # µV, NaN-padded
    wf_time = np.full((n_chan, n_trials, max_spk), np.nan)  # s, relative to Start

    # --- second pass: each in-window spike -> position 1..k per (row, trial) ---
    for i in range(n_trials):
        if _isnan(t_start[i]):
            continue
        sel = (spike_times >= t_start[i]) & (spike_times < t_end[i])
        if not np.any(sel):
            continue
        sel_idx = np.flatnonzero(sel)  # original spike indices, in time order
        rows = spike_row[sel_idx]
        cnt = np.zeros(n_chan, dtype=int)  # running per-row position within this trial
        for idx, r in zip(sel_idx, rows):
            wf[r, i, cnt[r], :] = spike_waveform[idx, :]
            wf_time[r, i, cnt[r]] = spike_times[idx] - rawstarttime[i]
            cnt[r] += 1

    return {
        "waveform": wf,  # NtotalUnit x nTrials x maxSpikes x nSamp, µV, NaN-padded
        "waveform_time": wf_time,  # NtotalUnit x nTrials x maxSpikes, s, rel to Start
        "waveform_nsamp": n_samp,  # samples per waveform
        "waveform_unit": "microVolts",
        "timeseq": {
            "alignedrawtime": rawstarttime,  # nTrials, abs time of Start marker (s)
            "aligned_marker": "Start",  # waveform_time = 0 at the Start marker
        },
        "info": {
            "Session": np.array([tr.get("Session") for tr in trials], dtype=float),
            "Trial_number": np.array(
                [tr.get("Trial_number") for tr in trials], dtype=float
            ),
            "Channel_Number": electrode,  # NtotalUnit, electrode per row
            "Unit_No": unit,  # NtotalUnit, unit per row
            "maxSpikes": max_spk,  # spike-dimension length (busiest unit-trial)
        },
    }
