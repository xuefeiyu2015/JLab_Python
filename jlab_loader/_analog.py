"""
Analog (NSx) trial segmentation.
Cuts the continuous analog stream into one slice per trial, mirroring
BlackrockLoader.segmentAnalog in the MATLAB version.
"""

from __future__ import annotations

import math

import numpy as np

from ._constants import SEGMENT_POST_MS, SEGMENT_PRE_MS


def _isnan(v) -> bool:
    try:
        return math.isnan(v)
    except (TypeError, ValueError):
        return False


def segment_analog(
    trials: list[dict],
    nsxdata: np.ndarray,
    nsx_abs_time: np.ndarray,
    nsx_samplingrate: float,
    pre_ms: float = SEGMENT_PRE_MS,
    post_ms: float = SEGMENT_POST_MS,
) -> dict:
    """
    Cut the continuous analog stream into one slice per trial.

    Port of BlackrockLoader.segmentAnalog. For each trial the window is
    ``[Start - pre_ms, End + post_ms]`` (ms buffers), matched against
    ``nsx_abs_time`` (seconds, same clock as the event timestamps). Slices are
    left-aligned (each starts at its own window start) and NaN-padded to the
    longest trial, so the result is one ``nChan × nTrials × maxSamples`` array
    that lines up 1:1 with ``trials``. A trial whose Start/End is NaN (or that
    has no samples in range) gets an all-NaN slice so the trial dimension stays
    index-aligned with ``trials``.

    Parameters
    ----------
    trials : list[dict]
        Parsed trials; each needs "Start" and "End" (seconds), plus "Session"
        and "Trial_number".
    nsxdata : np.ndarray
        Continuous analog data, shape ``(nChan, nSamples)`` (channels × samples).
    nsx_abs_time : np.ndarray
        Absolute time (seconds) of each analog sample, shape ``(nSamples,)``.
    nsx_samplingrate : float
        Analog sampling rate in Hz.
    pre_ms, post_ms : float
        Buffer (ms) before Start / after End. Default 500 each.

    Returns
    -------
    dict mirroring the MATLAB ``A`` struct:
        "data"     : (nChan, nTrials, maxSamples) float, NaN-padded
        "timeseq"  : {"alignedrawtime", "aligned_marker", "relative_time"}
        "info"     : {"samplingrate", "Session", "Trial_number"}
    """
    pre = pre_ms / 1000.0
    post = post_ms / 1000.0

    nsxdata = np.asarray(nsxdata)
    nsx_abs_time = np.asarray(nsx_abs_time, dtype=float).ravel()
    n_chan = nsxdata.shape[0]
    n_trials = len(trials)

    # --- first pass: find each trial's sample window ---
    idx: list[np.ndarray | None] = [None] * n_trials
    n = np.zeros(n_trials, dtype=int)
    rawstarttime = np.full(n_trials, np.nan)
    for i, tr in enumerate(trials):
        start, end = tr.get("Start"), tr.get("End")
        if _isnan(start) or _isnan(end):
            continue  # missing marker -> all-NaN slice
        t0 = start - pre
        t1 = end + post
        w = np.flatnonzero((nsx_abs_time >= t0) & (nsx_abs_time <= t1))
        if w.size == 0:
            continue
        idx[i] = w
        n[i] = w.size
        rawstarttime[i] = start  # abs time of the Start marker (s)

    # --- second pass: stack into NaN-padded 3-D array (left-aligned) ---
    max_samples = int(n.max()) if n.size else 0
    data = np.full((n_chan, n_trials, max_samples), np.nan)
    for i in range(n_trials):
        if n[i] > 0:
            data[:, i, : n[i]] = nsxdata[:, idx[i]]

    # reltime: 0 at the Start marker, negative through the pre-buffer
    reltime = np.arange(max_samples) / nsx_samplingrate - pre

    return {
        "data": data,  # nChan x nTrials x maxSamples, NaN-padded
        "timeseq": {
            "alignedrawtime": rawstarttime,  # nTrials, abs time of Start marker (s)
            "aligned_marker": "Start",
            "relative_time": reltime,  # maxSamples, seconds from aligned marker
        },
        "info": {
            "samplingrate": nsx_samplingrate,
            "Session": np.array([tr.get("Session") for tr in trials], dtype=float),
            "Trial_number": np.array(
                [tr.get("Trial_number") for tr in trials], dtype=float
            ),
        },
    }
