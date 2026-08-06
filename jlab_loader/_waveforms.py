"""
Online-spike waveform trial segmentation.
Collects every in-window spike's waveform into one dense, NaN-padded 4-D array
(microVolts), mirroring BlackrockLoader.segmentSpikeWaveforms in the MATLAB
version. Rows match segment_spikes 1:1.
"""

from __future__ import annotations

import warnings

import numpy as np

from ._constants import SEGMENT_POST_MS, SEGMENT_PRE_MS
from ._continuous import _trial_bounds
from ._spikes import channel_keys, trial_spike_index


def _relative_times(
    sorted_times: np.ndarray,
    spk_idx: np.ndarray,
    trial_of: np.ndarray,
    rawstarttime: np.ndarray,
    sorted_ticks: np.ndarray | None,
    start_ticks: np.ndarray | None,
    time_res: float | None,
) -> np.ndarray:
    """Each spike's time relative to its trial's Start marker, in seconds.

    Both clocks are absolute epoch values (~1.5e18 ns), so in seconds a float64
    resolves only ~238 ns and a straight subtraction inherits that from BOTH
    operands. When the raw integer ticks are available, subtract them first
    (exact in uint64) and divide after — the result then lands near zero, where
    a float64 resolves ~0.004 ns. Port of BlackrockLoader.m:2043-2069.
    """
    fallback = sorted_times[spk_idx] - rawstarttime[trial_of]
    if sorted_ticks is None or start_ticks is None or not time_res:
        return fallback
    if sorted_ticks.size != sorted_times.size or start_ticks.size != rawstarttime.size:
        return fallback

    spk_tick = sorted_ticks[spk_idx].astype(np.uint64)
    ref_tick = start_ticks[trial_of].astype(np.uint64)
    # uint64 cannot go negative; split so pre-Start spikes keep their sign
    # instead of saturating at zero.
    after = spk_tick >= ref_tick
    dt = np.zeros(spk_idx.size, dtype=float)
    dt[after] = (spk_tick[after] - ref_tick[after]).astype(float) / time_res
    dt[~after] = -((ref_tick[~after] - spk_tick[~after]).astype(float) / time_res)
    # trials with no recorded Start tick fall back to seconds
    no_ref = ref_tick == 0
    if np.any(no_ref):
        dt[no_ref] = fallback[no_ref]
    return dt


def segment_waveforms(
    trials: list[dict],
    spike_times: np.ndarray,
    spike_channel: np.ndarray,
    spike_unit: np.ndarray,
    spike_waveform: np.ndarray,
    pre_ms: float = SEGMENT_PRE_MS,
    post_ms: float = SEGMENT_POST_MS,
    spike_ticks: np.ndarray | None = None,
    start_ticks: np.ndarray | None = None,
    time_res: float | None = None,
) -> dict:
    """
    Collect every in-window spike's waveform into a dense, NaN-padded 4-D array.

    Source-agnostic: operates purely on the ``spike_times/channel/unit`` +
    ``spike_waveform`` arrays, so it is the shared waveform "align to trials"
    stage for any spike source (online HUB NEV now, offline sorted spikes later).

    Port of BlackrockLoader.segmentSpikeWaveforms. For each trial the window is
    ``[Start - pre_ms, End + post_ms]`` (ms buffers), matched against
    ``spike_times`` (seconds). Rows are one per (channel, unit) pair, sorted
    lexicographically — the SAME ordering as segment_spikes, so waveform rows
    line up 1:1 with the raster rows. Within a (row, trial), spikes are packed
    in time order into positions 0..k-1 and the array is NaN-padded to
    ``maxSpikes`` (the busiest (unit, trial) count). A trial whose Start/End is
    NaN contributes no spikes, staying index-aligned with ``trials``.

    Parameters
    ----------
    trials : list[dict]
        Parsed trials; each needs "Start" and "End" (seconds), plus "Session"
        and "Trial_number".
    spike_times : np.ndarray
        Spike timestamps in seconds.
    spike_channel, spike_unit : np.ndarray
        Electrode/channel and unit id per spike.
    spike_waveform : np.ndarray
        Per-spike waveforms in microVolts, shape ``(nSpikes, nSamp)``, aligned
        1:1 with spike_times/channel/unit.
    pre_ms, post_ms : float
        Buffer (ms) before Start / after End. Default 500 each.
    spike_ticks : np.ndarray or None
        Per-spike raw uint64 clock ticks. With ``start_ticks`` and ``time_res``
        these give exact-integer relative times (see _relative_times).
    start_ticks : np.ndarray or None
        Per-trial raw uint64 tick of the Start marker (0 where unknown).
    time_res : float or None
        Ticks per second for both tick arrays.

    Returns
    -------
    dict mirroring the MATLAB ``W`` struct:
        "waveform"       : (NtotalUnit, nTrials, maxSpikes, nSamp) float32 µV
        "waveform_time"  : (NtotalUnit, nTrials, maxSpikes) float64 s, rel to Start
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
    spike_waveform = np.asarray(spike_waveform)
    if spike_waveform.ndim != 2:
        raise ValueError("spike_waveform must be 2-D (nSpikes, nSamp).")
    n_samp = spike_waveform.shape[1]

    n_trials = len(trials)

    chan_keys, spike_row = channel_keys(spike_channel, spike_unit)
    electrode = chan_keys[:, 0]
    unit = chan_keys[:, 1]
    n_chan = chan_keys.shape[0]

    # --- trial windows ---
    starts, ends, ok = _trial_bounds(trials)
    t_start = np.where(ok, starts - pre, np.nan)
    t_end = np.where(ok, ends + post, np.nan)
    rawstarttime = np.where(ok, starts, np.nan)

    # --- every (trial, in-window spike) pair, and each spike's position within
    # its (row, trial) group ---
    order = np.argsort(spike_times, kind="stable")
    sorted_times = spike_times[order]
    spk_idx, trial_of = trial_spike_index(sorted_times, t_start, t_end)
    orig_idx = order[spk_idx]  # index into the unsorted spike arrays
    row_of = spike_row[orig_idx]

    n_pair = spk_idx.size
    pos = np.zeros(n_pair, dtype=np.int64)
    if n_pair:
        # Rank within each (row, trial) group: sort by group (stable, so time
        # order is kept inside a group), then a counter that resets at every
        # group boundary. Port of BlackrockLoader.m:2013-2022.
        g = trial_of * n_chan + row_of
        gord = np.argsort(g, kind="stable")
        gs = g[gord]
        is_new = np.empty(n_pair, dtype=bool)
        is_new[0] = True
        np.not_equal(gs[1:], gs[:-1], out=is_new[1:])
        run = np.arange(n_pair, dtype=np.int64)
        pos[gord] = run - np.maximum.accumulate(np.where(is_new, run, -1))
    max_spk = int(pos.max()) + 1 if n_pair else 0

    # --- allocate (warn first if the dense array is large) ---
    n_bytes = n_chan * n_trials * max_spk * n_samp * 4  # float32
    if n_bytes > 2e9:
        warnings.warn(
            "segment_waveforms: waveform array is "
            f"{n_bytes / 1e9:.1f} GB "
            f"({n_chan} units x {n_trials} trials x {max_spk} spikes x "
            f"{n_samp} samples). Consider narrowing the data "
            "(fewer/sorted units or trials).",
            stacklevel=2,
        )
    wf = np.full((n_chan, n_trials, max_spk, n_samp), np.nan, dtype=np.float32)
    # Times stay float64. Only voltages go float32: at a few seconds from the
    # marker a float32 resolves ~0.5 µs, which would throw away real resolution
    # on a 30 kHz (let alone nanosecond PTP) spike clock.
    wf_time = np.full((n_chan, n_trials, max_spk), np.nan)

    # --- fill: two indexed writes instead of a scalar write per spike ---
    if n_pair:
        sorted_ticks = None
        if spike_ticks is not None:
            spike_ticks = np.asarray(spike_ticks)
            if spike_ticks.size == spike_times.size:
                sorted_ticks = spike_ticks[order]
        start_arr = None if start_ticks is None else np.asarray(start_ticks)
        wf_time[row_of, trial_of, pos] = _relative_times(
            sorted_times, spk_idx, trial_of, rawstarttime,
            sorted_ticks, start_arr, time_res,
        )
        wf[row_of, trial_of, pos, :] = spike_waveform[orig_idx, :].astype(np.float32)

    return {
        "waveform": wf,  # NtotalUnit x nTrials x maxSpikes x nSamp, µV, NaN-padded
        "waveform_time": wf_time,  # NtotalUnit x nTrials x maxSpikes, s, rel to Start
        # Scalars are float: MATLAB has no integer type here, so a Python int
        # would land in the .mat as int64 where MATLAB wrote double.
        "waveform_nsamp": float(n_samp),  # samples per waveform
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
            "maxSpikes": float(max_spk),  # spike-dimension length (busiest unit-trial)
        },
    }
