"""
Converts a list of (comment_text, time_s) pairs into structured dicts
that match the MATLAB BackRockFileLoader.m output exactly.
"""

from __future__ import annotations

import math

from ._constants import (
    DASH_EVENTS,
    EXP_EVENTS,
    INFORMATION_EVENTS,
    OUTCOME_EVENTS,
    RE_COORD,
    RE_EXP,
    RE_REWARD,
    RE_SIZE_DEG,
    RE_TIME_MS,
    RE_TRIAL,
    SEGMENT_EVENTS,
    TIME_EVENTS,
)

NaN = float("nan")


def _make_empty_experiment() -> dict:
    return {
        "git_commit": None,
        "viewing_distance": NaN,
        "screen_size": [NaN, NaN],
        "screen_resolution": [NaN, NaN],
        "FPS": NaN,
        "eyetracker_rate": NaN,
        "eye_tracked": None,
        "start": NaN,
        "end": NaN,
    }


def _make_empty_trial() -> dict:
    return {
        "Trial_number": NaN,
        "Task": None,
        "Trial_type": None,
        "Start": NaN,
        "Fixation_position": [NaN, NaN],
        "Fixation_size": NaN,
        "Fixation_acceptance_window": NaN,
        "Fixation_color": None,
        "Requested_fixation_hold_time": NaN,
        "Requested_fixation_duration": NaN,
        "Requested_timeout": NaN,
        "Requested_time_between_trials": NaN,
        "Target_1_position": [NaN, NaN],
        "Target_1_size": NaN,
        "Target_1_acceptance_window": NaN,
        "Target_1_color": None,
        "Requested_target_1_hold_time": NaN,
        "Requested_target_1_timeout": NaN,
        "Requested_target_1_duration": NaN,
        "Requested_target_2_time_offset": NaN,
        "Requested_target_2_hold_time": NaN,
        "Requested_penalty_box_duration": NaN,
        "Requested_target_dim_opacity": NaN,
        "Requested_target_1_visible_duration": NaN,
        "Target_2_position": [NaN, NaN],
        "Target_2_size": NaN,
        "Target_2_acceptance_window": NaN,
        "Target_2_color": None,
        "Fixation_point_on": NaN,
        "Fixation_acquired": NaN,
        "Fixation_point_off": NaN,
        "Broke_fixation": NaN,
        "Target_1_presented": NaN,
        "Target_2_presented": NaN,
        "Targets_off": NaN,
        "Target_1_off": NaN,
        "Choiceoutcome": None,
        "Choosen_choice": None,
        "Choicetime": NaN,
        "End": NaN,
        "Trialoutcome": None,
        "Reward_start": NaN,
        "Reward_amount": NaN,
        "Reward_end": NaN,
        "Save_complete": 0,
        "undefined": [],
        "duplicates": [],
    }


def _is_nan(v) -> bool:
    try:
        return math.isnan(v)
    except (TypeError, ValueError):
        return False


def _field_is_empty(trial: dict, field: str) -> bool:
    val = trial[field]
    if val is None:
        return True
    if isinstance(val, list):
        return all(_is_nan(x) for x in val)
    return _is_nan(val)


# ── Experiment event handler ───────────────────────────────────────────────

def _handle_exp_event(text: str, time_s: float, exp: dict) -> None:
    m = RE_EXP.match(text)
    if not m:
        return
    marker, payload = m.group(1), m.group(2).strip()

    # Record start/end timestamp. A recording can contain multiple experiment
    # runs (e.g. an aborted start, then the real session). Keep the EARLIEST
    # start and the LATEST end so the session spans the whole recording instead
    # of collapsing onto the first (possibly aborted) run.
    if marker == "start":
        if _is_nan(exp["start"]):
            exp["start"] = time_s
    else:  # marker == "end"
        exp["end"] = time_s

    # Parse payload into metadata fields
    if payload.startswith("git commit"):
        exp["git_commit"] = payload[len("git commit"):].strip()
        return

    if payload.startswith("eyetracker tracking"):
        word = payload[len("eyetracker tracking"):].strip().split()[0]
        exp["eye_tracked"] = word
        return

    # Numeric: one or two numbers after the key name
    num_m = _match_exp_key(payload)
    if num_m:
        field, nums = num_m
        exp[field] = nums if len(nums) > 1 else nums[0]


def _match_exp_key(payload: str):
    """Try to match payload against known EXP_EVENTS keys; return (field, [nums]) or None."""
    import re as _re
    m = _re.match(r"^\s*(.*?)\s+(\d+\.?\d*)\D*(\d+\.?\d*)?", payload)
    if not m:
        return None
    key_text = m.group(1).strip()
    nums = [float(g) for g in m.groups()[1:] if g is not None]
    field = next((v for k, v in EXP_EVENTS.items() if k in key_text or key_text in k), None)
    if field is None:
        return None
    return field, nums


# ── Trial event handlers ───────────────────────────────────────────────────

def _handle_time_event(text: str, time_s: float, trial: dict) -> bool:
    field = next((v for k, v in TIME_EVENTS.items() if k in text), None)
    if field is None:
        return False
    if _is_nan(trial[field]):
        trial[field] = time_s
    else:
        trial["duplicates"].append(text)
        print(f"Duplicate found for time event: {text}")
    return True


def _handle_info_event(text: str, time_s: float, trial: dict) -> bool:
    field_key = next((k for k in INFORMATION_EVENTS if k in text), None)
    if field_key is None:
        return False
    field = INFORMATION_EVENTS[field_key]

    # Try coordinate pattern: "Name (x, y) deg"
    m = RE_COORD.match(text)
    if m:
        coord = [float(m.group(2)), float(m.group(3))]
        if _field_is_empty(trial, field):
            trial[field] = coord
        else:
            trial["duplicates"].append(field_key)
            print(f"Duplicate found for coord event: {field_key}")
        return True

    # Try reward pattern: "Name(NNms ...)"
    m = RE_REWARD.match(text)
    if m:
        amount = float(m.group(2))
        if _is_nan(trial[field]):
            trial[field] = time_s  # save timestamp as reward start
        else:
            trial["duplicates"].append(field)
            print(f"Duplicate found for reward event: {field_key}")
        if _is_nan(trial["Reward_amount"]):
            trial["Reward_amount"] = amount
        else:
            trial["duplicates"].append("Reward_amount")
            print("Duplicate found for reward amount")
        return True

    # Try duration (ms) or size (deg)
    m = RE_TIME_MS.match(text) or RE_SIZE_DEG.match(text)
    if m:
        dur = float(m.group(2))
        if _is_nan(trial[field]):
            trial[field] = dur
        else:
            trial["duplicates"].append(field)
            print(f"Duplicate found for duration/size event: {field_key}")
        return True

    return True  # matched key but no pattern — skip silently


def _handle_segment_event(text: str, trial: dict) -> bool:
    field_key = next((k for k in SEGMENT_EVENTS if k in text), None)
    if field_key is None:
        return False
    field = SEGMENT_EVENTS[field_key]
    # Split on last space to get value
    event_name, _, value = text.rpartition(" ")
    if trial[field] is None:
        trial[field] = value.strip()
    else:
        trial["duplicates"].append(field)
        print(f"Duplicate found for segment event: {field}")
    return True


def _handle_dash_event(text: str, time_s: float, trial: dict) -> bool:
    if "-" not in text:
        return False
    if not any(d in text for d in DASH_EVENTS):
        return False
    dash_idx = text.index("-")
    event = text[:dash_idx].strip()
    outcome = text[dash_idx + 1:].strip()

    if "End" in event:
        if _is_nan(trial["End"]):
            trial["End"] = time_s
        else:
            trial["duplicates"].append(event)
            print(f"Duplicate End found for dash event: {event}")
        if trial["Trialoutcome"] is None:
            trial["Trialoutcome"] = outcome
        else:
            trial["duplicates"].append("Trialoutcome")
            print(f"Duplicate outcome found for dash event: {event}")
        return True

    if "choice" in event.lower():
        if trial["Choosen_choice"] is None:
            trial["Choosen_choice"] = outcome
        else:
            trial["duplicates"].append("Choosen_choice")
            print(f"Duplicate found for dash event: {event}")
        return True

    # Unknown dash event
    print(f"Undefined dash event: {text}")
    trial["undefined"].append(text)
    return True


def _handle_outcome_event(text: str, trial: dict) -> bool:
    if not any(o in text for o in OUTCOME_EVENTS):
        return False
    if trial["Choiceoutcome"] is None:
        trial["Choiceoutcome"] = text
    else:
        trial["duplicates"].append("Choiceoutcome")
        print(f"Duplicate found for outcome event: {text}")
    return True


def _dispatch_trial_event(text: str, time_s: float, trial: dict) -> None:
    if _handle_time_event(text, time_s, trial):
        return
    if _handle_info_event(text, time_s, trial):
        return
    if _handle_segment_event(text, trial):
        return
    if _handle_dash_event(text, time_s, trial):
        return
    if _handle_outcome_event(text, trial):
        return
    # Truly undefined
    print(f"Undefined event detected: {text}")
    trial["undefined"].append(text)


# ── Top-level parser ───────────────────────────────────────────────────────

def parse_comments(
    comments: list[str],
    times_s: list[float],
) -> tuple[dict, list[dict]]:
    """
    Parse a flat list of NEV event comments into experiment metadata and trials.

    Parameters
    ----------
    comments : list[str]
        Event comment strings from brpylib (data['comments']['Data']).
    times_s : list[float]
        Timestamps in seconds, one per comment.

    Returns
    -------
    experiment : dict
        Experiment-level metadata fields.
    trials : list[dict]
        One dict per trial, in order of first appearance.
    """
    experiment = _make_empty_experiment()
    trials: list[dict] = []
    prev_trial_num: int | None = None

    for text, time_s in zip(comments, times_s):
        text = text.strip()
        if not text:
            continue

        # ── Experiment-level event ─────────────────────────────────────────
        if RE_EXP.match(text):
            _handle_exp_event(text, time_s, experiment)
            continue

        # ── Trial-level event ──────────────────────────────────────────────
        m = RE_TRIAL.match(text)
        if not m:
            continue  # skip unrecognised top-level lines

        trial_num = int(m.group(1))
        event_text = m.group(2).strip()

        # A new trial begins whenever the parsed trial number changes from the
        # previous trial event. Trials are keyed by POSITION, not by number, so
        # a reset/non-monotonic counter (e.g. ...30, 0, 1...) starts a new trial
        # instead of merging into an earlier same-numbered trial.
        if not trials or trial_num != prev_trial_num:
            t = _make_empty_trial()
            t["Trial_number"] = trial_num
            trials.append(t)
        prev_trial_num = trial_num

        trial = trials[-1]

        _dispatch_trial_event(event_text, time_s, trial)

        # Update Save_complete after each event
        if not _is_nan(trial["Start"]) and trial["Start"] > 0:
            trial["Save_complete"] = 1

    return experiment, trials
