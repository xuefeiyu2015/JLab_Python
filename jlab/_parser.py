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
    RE_OFFSET_ACTIVE,
    RE_OFFSET_RANGE,
    RE_PD_COORD,
    RE_REWARD,
    RE_SIZE_DEG,
    RE_TIME_MS,
    RE_TRIAL,
    SEGMENT_EVENTS,
    TIME_EVENTS,
)

NaN = float("nan")


def _make_empty_experiment() -> dict:
    # Field order mirrors the MATLAB exp_template (BackRockFileLoader.m).
    # One of these dicts is created per experiment session within a recording.
    return {
        "git_commit": None,
        "viewing_distance": NaN,
        "screen_size": [NaN, NaN],
        "screen_resolution": [NaN, NaN],
        "FPS": NaN,
        "eyetracker_rate": NaN,
        "eye_tracked": None,
        "photodiode_circles": None,
        "photodiode_fixation_position": [NaN, NaN],
        "photodiode_target_1_position": [NaN, NaN],
        "photodiode_target_2_position": [NaN, NaN],
        "start": NaN,
        "end": NaN,
        "end_by": None,
    }


def _make_empty_trial() -> dict:
    # Field order mirrors the MATLAB trial struct (BackRockFileLoader.m); the CSV
    # column order is derived from this, so keep it in sync with MATLAB.
    return {
        "Trial_number": NaN,
        "Session": NaN,
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
        "Target_2_position": [NaN, NaN],
        "Target_2_size": NaN,
        "Target_2_acceptance_window": NaN,
        "Target_2_color": None,
        "Requested_target_2_time_offset": NaN,
        "Requested_target_2_hold_time": NaN,
        "Requested_penalty_box_duration": NaN,
        "Requested_target_dim_opacity": NaN,
        "Requested_target_1_visible_duration": NaN,
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
        # Newly added trial events (06-17-2026)
        "Feedback_flash_on": NaN,
        "Feedback_flash_off": NaN,
        "Fixation_exited": NaN,
        "Target_deadline_exceeded": NaN,
        "Requested_feedback_flash_duration": NaN,
        "Requested_choice_timeout": NaN,
        "Requested_target_reach_deadline": NaN,
        "Target_1_side": None,
        "Requested_time_offset_min": NaN,
        "Requested_time_offset_max": NaN,
        "Requested_time_offset_active": None,
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

def _handle_exp_event(marker: str, payload: str, time_s: float, exp: dict) -> None:
    """
    Update a single session's metadata dict from one "Experiment start/end:" line.

    The session's `start` timestamp is stamped when the session is created (see
    parse_comments); here we record the latest `end` (last end line wins), the
    reason the session ended (`end_by`), and any metadata fields. Mirrors
    BackRockFileLoader.m lines 295-339.
    """
    if marker == "end":
        exp["end"] = time_s  # last end line wins
        if not payload.startswith("git commit"):
            # Capture why the session ended (e.g. 'experimenter closed task').
            # The git-commit end line is just a commit re-stamp, not a reason.
            exp["end_by"] = payload

    # Parse payload into metadata fields
    if payload.startswith("git commit"):
        exp["git_commit"] = payload[len("git commit"):].strip()
        return

    if payload.startswith("eyetracker tracking"):
        word = payload[len("eyetracker tracking"):].strip().split()[0]
        exp["eye_tracked"] = word
        return

    if payload.startswith("photodiode"):
        # 'photodiode circles visible/hidden' or a '(x, y)' deg position.
        coord = RE_PD_COORD.search(payload)
        if payload.startswith("photodiode circles"):
            exp["photodiode_circles"] = payload[len("photodiode circles"):].strip()
        elif payload.startswith("photodiode fixation position") and coord:
            exp["photodiode_fixation_position"] = [float(coord.group(1)), float(coord.group(2))]
        elif payload.startswith("photodiode target_1 position") and coord:
            exp["photodiode_target_1_position"] = [float(coord.group(1)), float(coord.group(2))]
        elif payload.startswith("photodiode target_2 position") and coord:
            exp["photodiode_target_2_position"] = [float(coord.group(1)), float(coord.group(2))]
        return

    if payload == "experimenter closed task":
        # end-marker text, nothing to store
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
        raw = m.group(2)
        dur = NaN if raw.lower() == "none" else float(raw)  # "None ms" → NaN
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


def _handle_offset_range_event(text: str, trial: dict) -> bool:
    """
    Parse "Requested time offset range [min, max] ms (active: [v1, v2, ...])".
    Range → two numeric fields; active list → a space-separated string (CSV
    columns cannot hold multiple values). Mirrors BackRockFileLoader.m 579-592.
    """
    if "Requested time offset range" not in text:
        return False
    rng = RE_OFFSET_RANGE.search(text)
    if rng:
        trial["Requested_time_offset_min"] = float(rng.group(1))
        trial["Requested_time_offset_max"] = float(rng.group(2))
    active = RE_OFFSET_ACTIVE.search(text)
    if active:
        vals = [v.strip() for v in active.group(1).split(",") if v.strip()]
        trial["Requested_time_offset_active"] = " ".join(vals)
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
    if _handle_offset_range_event(text, trial):
        return
    # Truly undefined
    print(f"Undefined event detected: {text}")
    trial["undefined"].append(text)


# ── Top-level parser ───────────────────────────────────────────────────────

def parse_comments(
    comments: list[str],
    times_s: list[float],
) -> tuple[list[dict], list[dict]]:
    """
    Parse a flat list of NEV event comments into per-session experiment metadata
    and trials.

    A single recording can contain several experiment sessions (the task is
    started/ended multiple times). Each "Experiment start: git commit ..." line
    begins a new session, so `experiments` is a list with one dict per session,
    and every trial records which session it belongs to (`Session`, 1-based; 0
    if the trial precedes any session start).

    Parameters
    ----------
    comments : list[str]
        Event comment strings from brpylib (data['comments']['Data']).
    times_s : list[float]
        Timestamps in seconds, one per comment.

    Returns
    -------
    experiments : list[dict]
        Experiment-level metadata, one dict per session.
    trials : list[dict]
        One dict per trial, in order of first appearance.
    """
    experiments: list[dict] = []
    trials: list[dict] = []
    session_index = 0  # 0 = no session started yet
    prev_trial_num: int | None = None
    prev_session: int | None = None

    for text, time_s in zip(comments, times_s):
        text = text.strip()
        if not text:
            continue

        # ── Experiment-level event ─────────────────────────────────────────
        m = RE_EXP.match(text)
        if m:
            marker, payload = m.group(1), m.group(2).strip()
            # A new session begins at each "Experiment start: git commit ..." line
            # (the first line of every metadata block within the recording).
            if marker == "start" and payload.startswith("git commit"):
                session_index += 1
                experiments.append(_make_empty_experiment())
                experiments[-1]["start"] = time_s
            if session_index >= 1:
                _handle_exp_event(marker, payload, time_s, experiments[session_index - 1])
            continue

        # ── Trial-level event ──────────────────────────────────────────────
        m = RE_TRIAL.match(text)
        if not m:
            continue  # skip unrecognised top-level lines

        trial_num = int(m.group(1))
        event_text = m.group(2).strip()

        # A new trial begins whenever the parsed trial number OR the session
        # changes. Trials are keyed by POSITION, not by number, so a reset/
        # non-monotonic counter (e.g. ...30, 0, 1...) starts a new trial instead
        # of merging into an earlier same-numbered trial; the session test also
        # splits trials that share a number across two sessions.
        if not trials or trial_num != prev_trial_num or session_index != prev_session:
            t = _make_empty_trial()
            t["Trial_number"] = trial_num
            t["Session"] = session_index
            trials.append(t)
        prev_trial_num = trial_num
        prev_session = session_index

        trial = trials[-1]

        _dispatch_trial_event(event_text, time_s, trial)

        # Update Save_complete after each event
        if not _is_nan(trial["Start"]) and trial["Start"] > 0:
            trial["Save_complete"] = 1

    return experiments, trials
