"""
Converts a list of (comment_text, time_s) pairs into structured dicts
that match the MATLAB BackRockFileLoader.m output exactly.
"""

from __future__ import annotations

import math
import warnings

from ._constants import (
    DASH_EVENTS,
    EXP_EVENTS,
    INFORMATION_EVENTS,
    OUTCOME_EVENTS,
    POLAR_EVENTS,
    RE_COORD,
    RE_EXP,
    RE_EXP_PAUSE,
    RE_OFFSET_ACTIVE,
    RE_OFFSET_RANGE,
    RE_PD_COORD,
    RE_POLAR,
    RE_REWARD,
    RE_SIZE_DEG,
    RE_TIME_MS,
    RE_TRIAL,
    SEGMENT_EVENTS,
    TIME_EVENTS,
)

NaN = float("nan")

# Trial fields whose raw clock tick is kept alongside the seconds value. The
# segmenters use these to do window arithmetic in exact integers instead of on
# absolute epoch doubles, where the resolution is only ~238 ns.
TICK_FIELDS = ("Start", "End")


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
        # A session can pause and resume any number of times, so these are
        # growing lists, not scalars, and [] rather than NaN so an un-paused
        # session exports as '[]'. Not necessarily the same length: a session
        # paused and then ended without resuming has one more pause than resume.
        "pause_times": [],
        "resume_times": [],
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
        "Fixation_color": "",  # name or an RGB triplet as sent
        "Requested_fixation_hold_time": NaN,
        "Requested_fixation_duration": NaN,
        "Requested_timeout": NaN,
        "Requested_time_between_trials": NaN,
        "Target_1_position": [NaN, NaN],
        "Target_1_size": NaN,
        "Target_1_acceptance_window": NaN,
        "Target_1_color": "",  # name or an RGB triplet as sent
        "Requested_target_1_hold_time": NaN,
        "Requested_target_1_timeout": NaN,
        "Requested_target_1_duration": NaN,
        "Target_2_position": [NaN, NaN],
        "Target_2_size": NaN,
        "Target_2_acceptance_window": NaN,
        "Target_2_color": "",  # name or an RGB triplet as sent
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
        "Target_1_dimmed": NaN,
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
        # Jitter the task applies to the target's base polar position, and the
        # grid it quantises that jitter to (deg). 09-10-2026
        "Requested_jitter_theta": NaN,
        "Requested_jitter_theta_step": NaN,
        "Requested_jitter_rho": NaN,
        "Requested_jitter_rho_step": NaN,
        # Target polar position in the task's own convention: 0 = +x (right),
        # counter-clockwise, [0, 360). Written by a 'position polar' comment when
        # the task sends one, otherwise back-computed from the cartesian
        # position by compute_derived_features. Use _features.hemifield for
        # left/right -- a sign test does not work in this range.
        "Target_1_theta": NaN,
        "Target_2_theta": NaN,
        "Target_1_rho": NaN,
        "Target_2_rho": NaN,
        # The pre-jitter base position, same convention. Never back-computed.
        "Target_1_base_theta": NaN,
        "Target_2_base_theta": NaN,
        "Target_1_base_rho": NaN,
        "Target_2_base_rho": NaN,
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
    if marker in ("paused", "resumed"):
        # Timing only -- the marker's own timestamp, appended in file order.
        # These lines carry no metadata token, so they skip the key chain.
        exp["pause_times" if marker == "paused" else "resume_times"].append(time_s)
        return

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
# Port of BlackrockLoader.classifyEventBodies. A body is first assigned ONE kind
# by which map's key it contains (see _classify); the handler for that kind then
# parses it, and a body it cannot parse goes to trials.undefined -- it never
# falls through to a lower-priority kind.

def _match_index(name: str, keys: list[str]) -> str | None:
    """Key for an extracted event name: exact match where one exists, else the
    sole key that contains the name. None when neither is unique.

    Exact-first is what lets overlapping keys such as 'Requested jitter theta'
    and 'Requested jitter theta step' coexist. Port of BlackrockLoader.matchIndex.
    """
    if name in keys:
        return name
    cand = [k for k in keys if name in k]
    return cand[0] if len(cand) == 1 else None


def _write_first(trial: dict, field: str, value, dup_label: str) -> None:
    """First write wins; later writes are recorded as duplicates."""
    if _field_is_empty(trial, field):
        trial[field] = value
    else:
        trial["duplicates"].append(dup_label)


def _handle_time_event(text: str, time_s: float, trial: dict) -> None:
    # Exact match on the whole body; containing a key is only the kind test.
    field = TIME_EVENTS.get(text)
    if field is None:
        trial["undefined"].append(text)
        return
    _write_first(trial, field, time_s, text)  # duplicate label = the body


def _handle_info_event(text: str, time_s: float, trial: dict) -> None:
    keys = list(INFORMATION_EVENTS)

    # Coordinate pattern: "Name (x, y) deg" -> a 2-element field
    m = RE_COORD.match(text)
    if m:
        name = m.group(1).strip()
        key = _match_index(name, keys)
        if key is None:
            trial["undefined"].append(text)
            return
        _write_first(trial, INFORMATION_EVENTS[key],
                     [float(m.group(2)), float(m.group(3))], name)
        return

    # Reward pattern: "Name(NNms ...)" -> the field takes the TIMESTAMP, and
    # the amount lands in Reward_amount, each guarded separately.
    m = RE_REWARD.match(text)
    if m:
        key = _match_index(m.group(1).strip(), keys)
        if key is None:
            trial["undefined"].append(text)
            return
        field = INFORMATION_EVENTS[key]
        _write_first(trial, field, time_s, field)
        _write_first(trial, "Reward_amount", float(m.group(2)), "Reward_amount")
        return

    # Scalar: "<name> <value> ms" else "<name> <value> [deg]"
    m = RE_TIME_MS.match(text) or RE_SIZE_DEG.match(text)
    key = _match_index(m.group(1).strip(), keys) if m else None
    if key is None:
        trial["undefined"].append(text)
        return
    raw = m.group(2)
    value = NaN if raw.lower() == "none" else float(raw)  # "None ms" -> NaN
    field = INFORMATION_EVENTS[key]
    _write_first(trial, field, value, field)


def _handle_segment_event(text: str, trial: dict) -> None:
    # The value is everything after the KEY, verbatim. Splitting at the last
    # space dropped every value containing one, e.g. the bracketed RGB form
    # 'Target 1 color [-0.8, -0.8, -0.8]'. validate_event_maps guarantees no
    # key is a substring of another, so at most one key can match.
    for key in sorted(SEGMENT_EVENTS):
        if key not in text:
            continue
        value = text.split(key, 1)[1].strip()
        if not value:
            trial["undefined"].append(text)  # bare key carrying no value
            return
        field = SEGMENT_EVENTS[key]
        if trial[field] is None or trial[field] == "":
            trial[field] = value
        else:
            trial["duplicates"].append(field)
        return
    trial["undefined"].append(text)


def _handle_polar_event(text: str, trial: dict) -> None:
    # '<name> (theta <t>, rho <r>) deg' -> two scalar fields. Exact name match
    # only: every polar body names its key in full.
    m = RE_POLAR.match(text)
    pair = POLAR_EVENTS.get(m.group(1).strip()) if m else None
    if pair is None:
        trial["undefined"].append(text)
        return
    theta_field, rho_field = pair
    _write_first(trial, theta_field, float(m.group(2)), theta_field)
    _write_first(trial, rho_field, float(m.group(3)), rho_field)


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
        if trial["Trialoutcome"] is None:
            trial["Trialoutcome"] = outcome
        else:
            trial["duplicates"].append("Trialoutcome")
        return True

    if "choice" in event.lower():
        if trial["Choosen_choice"] is None:
            trial["Choosen_choice"] = outcome
        else:
            trial["duplicates"].append("Choosen_choice")
        return True

    # Unknown dash event
    trial["undefined"].append(text)
    return True


def _handle_outcome_event(text: str, trial: dict) -> bool:
    if not any(o in text for o in OUTCOME_EVENTS):
        return False
    if trial["Choiceoutcome"] is None:
        trial["Choiceoutcome"] = text
    else:
        trial["duplicates"].append("Choiceoutcome")
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


def _classify(text: str) -> int:
    """Event kind of one trial body, in MATLAB's priority order.

    Assigned lowest priority first so a higher one overwrites, exactly as
    classifyEventBodies does. Polar is last (highest): 'Target 1 position polar'
    contains the information key 'Target 1 position'.
    """
    kind = 0
    if "Requested time offset range" in text:
        kind = 6
    if any(o in text for o in OUTCOME_EVENTS):
        kind = 5
    if "-" in text and any(d in text for d in DASH_EVENTS):
        kind = 4
    if any(k in text for k in SEGMENT_EVENTS):
        kind = 3
    if any(k in text for k in INFORMATION_EVENTS):
        kind = 2
    if any(k in text for k in TIME_EVENTS):
        kind = 1
    if any(k in text for k in POLAR_EVENTS):
        kind = 7
    return kind


def _dispatch_trial_event(text: str, time_s: float, trial: dict) -> None:
    kind = _classify(text)
    if kind == 7:
        _handle_polar_event(text, trial)
    elif kind == 1:
        _handle_time_event(text, time_s, trial)
    elif kind == 2:
        _handle_info_event(text, time_s, trial)
    elif kind == 3:
        _handle_segment_event(text, trial)
    elif kind == 4:
        _handle_dash_event(text, time_s, trial)
    elif kind == 5:
        _handle_outcome_event(text, trial)
    elif kind == 6:
        _handle_offset_range_event(text, trial)
    else:
        trial["undefined"].append(text)  # claimed by nothing


# ── Event-map validation ───────────────────────────────────────────────────

def validate_event_maps() -> None:
    """Fail loudly on an event map that would mis-bind or silently drop events.

    Port of BlackrockLoader.validateEventMaps.
      * No SEGMENT_EVENTS key may be a substring of another: that map is the only
        one matched in key order, so overlapping keys would bind by sort order.
      * Every mapped field must exist in the template it writes to. Otherwise
        the comment is parsed and then vanishes -- not into the trial, not into
        trials.undefined, not into the parse report.
    """
    keys = list(SEGMENT_EVENTS)
    for a in keys:
        for b in keys:
            if a != b and a in b:
                raise ValueError(
                    f'SEGMENT_EVENTS: key "{a}" is a substring of "{b}". That map is '
                    "matched in key order and cannot disambiguate these."
                )

    trial_fields = set(_make_empty_trial())
    exp_fields = set(_make_empty_experiment())
    checks = [
        ("TIME_EVENTS", TIME_EVENTS.values(), trial_fields),
        ("INFORMATION_EVENTS", INFORMATION_EVENTS.values(), trial_fields),
        ("SEGMENT_EVENTS", SEGMENT_EVENTS.values(), trial_fields),
        ("POLAR_EVENTS", [f for pair in POLAR_EVENTS.values() for f in pair], trial_fields),
        ("EXP_EVENTS", EXP_EVENTS.values(), exp_fields),
    ]
    for name, fields, template in checks:
        missing = [f for f in fields if f not in template]
        if missing:
            raise ValueError(
                f"{name} maps to field(s) that do not exist in the template: "
                f"{', '.join(missing)}. Add them to the trial/experiment template, "
                "or the matching comments would be parsed and then silently dropped."
            )


# ── Parse report ───────────────────────────────────────────────────────────

def _format_trial_list(labels: list[tuple[int, int]], max_show: int = 8) -> str:
    """'[S1T12, S1T40, +3 more]' from (session, trial_number) pairs."""
    if not labels:
        return "[before the first trial]"
    shown = ", ".join(f"S{s}T{t}" for s, t in labels[:max_show])
    more = f", +{len(labels) - max_show} more" if len(labels) > max_show else ""
    return f"[{shown}{more}]"


def _group_labels(pairs: list[tuple[str, int]], trials: list[dict]) -> list[tuple[str, int, str]]:
    """(label, count, trial list) per DISTINCT label, sorted by label.

    pairs holds (label, trial row) in occurrence order; rows < 0 (before the
    first trial) are counted but not listed.
    """
    rows_of: dict[str, list[int]] = {}
    count: dict[str, int] = {}
    for label, row in pairs:
        count[label] = count.get(label, 0) + 1
        rows = rows_of.setdefault(label, [])
        if row >= 0 and row not in rows:
            rows.append(row)
    out = []
    for label in sorted(count):
        tl = [(int(trials[r]["Session"]), int(trials[r]["Trial_number"])) for r in rows_of[label]]
        out.append((label, count[label], _format_trial_list(tl)))
    return out


def compute_parse_report(trials: list[dict], parse_info: dict) -> dict:
    """Summary of what the parser could not place. Pure.

    For undefined events, duplicates and malformed comments: each distinct
    label with how often it occurred and the trials it came from, so the
    offending comment can actually be found in the file.
    """
    undefined = [(lab, r) for r, t in enumerate(trials) for lab in t["undefined"]]
    duplicates = [(lab, r) for r, t in enumerate(trials) for lab in t["duplicates"]]
    malformed = parse_info["malformed"]
    sessions = [t["Session"] for t in trials]
    return {
        "n_comments": parse_info["n_comments"],
        "n_trials": len(trials),
        "n_sessions": int(max(sessions + [0])),
        "n_bodies": parse_info["n_bodies"],
        "n_undefined": len(undefined),
        "n_duplicates": len(duplicates),
        "n_malformed": len(malformed),
        "undefined": _group_labels(undefined, trials),
        "duplicates": _group_labels(duplicates, trials),
        "malformed": _group_labels(malformed, trials),
    }


def print_parse_report(report: dict) -> None:
    """Render a compute_parse_report() result, in MATLAB's parseEvents format.

    Always printed, deliberately not behind a verbose flag: the undefined list
    is the only place a comment shape the parser does not know ever surfaces.
    """
    print(
        f"parseEvents: {report['n_comments']} comments -> {report['n_trials']} trials, "
        f"{report['n_sessions']} session(s); {report['n_bodies']} distinct event bodies"
    )
    print(
        f"  undefined {report['n_undefined']}, duplicates {report['n_duplicates']}, "
        f"malformed {report['n_malformed']}"
    )
    blocks = [
        ("undefined events (add a key for these, or they stay dropped)", report["undefined"]),
        ("duplicated events (first write kept)", report["duplicates"]),
        ('malformed comments (neither "Experiment" nor "Trial N:", skipped; '
         "shown against the trial that was open)", report["malformed"]),
    ]
    for heading, rows in blocks:
        if not rows:
            continue
        print(f"  {heading}:")
        for label, cnt, tl in rows:
            print(f"    x{cnt:<4d} {label:<46s} {tl}")
    if report["malformed"]:
        # Warned as well as listed: this is the one class dropped outright.
        warnings.warn(
            f"{report['n_malformed']} comment(s) matched neither the Experiment nor "
            f'the "Trial N:" form and were skipped ({len(report["malformed"])} '
            "distinct, listed above).",
            stacklevel=2,
        )


# ── Session labelling ──────────────────────────────────────────────────────

def _session_labels_from_resets(
    tnum_per_trial: list[int], sess_git_per_trial: list[int]
) -> list[int]:
    """
    Label sessions from trial-number resets, cross-checked against the
    "Experiment start: git commit" markers. Port of
    BlackrockLoader.sessionLabelsFromResets.

    The markers alone are not trustworthy: a file that was not saved cleanly can
    be missing its leading metadata block, and the old marker-count rule then
    stamped every trial of that block Session = 0 — which matters, because
    Session is a join key downstream. Counting resets always yields 1-based
    labels covering every trial.

    '<= 0' rather than '< 0' because the trial-row rule keeps the git-session
    term: two adjacent trial rows can share a trial number only when a marker
    split them, which is itself a session change.
    """
    n = len(tnum_per_trial)
    if n == 0:
        return []

    new_reset = [True] + [
        tnum_per_trial[i] - tnum_per_trial[i - 1] <= 0 for i in range(1, n)
    ]
    session, acc = [], 0
    for flag in new_reset:
        acc += int(flag)
        session.append(acc)

    if max(sess_git_per_trial) == 0:
        warnings.warn(
            'No "Experiment start: git commit" marker precedes any trial; '
            f"session labels derive from trial-number resets alone "
            f"({session[-1]} session(s)).",
            stacklevel=3,
        )
        return session

    new_git = [True] + [
        sess_git_per_trial[i] != sess_git_per_trial[i - 1] for i in range(1, n)
    ]
    if new_reset != new_git:
        d = next(i for i in range(n) if new_reset[i] != new_git[i])
        warnings.warn(
            f"Session partition from trial-number resets ({session[-1]} session(s)) "
            f"disagrees with the git-commit markers ({sum(new_git)} session(s)); "
            f"first disagreement at trial row {d} (Trial_number "
            f"{tnum_per_trial[d]}). Using the reset-derived labels.",
            stacklevel=3,
        )
    return session


def _reindex_experiments(
    exp_by_marker: list[dict],
    marker_next_row: list[int | None],
    session: list[int],
) -> list[dict]:
    """
    Re-key metadata blocks from "one per git-commit marker" to "one per Session
    label". Port of the tail of BlackrockLoader.buildExperimentMeta.

    These coincide for a clean file; they diverge when a file is missing its
    leading metadata block, which is exactly the case the reset-derived labels
    exist to handle. Sessions with no metadata keep a blank template entry so
    experiments[Session - 1] is always addressable.
    """
    n_sess = max(session) if session else 0

    sess_of_marker: list[int] = []
    for row in marker_next_row:
        if row is None:
            # Trailing marker with no trials after it; give it its own session
            # rather than dropping the metadata.
            n_sess += 1
            sess_of_marker.append(n_sess)
        else:
            sess_of_marker.append(session[row])

    experiments = [_make_empty_experiment() for _ in range(n_sess)]
    filled = [False] * n_sess
    for m, s in enumerate(sess_of_marker):
        if not filled[s - 1]:
            experiments[s - 1] = exp_by_marker[m]
            filled[s - 1] = True
        else:
            # Two markers inside one reset-session: keep the opening block and
            # take the closing one's end/end_by. Only reachable when
            # _session_labels_from_resets already warned.
            experiments[s - 1]["end"] = exp_by_marker[m]["end"]
            experiments[s - 1]["end_by"] = exp_by_marker[m]["end_by"]
            # Pause/resume are concatenated rather than replaced -- both
            # blocks' markers happened inside this one session.
            experiments[s - 1]["pause_times"] += exp_by_marker[m]["pause_times"]
            experiments[s - 1]["resume_times"] += exp_by_marker[m]["resume_times"]
    return experiments


# ── Top-level parser ───────────────────────────────────────────────────────

def parse_comments(
    comments: list[str],
    times_s: list[float],
    ticks: list[int] | None = None,
) -> tuple[list[dict], list[dict], dict[str, list[int]], dict]:
    """
    Parse a flat list of NEV event comments into per-session experiment metadata
    and trials.

    Trials are keyed by POSITION, not by number: a new trial row opens whenever
    the parsed trial number differs from the previous trial comment's, or the
    git-marker session changed. `Session` is then derived from trial-number
    resets (see _session_labels_from_resets), which is 1-based and covers every
    trial even when the recording is missing its leading metadata block.

    Parameters
    ----------
    comments : list[str]
        Event comment strings from brpylib (data['comments']['Data']).
    times_s : list[float]
        Timestamps in seconds, one per comment.
    ticks : list[int] or None
        Raw integer clock ticks, one per comment. Used to record each trial's
        Start-marker tick for the exact-tick waveform timing path.

    Returns
    -------
    experiments : list[dict]
        Experiment-level metadata, one dict per Session label.
    trials : list[dict]
        One dict per trial, in order of first appearance.
    trial_ticks : dict[str, list[int]]
        Raw clock tick of each trial's Start and End markers, keyed by field
        name; 0 where the trial never got that marker. Feeds the exact-integer
        window arithmetic in the segmenters.
    parse_info : dict
        Inputs for compute_parse_report: ``n_comments``, ``n_bodies`` (distinct
        trial-event bodies) and ``malformed`` -- (text, open trial row) for each
        comment that is neither an Experiment nor a "Trial N:" line; the row is
        -1 before the first trial.
    """
    if ticks is None:
        ticks = [0] * len(comments)

    # ── Pass 1: classify every comment; build trial rows and git sessions ───
    # sess_git counts "Experiment start: git commit" markers, 0 before the
    # first, and is used ONLY to split trials that share a number across a
    # session boundary. The exported Session comes from resets, below.
    trials: list[dict] = []
    trial_ticks: dict[str, list[int]] = {k: [] for k in TICK_FIELDS}
    tnum_per_trial: list[int] = []
    sess_git_per_trial: list[int] = []

    exp_by_marker: list[dict] = []
    marker_next_row: list[int | None] = []  # first trial row after each marker
    sess_git = 0
    prev_trial_num: int | None = None
    prev_sess_git: int | None = None
    malformed: list[tuple[str, int]] = []
    bodies: set[str] = set()

    for text, time_s, tick in zip(comments, times_s, ticks):
        # NUL padding would defeat every match below (strip() keeps NUL).
        text = text.replace("\x00", " ").strip()

        # ── Experiment-level event ─────────────────────────────────────────
        m = RE_EXP.match(text) or RE_EXP_PAUSE.match(text)
        if m:
            marker, payload = m.group(1), m.group(2).strip()
            if marker == "start" and payload.startswith("git commit"):
                sess_git += 1
                exp_by_marker.append(_make_empty_experiment())
                exp_by_marker[-1]["start"] = time_s
                marker_next_row.append(None)  # resolved when the next trial opens
            if exp_by_marker:
                # Metadata before the first git-commit start has nowhere to go.
                _handle_exp_event(marker, payload, time_s, exp_by_marker[-1])
            continue

        # ── Trial-level event ──────────────────────────────────────────────
        m = RE_TRIAL.match(text)
        if not m:
            # Neither an Experiment nor a "Trial N:" line (empty lines included):
            # there is no trial to attach it to, so it is reported and skipped.
            malformed.append((text, len(trials) - 1))
            continue

        trial_num = int(m.group(1))
        event_text = m.group(2).strip()
        bodies.add(event_text)

        if not trials or trial_num != prev_trial_num or sess_git != prev_sess_git:
            t = _make_empty_trial()
            t["Trial_number"] = trial_num
            trials.append(t)
            for _k in TICK_FIELDS:
                trial_ticks[_k].append(0)
            tnum_per_trial.append(trial_num)
            sess_git_per_trial.append(sess_git)
            # Any marker still waiting for a following trial resolves to this row.
            row = len(trials) - 1
            for k in range(len(marker_next_row)):
                if marker_next_row[k] is None:
                    marker_next_row[k] = row
        prev_trial_num = trial_num
        prev_sess_git = sess_git

        trial = trials[-1]

        # First write wins for Start and End, and that is the write whose raw
        # tick we keep: the window arithmetic is done in exact integers off
        # these, because a subtraction in seconds at ~1.5e9 s would inherit
        # ~238 ns of rounding from each operand.
        was_nan = {k: _is_nan(trial[k]) for k in TICK_FIELDS}
        _dispatch_trial_event(event_text, time_s, trial)
        for k in TICK_FIELDS:
            if was_nan[k] and not _is_nan(trial[k]):
                trial_ticks[k][-1] = int(tick)

        # Update Save_complete after each event
        if not _is_nan(trial["Start"]) and trial["Start"] > 0:
            trial["Save_complete"] = 1

    # ── Pass 2: Session labels from resets, then re-key the metadata ────────
    session = _session_labels_from_resets(tnum_per_trial, sess_git_per_trial)
    for t, s in zip(trials, session):
        t["Session"] = s

    experiments = _reindex_experiments(exp_by_marker, marker_next_row, session)

    parse_info = {
        "n_comments": len(comments),
        "n_bodies": len(bodies),
        "malformed": malformed,
    }
    return experiments, trials, trial_ticks, parse_info
