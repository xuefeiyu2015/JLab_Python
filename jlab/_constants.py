# Parameter map for jives comments
# last updates: 06-18-2026 by Xuefei Yu
import re

# ── Experiment-level event map ─────────────────────────────────────────────
# Maps text tokens in "Experiment start/end:" lines → experiment struct fields
EXP_EVENTS: dict[str, str] = {
    "git commit": "git_commit",
    "viewing distance": "viewing_distance",
    "screen size": "screen_size",
    "screen resolution": "screen_resolution",
    "FPS": "FPS",
    "eyetracker sample rate": "eyetracker_rate",
    "eyetracker tracking": "eye_tracked",
}

# ── Trial event maps ───────────────────────────────────────────────────────
# Each maps a substring of an event comment → trial struct field name.
# Priority order during dispatch: TIME → INFO → SEGMENT → DASH → OUTCOME

TIME_EVENTS: dict[str, str] = {
    "Start": "Start",
    "Fixation point on": "Fixation_point_on",
    "Fixation point off": "Fixation_point_off",
    "Reward end": "Reward_end",
    "Target 1 presented": "Target_1_presented",
    "Target 2 presented": "Target_2_presented",
    "Targets off": "Targets_off",
    "Fixation acquired": "Fixation_acquired",
    "Broke fixation": "Broke_fixation",
    "Target 1 acquired": "Choicetime",
    "Target 2 acquired": "Choicetime",
    "Target 1 off": "Target_1_off",
    "Feedback flash on": "Feedback_flash_on",
    "Feedback flash off": "Feedback_flash_off",
    "Fixation exited": "Fixation_exited",
    "Target deadline exceeded": "Target_deadline_exceeded",
}

SEGMENT_EVENTS: dict[str, str] = {
    "Experiment": "Task",
    "Fixation color": "Fixation_color",
    "Target 1 color": "Target_1_color",
    "Target 2 color": "Target_2_color",
    "Trial type": "Trial_type",
    "Target 1 on the": "Target_1_side",
}

# Keys that match coordinate pattern "(x, y) deg" use coord regex.
# Keys that match reward pattern "name(NNms)" use reward regex.
# All others use time_ms / size_deg regex.
INFORMATION_EVENTS: dict[str, str] = {
    "Fixation position": "Fixation_position",
    "Fixation size": "Fixation_size",
    "Fixation acceptance window": "Fixation_acceptance_window",
    "Target 1 position": "Target_1_position",
    "Target 1 size": "Target_1_size",
    "Target 1 acceptance window": "Target_1_acceptance_window",
    "Target 2 position": "Target_2_position",
    "Target 2 size": "Target_2_size",
    "Target 2 acceptance window": "Target_2_acceptance_window",
    "Requested fixation hold time": "Requested_fixation_hold_time",
    "Requested fixation duration": "Requested_fixation_duration",
    "Requested timeout": "Requested_timeout",
    "Requested time between trials": "Requested_time_between_trials",
    "Requested target 1 hold time": "Requested_target_1_hold_time",
    "Requested target 2 hold time": "Requested_target_2_hold_time",
    "Requested target 1 duration": "Requested_target_1_duration",
    "Requested target 2 time offset": "Requested_target_2_time_offset",
    "Requested target 1 timeout": "Requested_target_1_timeout",
    "Requested penalty box duration": "Requested_penalty_box_duration",
    "Reward start": "Reward_start",
    "Requested target dim opacity": "Requested_target_dim_opacity",
    "Requested target 1 visible duration": "Requested_target_1_visible_duration",
    "Requested feedback flash duration": "Requested_feedback_flash_duration",
    "Requested choice timeout": "Requested_choice_timeout",
    "Requested target reach deadline": "Requested_target_reach_deadline",
}

DASH_EVENTS: list[str] = ["End", "Correct choice", "Wrong choice"]
OUTCOME_EVENTS: list[str] = ["Correct choice", "No choice", "Wrong choice"]

# ── Fields whose values are [x, y] arrays, split into _x / _y in CSV ──────
COORD_FIELDS: set[str] = {"Fixation_position", "Target_1_position", "Target_2_position"}

# ── Compiled regex patterns (mirrors BackRockFileLoader.m) ─────────────────
RE_EXP = re.compile(r"^Experiment (start|end):\s*(.+)$")
RE_TRIAL = re.compile(r"^Trial\s+(\d+):\s*(.*)$")
RE_COORD = re.compile(
    r"^(.*?)\s*\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)\s*deg$"
)
RE_REWARD = re.compile(r"^(.*?)\s*\(([\d\.]+)ms")
# Value may be a number or 'None'/'none' (e.g. "... reach deadline None ms" → NaN)
RE_TIME_MS = re.compile(r"^(.*?)\s+([-+]?\d*\.?\d+|None|none)\s*ms$")
RE_SIZE_DEG = re.compile(r"^(.*?)\s+([-+]?\d*\.?\d+)\s*(?:deg)?$")

# Offset-range event: "Requested time offset range [min, max] ms (active: [v1, v2, ...])"
RE_OFFSET_RANGE = re.compile(
    r"range\s*\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]"
)
RE_OFFSET_ACTIVE = re.compile(r"active:\s*\[([^\]]*)\]")

# Photodiode position coordinate "(x, y)" (no 'deg' suffix); used with .search()
RE_PD_COORD = re.compile(
    r"\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)"
)

