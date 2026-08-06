# Parameter map for jives comments
# last updates: 06-18-2026 by Xuefei Yu
import re

# ── File role schema (mirrors BlackrockLoader.m properties) ────────────────
# One session is split across files by filename prefix; the loader picks each
# file by its prefix and verifies the expected data is present before using it.
COMMENT_PREFIX_PRIMARY = "NSP"   # NSP-*.nev: comments + comment timing
COMMENT_PREFIX_LEGACY = "HUB"    # HUB-*.nev: legacy fallback for comments
SPIKE_PREFIX = "HUB"             # HUB-*.nev: online spike timing + waveforms
EYE_PREFIX = "NSP"               # NSP-*.ns2: eye data (+ photodiode by default)
EYE_IDENTIFIER = "ns2"           # eye stream extension
LFP_PREFIX = "Hub"               # Hub-*.ns2: local field potential
LFP_IDENTIFIER = "ns2"           # LFP extension (mirrors EYE_IDENTIFIER)
PHOTODIODE_PREFIX = "NSP"        # fallback/forced separate-file prefix
PHOTODIODE_IDENTIFIER = "ns4"    # fallback/forced separate-file extension

# ── Channel layout of the eye EYE_PREFIX-*.ns2 stream (1-based, as MATLAB) ──
# The photodiode rides in the same file by default: eye signal on rows 1-3,
# photodiode on rows 4-6. Kept 1-based so these line up with the MATLAB
# properties; they are converted to 0-based only at the indexing site.
EYE_CHANNELS = (1, 2, 3)
PHOTODIODE_CHANNELS = (4, 5, 6)

# ── Physical units of the continuous streams ───────────────────────────────
# Each NSx extended header carries a Units string alongside the
# MaxAnalogValue/MaxDigitalValue scale factor. NPMK's openNSx ignores it and
# calls every scaled result "uV" (openNSx.m:1350), which is right for the LFP
# (Units 'uV') but off by 1000x for the analog-input eye/photodiode channels
# (Units 'mV'). We keep MATLAB's numbers and record what the unit actually is.
UNIT_LABELS: dict[str, str] = {
    "uV": "microVolts",
    "µV": "microVolts",
    "mV": "milliVolts",
    "V": "Volts",
}
# Multiplier that takes a value in the header's unit to microVolts, for the
# opt-in to_microvolts=True conversion.
UNIT_TO_MICROVOLTS: dict[str, float] = {
    "uV": 1.0,
    "µV": 1.0,
    "mV": 1e3,
    "V": 1e6,
}
MICROVOLTS_LABEL = "microVolts"

# Online-spike "unit" ids that are not sorted single units: 0 = unsorted
# threshold crossings, 255 = noise/invalidated. Dropped unless include_unsorted=True.
UNSORTED_UNIT_IDS = (0, 255)

# ── Trial-segmentation defaults (ms) ───────────────────────────────────────
# Window per trial = [Start - pre, End + post]; spikes binned at bin width.
# Waveforms reuse the same pre/post window (one waveform per in-window spike).
SEGMENT_PRE_MS = 500
SEGMENT_POST_MS = 500
SEGMENT_BIN_MS = 1

# Refractory window (ms) for info.ViolationRate: the fraction of a unit's ISIs
# shorter than this over its full spike train.
SPIKE_ISI_VIOLATION_MS = 1

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

