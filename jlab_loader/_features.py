"""
Derived feature computation.
Port of BlackrockLoader.addDerivedTrialFeatures: relabel memory-type visual
saccades, fill the target polar position, then append the derived choice fields.
All functions mutate the trials list in-place.
"""

from __future__ import annotations

import math


def target_polar(xy, sent_theta: float, sent_rho: float) -> tuple[float, float]:
    """Polar (theta, rho) for one target, preferring the values the task sent.

    Port of BlackrockLoader.targetPolar. The result is in the TASK'S OWN
    convention: 0 = +x (right), counter-clockwise, [0, 360). The cartesian
    branch is wrapped into the same range so both paths agree.

    This is deliberately NOT the old compass frame (0 = up, clockwise,
    (-180, 180]): left/right needs hemifield() rather than a sign test, and a
    trials CSV written before this change holds compass angles.

    The sent values win because the cartesian comments are printed to two
    decimals: '(4.95, 4.95)' back-computes rho 6.99985 where the task sent 7.00.
    """
    if not (math.isnan(sent_theta) or math.isnan(sent_rho)):
        theta, rho = sent_theta, sent_rho
    else:
        x, y = xy
        theta = math.degrees(math.atan2(y, x))  # NaN propagates
        rho = math.hypot(x, y)
    return theta % 360.0, rho


def hemifield(theta: float) -> int:
    """+1 = right hemifield, -1 = left, for theta in [0, 360), 0 = +x, CCW.

    Port of BlackrockLoader.hemifield. Right is [0, 90] u (270, 360), which
    selects exactly what the old compass-frame '(angle >= 0)' test did, tie
    breaks included: straight up (90) is right, straight down (270) is left.
    NaN falls through to -1, as '(NaN >= 0)*2 - 1' did in MATLAB.
    """
    return 1 if (theta <= 90.0 or theta > 270.0) else -1


# Names of fields APPENDED by compute_derived_features(), in output order.
# Target_*_theta / Target_*_rho are also written, but they already exist in the
# trial template, so (as with MATLAB's deal into an existing field) they keep
# their template position. This list drives the CSV column order.
DERIVED_FIELDS: list[str] = [
    "Stimulus_direction",
    "Choose_target",
    "Choose_leftright",
]


def relabel_memory_saccades(trials: list[dict]) -> None:
    """
    Rewrite Task to 'memory_saccades_experiment' for memory-type visual saccades.

    Port of the head of BlackrockLoader.addDerivedTrialFeatures
    (BlackrockLoader.m:2838-2846):

        memory_idx = strcmp(trial_type, 'memory');
        task_idx   = strcmp(tasks, 'visual_saccades_experiment');
        tasks(memory_idx & task_idx) = {'memory_saccades_experiment'};

    BOTH conditions must hold — a 'memory' trial recorded under some other task
    name is left alone. Runs before the geometry/choice fields, matching MATLAB.
    """
    for trial in trials:
        if (
            trial.get("Trial_type") == "memory"
            and trial.get("Task") == "visual_saccades_experiment"
        ):
            trial["Task"] = "memory_saccades_experiment"


def compute_derived_features(trials: list[dict]) -> None:
    """
    Relabel memory-type visual saccades, fill Target_1/2_theta and _rho, then
    add Stimulus_direction, Choose_target and Choose_leftright to each trial.
    """
    relabel_memory_saccades(trials)

    for trial in trials:
        nan2 = [math.nan, math.nan]
        t1_theta, t1_rho = target_polar(
            trial.get("Target_1_position", nan2),
            trial.get("Target_1_theta", math.nan),
            trial.get("Target_1_rho", math.nan),
        )
        t2_theta, t2_rho = target_polar(
            trial.get("Target_2_position", nan2),
            trial.get("Target_2_theta", math.nan),
            trial.get("Target_2_rho", math.nan),
        )
        trial["Target_1_theta"] = t1_theta
        trial["Target_2_theta"] = t2_theta
        trial["Target_1_rho"] = t1_rho
        trial["Target_2_rho"] = t2_rho

        # Stimulus direction: hemifield of Target 1, NaN when it has no position
        trial["Stimulus_direction"] = (
            math.nan if math.isnan(t1_theta) else hemifield(t1_theta)
        )

        # Choose_target: last character of Choosen_choice string ("choice-1" → 1)
        cc = trial.get("Choosen_choice")
        if cc and isinstance(cc, str) and cc[-1].isdigit():
            choose_target = int(cc[-1])
        else:
            choose_target = math.nan
        trial["Choose_target"] = choose_target

        # Choose_leftright: hemifield of the chosen target. MATLAB seeds this
        # from Choose_target and only overwrites the 1 and 2 cases, so a stray
        # Choose_target of 3 passes straight through, and a NaN theta yields -1
        # (hemifield's fall-through), not NaN. Both quirks are kept deliberately.
        choose_leftright = choose_target
        if choose_target == 1:
            choose_leftright = hemifield(t1_theta)
        elif choose_target == 2:
            choose_leftright = hemifield(t2_theta)
        trial["Choose_leftright"] = choose_leftright
