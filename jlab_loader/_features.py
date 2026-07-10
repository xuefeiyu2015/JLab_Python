"""
Derived feature computation.
Mirrors the feature block in BackRockFileLoader.m (lines 518-550).
All functions mutate the trials list in-place.
"""

from __future__ import annotations

import math


def _polar_angle(x: float, y: float) -> float:
    """
    Convert (x, y) Cartesian to polar angle in degrees using MATLAB convention:
        angle = mod(90 - rad2deg(atan2(y, x)), 360)
        if angle >= 180: angle -= 360
    Result is in (-180, 180].
    """
    theta_deg = math.degrees(math.atan2(y, x))
    angle = (90.0 - theta_deg) % 360.0
    if angle >= 180.0:
        angle -= 360.0
    return angle


def _eccentricity(x: float, y: float) -> float:
    return math.sqrt(x * x + y * y)


# Names of fields added by compute_derived_features(), in output order.
# Add new derived field names here when extending the function below.
DERIVED_FIELDS: list[str] = [
    "Target_1_angle",
    "Target_1_eccentricity",
    "Target_2_angle",
    "Target_2_eccentricity",
    "Stimulus_direction",
    "Choose_target",
    "Choose_leftright",
]


def compute_derived_features(trials: list[dict]) -> None:
    """
    Add Target_1/2_angle, Target_1/2_eccentricity, Stimulus_direction,
    Choose_target, Choose_leftright to each trial dict.
    """
    for trial in trials:
        t1x, t1y = trial.get("Target_1_position", [math.nan, math.nan])
        t2x, t2y = trial.get("Target_2_position", [math.nan, math.nan])

        t1_ok = not (math.isnan(t1x) or math.isnan(t1y))
        t2_ok = not (math.isnan(t2x) or math.isnan(t2y))

        t1_angle = _polar_angle(t1x, t1y) if t1_ok else math.nan
        t2_angle = _polar_angle(t2x, t2y) if t2_ok else math.nan
        t1_ecc = _eccentricity(t1x, t1y) if t1_ok else math.nan
        t2_ecc = _eccentricity(t2x, t2y) if t2_ok else math.nan

        trial["Target_1_angle"] = t1_angle
        trial["Target_2_angle"] = t2_angle
        trial["Target_1_eccentricity"] = t1_ecc
        trial["Target_2_eccentricity"] = t2_ecc

        # Stimulus direction: +1 if Target 1 is on the right (angle >= 0), -1 if left
        if not math.isnan(t1_angle):
            trial["Stimulus_direction"] = 1 if t1_angle >= 0 else -1
        else:
            trial["Stimulus_direction"] = math.nan

        # Choose_target: last character of Choosen_choice string ("choice-1" → 1)
        cc = trial.get("Choosen_choice")
        if cc and isinstance(cc, str) and cc[-1].isdigit():
            choose_target = int(cc[-1])
        else:
            choose_target = math.nan
        trial["Choose_target"] = choose_target

        # Choose_leftright: +1 if chosen target is on the right, -1 if left
        if not math.isnan(choose_target):
            chosen_angle = t1_angle if choose_target == 1 else t2_angle
            if not math.isnan(chosen_angle):
                trial["Choose_leftright"] = 1 if chosen_angle >= 0 else -1
            else:
                trial["Choose_leftright"] = math.nan
        else:
            trial["Choose_leftright"] = math.nan
