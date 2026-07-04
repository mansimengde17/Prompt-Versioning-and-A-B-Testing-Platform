"""Welch's t-test with a dependency-free p-value.

Uses the normal approximation to the t distribution for the two-sided
p-value, which is accurate for the sample sizes experiments run at
(hundreds per variant). scipy.stats.ttest_ind(equal_var=False) can be
swapped in without changing callers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class TestResult:
    mean_a: float
    mean_b: float
    diff: float
    t_stat: float
    p_value: float
    ci_low: float
    ci_high: float
    significant: bool
    n_a: int
    n_b: int


def _mean_var(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    return mean, var


def _normal_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2))


def welch_t_test(a: list[float], b: list[float],
                 alpha: float = 0.05) -> TestResult:
    if len(a) < 2 or len(b) < 2:
        return TestResult(0, 0, 0, 0, 1.0, 0, 0, False, len(a), len(b))
    mean_a, var_a = _mean_var(a)
    mean_b, var_b = _mean_var(b)
    se = math.sqrt(var_a / len(a) + var_b / len(b))
    if se == 0:
        identical = mean_a == mean_b
        return TestResult(mean_a, mean_b, mean_b - mean_a, 0.0,
                          1.0 if identical else 0.0, 0, 0, not identical,
                          len(a), len(b))
    t = (mean_b - mean_a) / se
    p = 2 * _normal_sf(abs(t))
    z_crit = 1.959964
    diff = mean_b - mean_a
    return TestResult(
        mean_a=round(mean_a, 5), mean_b=round(mean_b, 5),
        diff=round(diff, 5), t_stat=round(t, 4), p_value=round(p, 6),
        ci_low=round(diff - z_crit * se, 5),
        ci_high=round(diff + z_crit * se, 5),
        significant=p < alpha, n_a=len(a), n_b=len(b))


def min_detectable_effect(a: list[float], b: list[float],
                          alpha: float = 0.05, power_z: float = 0.8416) -> float:
    """Approximate MDE at current sample sizes (80 percent power)."""
    if len(a) < 2 or len(b) < 2:
        return float("inf")
    _, var_a = _mean_var(a)
    _, var_b = _mean_var(b)
    se = math.sqrt(var_a / len(a) + var_b / len(b))
    return round((1.959964 + power_z) * se, 5)
