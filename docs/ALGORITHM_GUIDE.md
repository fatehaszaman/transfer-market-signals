# Algorithm guide

These cards explain the age-value heuristic already implemented. The parameters
are scenario assumptions, not an empirically validated transfer-price forecast.

## Annual age-value rate

Implementation: [`AgeValueCurve.annual_growth_rate`](../signals/age_value_curve.py).

```text
# Age Curve / Piecewise Lookup
# Input: age, position code, rating
# Output: one annual fractional growth/depreciation rate
# Time: O(1) for the fixed position/rating tables and bounded position codes
# Memory: O(1) additional

curve = position settings, or fallback for an unknown position
multiplier = first matching rating band
IF age is before peak:
    RETURN (base growth + capped years-to-peak boost) * multiplier
IF age is within peak:
    RETURN (0.02 - distance_from_peak_midpoint * 0.005) * multiplier
drop = cliff drop if at cliff age, otherwise increasing post-peak drop
RETURN -drop / multiplier
```

Position lookup is case-insensitive. Unknown positions use the fallback rather
than failing, and ratings outside the nominal range are not rejected here.

## Five-year trajectory and bounded peak search

Implementation: [`AgeValueCurve.expected_value_at_age`](../signals/age_value_curve.py).

```text
# Value Trajectory / Fixed-Horizon Compounding
# Input: current value, age, position, rating
# Output: five yearly rows, peak estimate, phase, five-year return
# Time: O(1) as written: 5 projection steps and at most 15 peak-search steps
# Memory: O(1) as written, including five output rows
# If horizons became H and P: O(H+P) time, O(H) output, O(1) auxiliary state

value = current value
FOR offsets 1 through 5:
    rate = annual_growth_rate(age entering the year)
    value = MAX(value * (1+rate), 0.5)
    APPEND rounded age, value, rate
RESTART from current value
FOR at most 15 future steps:
    COMPOUND with the same floor
    UPDATE maximum; stop after peak when value no longer increases
CLASSIFY current career phase
COMPUTE five-year return using the final rounded trajectory value
RETURN summary
```

Watch: zero starting value divides by zero in the return calculation.
The 0.5 floor is in EUR millions and affects low-valued scenarios. The reported
peak is only a bounded search, not a lifetime optimum. Fixed-size loops are
constant-time here; describing them as O(number of players) would incorrectly
include an outer batch operation this method does not perform.
