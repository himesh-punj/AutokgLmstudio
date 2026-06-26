"""
validators/quantity.py
----------------------
Deterministic quantity normalisation + comparison — the part of validation that
must NOT be left to a stochastic LLM (it flip-flopped run-to-run). Given a value
string and an operator, this reduces both sides to a magnitude in a canonical
unit and compares them with plain arithmetic, so the numeric verdict is 100%
reproducible.

Design notes:
  • Notation is normalised to a *physical magnitude*, not the raw printed number:
    a ratio "1 in N" / "1:N" and a percentage are both dimensionless ratios with
    magnitude 1/N and pct/100 — so a slope written either way is comparable, and a
    larger N is correctly a *gentler* slope.
  • Dimensions are tracked so incommensurable quantities (length vs angle, % vs
    metres) are never compared — the caller treats that as 'Needs Human'.
  • This is intentionally a *common-case* parser. Anything it cannot parse returns
    (None, None); the caller then defers to a human rather than guessing. Extend
    the unit tables as new notations appear — that is the only thing to grow here,
    and it grows deterministically.
"""

import math
import re

# canonical unit tables  → (multiplier to base, dimension). Base units:
#   length → metres, speed → km/h, mass → tonnes, angle → degrees
_SPEED  = {"km/h": 1.0, "km/hr": 1.0, "kmph": 1.0, "kph": 1.0, "kmh": 1.0}
_MASS   = {"tonne": 1.0, "tonnes": 1.0, "ton": 1.0, "tons": 1.0, "te": 1.0, "t": 1.0, "kg": 0.001}
_ANGLE  = {"degree": 1.0, "degrees": 1.0, "deg": 1.0, "°": 1.0}
_LENGTH = {"km": 1000.0, "metre": 1.0, "metres": 1.0, "meter": 1.0, "meters": 1.0,
           "m": 1.0, "cm": 0.01, "mm": 0.001}

# speed checked before length so "km/h" isn't misread as the length "km"
_UNIT_TABLES = (("speed", _SPEED), ("mass", _MASS), ("angle", _ANGLE), ("length", _LENGTH))

_RATIO_RE = re.compile(r"1\s*(?:in|:)\s*([0-9][0-9,]*\.?[0-9]*)", re.I)
_PCT_RE   = re.compile(r"([0-9]+\.?[0-9]*)\s*%")
_NUM_RE   = re.compile(r"-?[0-9][0-9,]*\.?[0-9]*")

NUMERIC_OPS = {">=", "<=", ">", "<", "==", "in_range", "every"}


def _f(s: str) -> float:
    return float(s.replace(",", ""))


def to_magnitude(text) -> tuple[float | None, str | None]:
    """Reduce a value string to (magnitude, dimension) in canonical units, or
    (None, None) if it cannot be parsed. dimension ∈ {ratio, speed, mass, angle,
    length, number}. Ratios and percentages collapse to a dimensionless 'ratio'."""
    if text is None:
        return (None, None)
    t = str(text).strip()
    if not t:
        return (None, None)
    low = t.lower()

    # "1 in N" / "1:N" → magnitude 1/N (gentler as N grows)
    m = _RATIO_RE.search(t)
    if m:
        n = _f(m.group(1))
        return (1.0 / n, "ratio") if n else (None, None)

    # "X%" → fraction (same dimension as a ratio, so % and 1-in-N are comparable)
    m = _PCT_RE.search(t)
    if m:
        return (_f(m.group(1)) / 100.0, "ratio")

    # number + (optional) unit
    nm = _NUM_RE.search(low)
    if not nm:
        return (None, None)
    val = _f(nm.group(0))

    for dim, table in _UNIT_TABLES:
        for u, mult in table.items():
            if re.search(rf"(?<![a-z]){re.escape(u)}(?![a-z])", low):
                return (val * mult, dim)
    if re.search(r"m\s*/\s*s|mps", low):     # m/s → km/h
        return (val * 3.6, "speed")
    return (val, "number")                    # bare number, no recognised unit


def commensurable(dim_a: str | None, dim_b: str | None) -> bool:
    """True iff two dimensions can be soundly compared. A bare 'number' is allowed
    against a 'ratio' (e.g. a ratio rule whose value came through unitless)."""
    if dim_a is None or dim_b is None:
        return False
    if dim_a == dim_b:
        return True
    return {dim_a, dim_b} == {"ratio", "number"}


def compare(op: str, value_mag: float, threshold_mag: float, rel_tol: float = 0.02) -> bool | None:
    """Apply a numeric operator. Returns True (complies) / False (violates), or
    None if the operator isn't a numeric one this module handles."""
    if op == ">=":
        return value_mag >= threshold_mag * (1 - rel_tol)
    if op == "<=":
        return value_mag <= threshold_mag * (1 + rel_tol)
    if op == ">":
        return value_mag > threshold_mag
    if op == "<":
        return value_mag < threshold_mag
    if op in ("==", "in_range", "every"):
        return math.isclose(value_mag, threshold_mag, rel_tol=rel_tol)
    return None


def numeric_verdict(op: str, value_text, threshold_text) -> tuple[str, str]:
    """Deterministic verdict for a numeric requirement. Returns (verdict, reason)
    where verdict ∈ {Compliant, Non-Compliant, Needs Human}. 'Needs Human' means
    the values could not be put on a common, comparable footing — defer, never
    guess."""
    v_mag, v_dim = to_magnitude(value_text)
    t_mag, t_dim = to_magnitude(threshold_text)
    if v_mag is None or t_mag is None:
        return ("Needs Human", f"Could not parse a comparable magnitude from "
                               f"'{value_text}' vs '{threshold_text}'.")
    if not commensurable(v_dim, t_dim):
        return ("Needs Human", f"Incommensurable: value is {v_dim} but requirement is "
                               f"{t_dim} ({value_text} vs {threshold_text}).")
    ok = compare(op, v_mag, t_mag)
    if ok is None:
        return ("Needs Human", f"Operator '{op}' is not numeric.")
    rel = "satisfies" if ok else "violates"
    return (("Compliant" if ok else "Non-Compliant"),
            f"{v_dim}: {v_mag:g} {rel} {op} {t_mag:g} (normalised).")
