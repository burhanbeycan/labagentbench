from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


_RANGE_RE = re.compile(
    r"(?P<var>x\d+)\s*(?:between)?\s*(?P<lo>-?\d+(?:\.\d+)?)\s*(?:and|to|-)\s*(?P<hi>-?\d+(?:\.\d+)?)",
    flags=re.IGNORECASE,
)

_APPROX_RE = re.compile(
    r"(?P<var>x\d+)\s*~\s*(?P<val>-?\d+(?:\.\d+)?)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedHint:
    bounds: Optional[np.ndarray]  # (dim,2)
    center: Optional[np.ndarray]  # (dim,)


def parse_hint(text: str, var_names: Tuple[str, ...], default_bounds: np.ndarray) -> ParsedHint:
    """Parse a hint string into optional bounds narrowing and/or a centre point."""
    bounds = default_bounds.copy()
    seen_any_bounds = False

    name_to_i = {name.lower(): i for i, name in enumerate(var_names)}
    for m in _RANGE_RE.finditer(text):
        var = m.group("var").lower()
        if var in name_to_i:
            i = name_to_i[var]
            lo = float(m.group("lo"))
            hi = float(m.group("hi"))
            bounds[i, 0] = max(bounds[i, 0], min(lo, hi))
            bounds[i, 1] = min(bounds[i, 1], max(lo, hi))
            seen_any_bounds = True

    center = np.full((len(var_names),), np.nan, dtype=float)
    seen_any_center = False
    for m in _APPROX_RE.finditer(text):
        var = m.group("var").lower()
        if var in name_to_i:
            i = name_to_i[var]
            center[i] = float(m.group("val"))
            seen_any_center = True

    if not seen_any_center:
        center_out = None
    else:
        # fill missing centres with midpoint of bounds
        for i in range(len(var_names)):
            if np.isnan(center[i]):
                center[i] = 0.5 * (default_bounds[i, 0] + default_bounds[i, 1])
        center_out = center

    return ParsedHint(bounds=bounds if seen_any_bounds else None, center=center_out)
