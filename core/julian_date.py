"""Julian Date calculations.

The Julian Date (JD) is a continuous count of days from the start of
the Julian Period (noon, January 1, 4713 BC in the proleptic Julian
calendar).  It is the standard time scale in positional astronomy.

Key epochs:
    J2000.0  =  JD 2 451 545.0   (2000 Jan 1.5 TT)
    MJD zero =  JD 2 400 000.5   (1858 Nov 17.0 UT)

All algorithms follow Jean Meeus, *Astronomical Algorithms* (2nd ed.),
Chapters 7 and 12.
"""

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Tuple


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class JulianDate:
    """A Julian Date with convenience properties."""

    jd: float

    @property
    def mjd(self) -> float:
        """Modified Julian Date (MJD = JD − 2 400 000.5)."""
        return self.jd - 2_400_000.5

    @property
    def j2000(self) -> float:
        """Days elapsed since the J2000.0 epoch (JD 2 451 545.0)."""
        return self.jd - 2_451_545.0

    def __repr__(self) -> str:
        return f"JulianDate(jd={self.jd:.6f}, mjd={self.mjd:.6f})"


# ---------------------------------------------------------------------------
# Conversion functions
# ---------------------------------------------------------------------------

def gregorian_to_julian(year: int, month: int, day: float) -> float:
    """Convert a Gregorian calendar date to a Julian Date.

    Args:
        year:  Gregorian year (negative for BC, 0 = 1 BC).
        month: Month [1, 12].
        day:   Day of month; may include a fractional time-of-day
               (e.g. 1.5 = noon on the 1st).

    Returns:
        Julian Date as a float.
    """
    if month <= 2:
        year -= 1
        month += 12

    A = math.floor(year / 100.0)
    B = 2 - A + math.floor(A / 4.0)

    jd = (
        math.floor(365.25 * (year + 4716))
        + math.floor(30.6001 * (month + 1))
        + day
        + B
        - 1524.5
    )
    return jd


def julian_to_gregorian(jd: float) -> Tuple[int, int, float]:
    """Convert a Julian Date to a Gregorian calendar date.

    Args:
        jd: Julian Date.

    Returns:
        Tuple (year, month, day) where *day* may carry a fractional
        time-of-day component.
    """
    jd = jd + 0.5
    Z = math.floor(jd)
    F = jd - Z

    if Z < 2_299_161:
        A = Z
    else:
        alpha = math.floor((Z - 1_867_216.25) / 36_524.25)
        A = Z + 1 + alpha - math.floor(alpha / 4.0)

    B = A + 1524
    C = math.floor((B - 122.1) / 365.25)
    D = math.floor(365.25 * C)
    E = math.floor((B - D) / 30.6001)

    day = B - D - math.floor(30.6001 * E) + F

    month = int(E - 1 if E < 14 else E - 13)
    year = int(C - 4716 if month > 2 else C - 4715)

    return year, month, day


def datetime_to_julian(dt: datetime) -> float:
    """Convert a Python :class:`datetime` to a Julian Date.

    The datetime is assumed to be UTC when it carries no timezone info.

    Args:
        dt: A :class:`datetime` object.

    Returns:
        Julian Date as a float.
    """
    day_frac = (
        dt.day
        + dt.hour / 24.0
        + dt.minute / 1440.0
        + dt.second / 86_400.0
        + dt.microsecond / 86_400_000_000.0
    )
    return gregorian_to_julian(dt.year, dt.month, day_frac)


def julian_to_datetime(jd: float) -> datetime:
    """Convert a Julian Date to a UTC :class:`datetime`.

    Args:
        jd: Julian Date.

    Returns:
        A timezone-aware :class:`datetime` in UTC.
    """
    year, month, day_frac = julian_to_gregorian(jd)
    day = int(day_frac)
    frac = day_frac - day

    hours_total = frac * 24.0
    hour = int(hours_total)
    minutes_total = (hours_total - hour) * 60.0
    minute = int(minutes_total)
    seconds_total = (minutes_total - minute) * 60.0
    second = int(seconds_total)
    microsecond = round((seconds_total - second) * 1_000_000)

    # Handle potential overflow from rounding
    if microsecond >= 1_000_000:
        second += 1
        microsecond -= 1_000_000

    return datetime(year, month, day, hour, minute, second, microsecond,
                    tzinfo=timezone.utc)


def local_sidereal_time(jd: float, longitude_deg: float) -> float:
    """Calculate the Local Sidereal Time (LST) at a given instant.

    Uses the polynomial expression for Greenwich Mean Sidereal Time from
    the *Astronomical Almanac* (good to ~0.1 s over several centuries).

    Args:
        jd:            Julian Date (UT1).
        longitude_deg: Observer's east longitude in degrees.

    Returns:
        LST in degrees [0, 360).
    """
    D = jd - 2_451_545.0  # Days from J2000.0

    # Greenwich Mean Sidereal Time in degrees (USNO simplified formula)
    # Valid to ~0.1 s over several centuries.
    gmst = (280.460_618_37 + 360.985_647_366_29 * D) % 360.0

    return (gmst + longitude_deg) % 360.0


def julian_centuries_j2000(jd: float) -> float:
    """Return the number of Julian centuries elapsed since J2000.0.

    Args:
        jd: Julian Date.

    Returns:
        Julian centuries (T) since J2000.0.
    """
    return (jd - 2_451_545.0) / 36_525.0
