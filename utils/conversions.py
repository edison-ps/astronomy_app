"""Utility functions for astronomical unit and angle conversions.

This module provides the foundational conversion primitives used
throughout the astronomy_app package. All functions are pure and
stateless, making them straightforward to unit-test.
"""

import math
from typing import Tuple

# ---------------------------------------------------------------------------
# Physical and astronomical constants
# ---------------------------------------------------------------------------

AU_TO_KM: float = 1.495_978_707e8     # Astronomical Unit in km  (IAU 2012)
PARSEC_TO_LY: float = 3.261_563_78    # Parsecs to light-years
PARSEC_TO_AU: float = 206_264.806     # Parsecs to AU
PARSEC_TO_KM: float = 3.085_677_581e13  # Parsecs to km
LY_TO_KM: float = 9.460_730_472_5808e12  # Light-year in km
SPEED_OF_LIGHT_KMS: float = 299_792.458   # Speed of light in km/s


# ---------------------------------------------------------------------------
# Angle conversions
# ---------------------------------------------------------------------------

def degrees_to_radians(degrees: float) -> float:
    """Convert decimal degrees to radians."""
    return math.radians(degrees)


def radians_to_degrees(radians: float) -> float:
    """Convert radians to decimal degrees."""
    return math.degrees(radians)


def hms_to_degrees(hours: float, minutes: float, seconds: float) -> float:
    """Convert Right Ascension from H:M:S to decimal degrees.

    One hour of RA equals 15 degrees of arc.

    Args:
        hours: Hour component [0, 24).
        minutes: Minute component [0, 60).
        seconds: Second component [0, 60).

    Returns:
        Right Ascension in decimal degrees [0, 360).

    Raises:
        ValueError: If any component is outside its valid range.
    """
    if not (0 <= hours < 24):
        raise ValueError(f"Hours must be in [0, 24), got {hours}")
    if not (0 <= minutes < 60):
        raise ValueError(f"Minutes must be in [0, 60), got {minutes}")
    if not (0 <= seconds < 60):
        raise ValueError(f"Seconds must be in [0, 60), got {seconds}")

    total_hours = hours + minutes / 60.0 + seconds / 3600.0
    return total_hours * 15.0


def degrees_to_hms(degrees: float) -> Tuple[int, int, float]:
    """Convert decimal degrees to Hours:Minutes:Seconds.

    Args:
        degrees: Angle in degrees (will be normalised to [0, 360)).

    Returns:
        Tuple of (hours, minutes, seconds).
    """
    degrees = degrees % 360.0
    total_hours = degrees / 15.0
    hours = int(total_hours)
    remaining = (total_hours - hours) * 60.0
    minutes = int(remaining)
    seconds = (remaining - minutes) * 60.0
    return hours, minutes, seconds


def dms_to_degrees(degrees: float, arcminutes: float, arcseconds: float) -> float:
    """Convert Declination from D:M:S to decimal degrees.

    The sign is carried by the *degrees* argument; arcminutes and
    arcseconds are always interpreted as positive offsets.

    Args:
        degrees: Degree component [-90, 90].
        arcminutes: Arcminute component [0, 60).
        arcseconds: Arcsecond component [0, 60).

    Returns:
        Declination in decimal degrees.

    Raises:
        ValueError: If any component is outside its valid range.
    """
    if not (-90 <= degrees <= 90):
        raise ValueError(f"Degrees must be in [-90, 90], got {degrees}")
    if not (0 <= arcminutes < 60):
        raise ValueError(f"Arcminutes must be in [0, 60), got {arcminutes}")
    if not (0 <= arcseconds < 60):
        raise ValueError(f"Arcseconds must be in [0, 60), got {arcseconds}")

    sign = -1 if degrees < 0 else 1
    return sign * (abs(degrees) + arcminutes / 60.0 + arcseconds / 3600.0)


def degrees_to_dms(degrees: float) -> Tuple[int, int, float]:
    """Convert decimal degrees to Degrees:Arcminutes:Arcseconds.

    Returns:
        Tuple of (degrees, arcminutes, arcseconds).  The sign is on the
        degrees component.
    """
    sign = -1 if degrees < 0 else 1
    abs_deg = abs(degrees)
    d = int(abs_deg)
    remaining = (abs_deg - d) * 60.0
    m = int(remaining)
    s = (remaining - m) * 60.0
    return sign * d, m, s


def normalize_angle(angle_deg: float) -> float:
    """Normalise an angle to [0, 360)."""
    return angle_deg % 360.0


def normalize_angle_signed(angle_deg: float) -> float:
    """Normalise an angle to [-180, 180)."""
    angle = angle_deg % 360.0
    if angle >= 180.0:
        angle -= 360.0
    return angle


# ---------------------------------------------------------------------------
# Distance conversions
# ---------------------------------------------------------------------------

def parsec_to_light_years(parsecs: float) -> float:
    """Convert parsecs to light-years.

    Raises:
        ValueError: If parsecs is negative.
    """
    if parsecs < 0:
        raise ValueError(f"Distance cannot be negative, got {parsecs}")
    return parsecs * PARSEC_TO_LY


def light_years_to_parsecs(light_years: float) -> float:
    """Convert light-years to parsecs.

    Raises:
        ValueError: If light_years is negative.
    """
    if light_years < 0:
        raise ValueError(f"Distance cannot be negative, got {light_years}")
    return light_years / PARSEC_TO_LY


def au_to_km(au: float) -> float:
    """Convert Astronomical Units to kilometres."""
    return au * AU_TO_KM


def km_to_au(km: float) -> float:
    """Convert kilometres to Astronomical Units.

    Raises:
        ValueError: If km is negative.
    """
    if km < 0:
        raise ValueError(f"Distance cannot be negative, got {km}")
    return km / AU_TO_KM


def parsec_to_au(parsecs: float) -> float:
    """Convert parsecs to Astronomical Units."""
    if parsecs < 0:
        raise ValueError(f"Distance cannot be negative, got {parsecs}")
    return parsecs * PARSEC_TO_AU
