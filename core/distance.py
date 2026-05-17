"""Stellar and cosmological distance calculations.

Covers the primary distance indicators used in observational astronomy:

    * Trigonometric parallax  →  distance in parsecs
    * Distance modulus        →  distance in parsecs
    * Extinction correction
    * 3-D Euclidean distance between stars

All distances are in parsecs unless explicitly stated.
"""

import math
from typing import Optional

import numpy as np

from utils.conversions import PARSEC_TO_LY, parsec_to_light_years


# ---------------------------------------------------------------------------
# Parallax
# ---------------------------------------------------------------------------

def parallax_to_distance_parsec(parallax_arcsec: float) -> float:
    """Convert trigonometric parallax to distance in parsecs.

    The defining relationship is  d [pc] = 1 / p [arcsec].

    Args:
        parallax_arcsec: Annual parallax in arcseconds.

    Returns:
        Distance in parsecs.

    Raises:
        ValueError: If *parallax_arcsec* is not positive.
    """
    if parallax_arcsec <= 0.0:
        raise ValueError(
            f"Parallax must be positive (got {parallax_arcsec} arcsec)."
        )
    return 1.0 / parallax_arcsec


def parallax_to_distance_ly(parallax_arcsec: float) -> float:
    """Convert trigonometric parallax to distance in light-years."""
    return parsec_to_light_years(parallax_to_distance_parsec(parallax_arcsec))


# ---------------------------------------------------------------------------
# Distance modulus
# ---------------------------------------------------------------------------

def distance_modulus(apparent_mag: float, absolute_mag: float) -> float:
    """Compute the distance modulus  μ = m − M.

    Args:
        apparent_mag:  Apparent magnitude *m*.
        absolute_mag:  Absolute magnitude *M*.

    Returns:
        Distance modulus (dimensionless; in magnitudes).
    """
    return apparent_mag - absolute_mag


def distance_from_modulus(distance_mod: float) -> float:
    """Convert distance modulus to distance in parsecs.

    d = 10^((μ + 5) / 5)

    Args:
        distance_mod: Distance modulus.

    Returns:
        Distance in parsecs.
    """
    return 10.0 ** ((distance_mod + 5.0) / 5.0)


def modulus_from_distance(distance_pc: float) -> float:
    """Convert distance in parsecs to distance modulus.

    μ = 5 · log₁₀(d) − 5

    Args:
        distance_pc: Distance in parsecs (must be positive).

    Returns:
        Distance modulus.

    Raises:
        ValueError: If *distance_pc* is not positive.
    """
    if distance_pc <= 0.0:
        raise ValueError(
            f"Distance must be positive (got {distance_pc} pc)."
        )
    return 5.0 * math.log10(distance_pc) - 5.0


# ---------------------------------------------------------------------------
# Extinction-corrected distance
# ---------------------------------------------------------------------------

def extinction_corrected_distance(
    apparent_mag: float,
    absolute_mag: float,
    extinction: float = 0.0,
) -> float:
    """Calculate distance accounting for interstellar extinction.

    The corrected distance modulus is  μ = m − M − A_V,
    where *A_V* is the extinction in the observed band.

    Args:
        apparent_mag: Observed apparent magnitude.
        absolute_mag: Intrinsic absolute magnitude.
        extinction:   Interstellar extinction in magnitudes (default 0).

    Returns:
        Distance in parsecs.
    """
    corrected_modulus = distance_modulus(apparent_mag, absolute_mag) - extinction
    return distance_from_modulus(corrected_modulus)


# ---------------------------------------------------------------------------
# 3-D stellar distance
# ---------------------------------------------------------------------------

def stellar_distance_3d(
    ra1: float, dec1: float, dist1_pc: float,
    ra2: float, dec2: float, dist2_pc: float,
) -> float:
    """Compute the 3-D Euclidean distance between two stars.

    Converts spherical (RA, Dec, r) coordinates to Cartesian, then
    applies the Euclidean norm.

    Args:
        ra1, dec1:  Equatorial coordinates of star 1 in degrees.
        dist1_pc:   Distance to star 1 in parsecs.
        ra2, dec2:  Equatorial coordinates of star 2 in degrees.
        dist2_pc:   Distance to star 2 in parsecs.

    Returns:
        3-D separation in parsecs.
    """
    def to_cartesian(ra_deg: float, dec_deg: float, r: float) -> np.ndarray:
        ra_r = math.radians(ra_deg)
        dec_r = math.radians(dec_deg)
        return np.array([
            r * math.cos(dec_r) * math.cos(ra_r),
            r * math.cos(dec_r) * math.sin(ra_r),
            r * math.sin(dec_r),
        ])

    p1 = to_cartesian(ra1, dec1, dist1_pc)
    p2 = to_cartesian(ra2, dec2, dist2_pc)
    return float(np.linalg.norm(p2 - p1))


# ---------------------------------------------------------------------------
# Photometric distance array (NumPy-vectorised)
# ---------------------------------------------------------------------------

def photometric_distances(
    apparent_mags: np.ndarray,
    absolute_mag: float,
    extinction: float = 0.0,
) -> np.ndarray:
    """Compute photometric distances for an array of apparent magnitudes.

    Useful for processing catalogues with NumPy.

    Args:
        apparent_mags: 1-D array of apparent magnitudes.
        absolute_mag:  Absolute magnitude of the standard candle.
        extinction:    Foreground extinction in magnitudes (scalar).

    Returns:
        Array of distances in parsecs.
    """
    mu = apparent_mags - absolute_mag - extinction
    return 10.0 ** ((mu + 5.0) / 5.0)
