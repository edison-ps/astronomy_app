"""Stellar photometry and magnitude calculations.

Implements Pogson's logarithmic magnitude scale and the standard
flux–magnitude conversions used in optical and near-infrared astronomy.

The default zero-point assumes the AB magnitude system (Oke & Gunn 1983),
where all bands share a flat spectral reference flux of 3 631 Jy.
For Vega-system zero-points see :data:`ZERO_POINT_FLUX_VEGA`.
"""

import math
from typing import Dict, List

import numpy as np

# ---------------------------------------------------------------------------
# Photometric zero-point constants
# ---------------------------------------------------------------------------

#: AB system zero-point flux (Jy = 10⁻²⁶ W m⁻² Hz⁻¹)
ZERO_POINT_AB_JY: float = 3_631.0

#: Vega-system zero-point fluxes per band (Jy), from Bessell (1998)
ZERO_POINT_FLUX_VEGA: Dict[str, float] = {
    "U": 1_810.0,
    "B": 4_260.0,
    "V": 3_631.0,
    "R": 3_064.0,
    "I": 2_416.0,
    "J": 1_589.0,
    "H": 1_021.0,
    "K":   657.0,
}

#: Absolute magnitude of the Sun in the V-band (Willmer 2018)
SUN_ABSOLUTE_MAG_V: float = 4.83


# ---------------------------------------------------------------------------
# Flux ↔ magnitude
# ---------------------------------------------------------------------------

def flux_to_magnitude(flux: float, zero_point_flux: float = ZERO_POINT_AB_JY) -> float:
    """Convert flux density to magnitude using Pogson's law.

    m = −2.5 · log₁₀(F / F₀)

    Args:
        flux:            Observed flux density in Jy.
        zero_point_flux: Zero-point flux density F₀ in Jy.

    Returns:
        Magnitude on the chosen photometric system.

    Raises:
        ValueError: If *flux* is not positive.
    """
    if flux <= 0.0:
        raise ValueError(f"Flux must be positive (got {flux} Jy).")
    return -2.5 * math.log10(flux / zero_point_flux)


def magnitude_to_flux(
    magnitude: float,
    zero_point_flux: float = ZERO_POINT_AB_JY,
) -> float:
    """Convert magnitude to flux density.

    F = F₀ · 10^(−m / 2.5)

    Args:
        magnitude:       Magnitude value.
        zero_point_flux: Zero-point flux density F₀ in Jy.

    Returns:
        Flux density in Jy.
    """
    return zero_point_flux * 10.0 ** (-magnitude / 2.5)


# ---------------------------------------------------------------------------
# Absolute ↔ apparent magnitude
# ---------------------------------------------------------------------------

def absolute_magnitude(
    apparent_mag: float,
    distance_pc: float,
    extinction: float = 0.0,
) -> float:
    """Derive absolute magnitude from apparent magnitude and distance.

    M = m − 5 · log₁₀(d) + 5 − A

    Args:
        apparent_mag: Observed apparent magnitude.
        distance_pc:  Distance in parsecs.
        extinction:   Interstellar extinction in magnitudes (default 0).

    Returns:
        Absolute magnitude.

    Raises:
        ValueError: If *distance_pc* is not positive.
    """
    if distance_pc <= 0.0:
        raise ValueError(f"Distance must be positive (got {distance_pc} pc).")
    return apparent_mag - 5.0 * math.log10(distance_pc) + 5.0 - extinction


def apparent_magnitude(
    absolute_mag: float,
    distance_pc: float,
    extinction: float = 0.0,
) -> float:
    """Compute apparent magnitude from absolute magnitude and distance.

    m = M + 5 · log₁₀(d) − 5 + A

    Args:
        absolute_mag: Intrinsic absolute magnitude.
        distance_pc:  Distance in parsecs.
        extinction:   Interstellar extinction in magnitudes (default 0).

    Returns:
        Apparent magnitude.

    Raises:
        ValueError: If *distance_pc* is not positive.
    """
    if distance_pc <= 0.0:
        raise ValueError(f"Distance must be positive (got {distance_pc} pc).")
    return absolute_mag + 5.0 * math.log10(distance_pc) - 5.0 + extinction


# ---------------------------------------------------------------------------
# Flux ratios and combined magnitudes
# ---------------------------------------------------------------------------

def magnitude_difference_to_flux_ratio(delta_mag: float) -> float:
    """Convert a magnitude difference Δm = m₂ − m₁ to a flux ratio F₁/F₂.

    F₁/F₂ = 10^(Δm / 2.5)
    """
    return 10.0 ** (delta_mag / 2.5)


def combined_magnitude(magnitudes: List[float]) -> float:
    """Compute the combined magnitude of multiple unresolved sources.

    m_total = −2.5 · log₁₀( Σ 10^(−mᵢ / 2.5) )

    Args:
        magnitudes: List of individual magnitudes.

    Returns:
        Total (combined) magnitude.

    Raises:
        ValueError: If *magnitudes* is empty.
    """
    if not magnitudes:
        raise ValueError("Magnitudes list must not be empty.")
    flux_sum = sum(10.0 ** (-m / 2.5) for m in magnitudes)
    return -2.5 * math.log10(flux_sum)


# ---------------------------------------------------------------------------
# Luminosity and surface brightness
# ---------------------------------------------------------------------------

def luminosity_solar(
    absolute_mag: float,
    solar_abs_mag: float = SUN_ABSOLUTE_MAG_V,
) -> float:
    """Convert absolute magnitude to luminosity in solar units.

    L / L☉ = 10^((M☉ − M) / 2.5)

    Args:
        absolute_mag:  Absolute magnitude of the star.
        solar_abs_mag: Absolute magnitude of the Sun (V-band default).

    Returns:
        Luminosity in units of solar luminosity L☉.
    """
    return 10.0 ** ((solar_abs_mag - absolute_mag) / 2.5)


def surface_brightness(magnitude: float, area_arcsec2: float) -> float:
    """Compute surface brightness in magnitudes per square arcsecond.

    SB = m + 2.5 · log₁₀(Ω)

    Args:
        magnitude:    Integrated magnitude of the object.
        area_arcsec2: Apparent area in arcseconds².

    Returns:
        Surface brightness in mag arcsec⁻².

    Raises:
        ValueError: If *area_arcsec2* is not positive.
    """
    if area_arcsec2 <= 0.0:
        raise ValueError(f"Area must be positive (got {area_arcsec2} arcsec²).")
    return magnitude + 2.5 * math.log10(area_arcsec2)


# ---------------------------------------------------------------------------
# NumPy-vectorised utilities
# ---------------------------------------------------------------------------

def magnitudes_to_flux_array(
    magnitudes: np.ndarray,
    zero_point_flux: float = ZERO_POINT_AB_JY,
) -> np.ndarray:
    """Vectorised magnitude → flux conversion for NumPy arrays.

    Args:
        magnitudes:      1-D array of magnitudes.
        zero_point_flux: Zero-point flux density in Jy.

    Returns:
        Array of flux densities in Jy.
    """
    return zero_point_flux * 10.0 ** (-magnitudes / 2.5)


def flux_array_to_magnitudes(
    fluxes: np.ndarray,
    zero_point_flux: float = ZERO_POINT_AB_JY,
) -> np.ndarray:
    """Vectorised flux → magnitude conversion for NumPy arrays.

    Args:
        fluxes:          1-D array of positive flux densities in Jy.
        zero_point_flux: Zero-point flux density in Jy.

    Returns:
        Array of magnitudes.

    Raises:
        ValueError: If any flux value is non-positive.
    """
    if np.any(fluxes <= 0.0):
        raise ValueError("All flux values must be positive.")
    return -2.5 * np.log10(fluxes / zero_point_flux)
