"""Orbital mechanics calculations.

Implements Kepler's laws and the vis-viva equation for computing the
fundamental parameters of two-body gravitational orbits.

Units used internally:
    * Length   → AU (Astronomical Units) or metres (SI functions)
    * Time     → years (Kepler) or seconds (SI functions)
    * Mass     → solar masses (Kepler) or kg (SI functions)
    * Velocity → km/s (all velocity outputs)

Constants are CODATA/IAU recommended values.
"""

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------

G: float = 6.674_30e-11     # Gravitational constant  [m³ kg⁻¹ s⁻²]
M_SUN: float = 1.989_00e30  # Solar mass              [kg]
M_EARTH: float = 5.972_17e24  # Earth mass            [kg]
R_EARTH: float = 6.371_0e6    # Mean Earth radius     [m]
R_SUN: float = 6.957_0e8      # Solar radius          [m]
AU_TO_M: float = 1.495_978_707e11  # 1 AU in metres
YR_TO_S: float = 3.155_76e7        # 1 Julian year in seconds


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class OrbitalElements:
    """Keplerian orbital elements for an elliptical orbit.

    All angular elements are in degrees; distances in AU.
    """

    semi_major_axis: float        # AU  — must be positive
    eccentricity: float           # dimensionless  [0, 1)
    inclination: float            # degrees
    longitude_ascending: float    # degrees  (Ω)
    argument_perihelion: float    # degrees  (ω)
    mean_anomaly_epoch: float     # degrees  at reference epoch

    def __post_init__(self) -> None:
        if self.semi_major_axis <= 0.0:
            raise ValueError("Semi-major axis must be positive.")
        if not (0.0 <= self.eccentricity < 1.0):
            raise ValueError(
                f"Eccentricity must be in [0, 1) for elliptical orbits "
                f"(got {self.eccentricity})."
            )


# ---------------------------------------------------------------------------
# Kepler's laws
# ---------------------------------------------------------------------------

def orbital_period(
    semi_major_axis_au: float,
    mass_star_solar: float = 1.0,
) -> float:
    """Compute orbital period via Kepler's third law.

    T² = a³ / M   (solar units: AU, years, M☉)

    Args:
        semi_major_axis_au: Semi-major axis in AU.
        mass_star_solar:    Central body mass in solar masses.

    Returns:
        Orbital period in years.

    Raises:
        ValueError: If either parameter is not positive.
    """
    if semi_major_axis_au <= 0.0:
        raise ValueError(f"Semi-major axis must be positive (got {semi_major_axis_au}).")
    if mass_star_solar <= 0.0:
        raise ValueError(f"Stellar mass must be positive (got {mass_star_solar}).")
    return math.sqrt(semi_major_axis_au ** 3 / mass_star_solar)


def orbital_velocity(
    semi_major_axis_au: float,
    distance_au: float,
    mass_star_solar: float = 1.0,
) -> float:
    """Compute instantaneous orbital speed via the vis-viva equation.

    v² = G·M · (2/r − 1/a)

    Args:
        semi_major_axis_au: Semi-major axis *a* in AU.
        distance_au:        Current distance from focus *r* in AU.
        mass_star_solar:    Central body mass in solar masses.

    Returns:
        Orbital speed in km/s.

    Raises:
        ValueError: If distances are not positive, or if the orbit would
                    be unbound (r > 2a).
    """
    if semi_major_axis_au <= 0.0 or distance_au <= 0.0:
        raise ValueError("Distances must be positive.")

    a_m = semi_major_axis_au * AU_TO_M
    r_m = distance_au * AU_TO_M
    M_kg = mass_star_solar * M_SUN

    v_sq = G * M_kg * (2.0 / r_m - 1.0 / a_m)
    if v_sq < 0.0:
        raise ValueError(
            f"Unbound condition: v² < 0 (r={distance_au} AU > 2a={2*semi_major_axis_au} AU)."
        )
    return math.sqrt(v_sq) / 1_000.0  # m/s → km/s


def escape_velocity(mass_kg: float, radius_m: float) -> float:
    """Compute the escape velocity from the surface of a body.

    v_esc = √(2·G·M / r)

    Args:
        mass_kg:  Mass of the body in kg.
        radius_m: Radius of the body in metres.

    Returns:
        Escape velocity in km/s.

    Raises:
        ValueError: If mass or radius are not positive.
    """
    if mass_kg <= 0.0 or radius_m <= 0.0:
        raise ValueError("Mass and radius must be positive.")
    return math.sqrt(2.0 * G * mass_kg / radius_m) / 1_000.0


# ---------------------------------------------------------------------------
# Kepler's equation and anomaly conversions
# ---------------------------------------------------------------------------

def solve_kepler_equation(
    mean_anomaly_deg: float,
    eccentricity: float,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> float:
    """Solve Kepler's equation  M = E − e·sin(E)  for eccentric anomaly.

    Uses Newton–Raphson iteration starting from E₀ = M.

    Args:
        mean_anomaly_deg: Mean anomaly *M* in degrees.
        eccentricity:     Orbital eccentricity *e*.
        tol:              Convergence criterion in radians.
        max_iter:         Maximum number of Newton–Raphson iterations.

    Returns:
        Eccentric anomaly *E* in degrees.

    Raises:
        ValueError: If the iteration fails to converge.
    """
    M = math.radians(mean_anomaly_deg)
    E = M  # initial guess

    for _ in range(max_iter):
        dE = (M - E + eccentricity * math.sin(E)) / (1.0 - eccentricity * math.cos(E))
        E += dE
        if abs(dE) < tol:
            return math.degrees(E)

    raise ValueError(
        f"Kepler's equation did not converge after {max_iter} iterations "
        f"(M={mean_anomaly_deg}°, e={eccentricity})."
    )


def true_anomaly_from_eccentric(
    eccentric_anomaly_deg: float,
    eccentricity: float,
) -> float:
    """Convert eccentric anomaly to true anomaly.

    tan(ν/2) = √((1+e)/(1−e)) · tan(E/2)

    Args:
        eccentric_anomaly_deg: Eccentric anomaly *E* in degrees.
        eccentricity:          Orbital eccentricity.

    Returns:
        True anomaly *ν* in degrees, normalised to [0, 360).
    """
    E = math.radians(eccentric_anomaly_deg)
    # Use atan2 form to avoid the tan singularity near E = π and fp artefacts near E = 2π
    nu_rad = 2.0 * math.atan2(
        math.sqrt(1.0 + eccentricity) * math.sin(E / 2.0),
        math.sqrt(1.0 - eccentricity) * math.cos(E / 2.0),
    )
    nu = math.degrees(nu_rad) % 360.0
    return nu if nu < 360.0 else 0.0


def mean_anomaly_at_time(
    mean_motion_deg_per_day: float,
    elapsed_days: float,
    mean_anomaly_epoch_deg: float = 0.0,
) -> float:
    """Compute mean anomaly at a time offset from epoch.

    M(t) = M₀ + n · Δt

    Args:
        mean_motion_deg_per_day:  Mean motion *n* in degrees/day.
        elapsed_days:             Time elapsed since epoch in days.
        mean_anomaly_epoch_deg:   Mean anomaly at epoch (default 0°).

    Returns:
        Mean anomaly in degrees, normalised to [0, 360).
    """
    return (mean_anomaly_epoch_deg + mean_motion_deg_per_day * elapsed_days) % 360.0


# ---------------------------------------------------------------------------
# Orbital geometry
# ---------------------------------------------------------------------------

def perihelion_distance(semi_major_axis_au: float, eccentricity: float) -> float:
    """Perihelion (closest approach) distance in AU.  q = a(1 − e)"""
    return semi_major_axis_au * (1.0 - eccentricity)


def aphelion_distance(semi_major_axis_au: float, eccentricity: float) -> float:
    """Aphelion (farthest point) distance in AU.  Q = a(1 + e)"""
    return semi_major_axis_au * (1.0 + eccentricity)


def orbital_energy(
    semi_major_axis_au: float,
    mass_star_solar: float = 1.0,
) -> float:
    """Compute the specific (per-unit-mass) orbital energy.

    ε = −G·M / (2·a)

    Args:
        semi_major_axis_au: Semi-major axis in AU.
        mass_star_solar:    Central body mass in solar masses.

    Returns:
        Specific orbital energy in J/kg (always negative for bound orbits).
    """
    a_m = semi_major_axis_au * AU_TO_M
    M_kg = mass_star_solar * M_SUN
    return -G * M_kg / (2.0 * a_m)


def hill_sphere_radius(
    semi_major_axis_au: float,
    eccentricity: float,
    mass_planet_kg: float,
    mass_star_kg: float = M_SUN,
) -> float:
    """Estimate the Hill sphere radius of a planet.

    r_H ≈ a(1−e) · ∛(m_p / (3·M_★))

    Args:
        semi_major_axis_au: Planet's semi-major axis in AU.
        eccentricity:       Planet's orbital eccentricity.
        mass_planet_kg:     Planet mass in kg.
        mass_star_kg:       Star mass in kg (default 1 M☉).

    Returns:
        Hill sphere radius in AU.
    """
    q = perihelion_distance(semi_major_axis_au, eccentricity)
    return q * (mass_planet_kg / (3.0 * mass_star_kg)) ** (1.0 / 3.0)
