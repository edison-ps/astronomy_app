"""Unit tests for core.orbital.

Demonstrates:
    * Physical reference values (solar system bodies)
    * Kepler's third law verification
    * Vis-viva orbital speed checks
    * Convergence tests for iterative Kepler equation solver
    * Parametrised tests across eccentricity and semi-major axis ranges
    * Boundary condition and exception handling
"""

import math

import pytest

from core.orbital import (
    G,
    M_EARTH,
    M_SUN,
    R_EARTH,
    R_SUN,
    AU_TO_M,
    OrbitalElements,
    aphelion_distance,
    escape_velocity,
    hill_sphere_radius,
    mean_anomaly_at_time,
    orbital_energy,
    orbital_period,
    orbital_velocity,
    perihelion_distance,
    solve_kepler_equation,
    true_anomaly_from_eccentric,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def earth_orbit() -> dict:
    """Approximate Keplerian elements for Earth (J2000.0)."""
    return {
        "semi_major_axis_au": 1.000_000_11,
        "eccentricity": 0.016_710_22,
        "period_years": 1.000_174,   # sidereal year ≈ 365.25 days
    }


@pytest.fixture
def jupiter_orbit() -> dict:
    """Approximate Keplerian elements for Jupiter."""
    return {
        "semi_major_axis_au": 5.202_9,
        "eccentricity": 0.048_9,
        "period_years": 11.86,
    }


@pytest.fixture
def halley_elements() -> OrbitalElements:
    """Approximate orbital elements for 1P/Halley."""
    return OrbitalElements(
        semi_major_axis=17.8,
        eccentricity=0.967,
        inclination=162.3,
        longitude_ascending=58.4,
        argument_perihelion=111.3,
        mean_anomaly_epoch=38.4,
    )


# ===========================================================================
# OrbitalElements dataclass
# ===========================================================================

class TestOrbitalElementsValidation:
    def test_valid_elements(self) -> None:
        elems = OrbitalElements(
            semi_major_axis=1.0,
            eccentricity=0.0,
            inclination=0.0,
            longitude_ascending=0.0,
            argument_perihelion=0.0,
            mean_anomaly_epoch=0.0,
        )
        assert elems.semi_major_axis == 1.0

    def test_negative_semi_major_axis_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            OrbitalElements(-1.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def test_zero_semi_major_axis_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            OrbitalElements(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def test_eccentricity_one_raises(self) -> None:
        """e = 1 is parabolic, not elliptical — should raise."""
        with pytest.raises(ValueError, match="[Ee]ccentricity"):
            OrbitalElements(1.0, 1.0, 0.0, 0.0, 0.0, 0.0)

    def test_eccentricity_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="[Ee]ccentricity"):
            OrbitalElements(1.0, -0.1, 0.0, 0.0, 0.0, 0.0)

    def test_high_eccentricity_accepted(self, halley_elements: OrbitalElements) -> None:
        assert halley_elements.eccentricity == pytest.approx(0.967)


# ===========================================================================
# Kepler's third law — orbital_period
# ===========================================================================

class TestOrbitalPeriod:
    def test_earth_period_one_year(self, earth_orbit: dict) -> None:
        period = orbital_period(earth_orbit["semi_major_axis_au"])
        assert period == pytest.approx(earth_orbit["period_years"], rel=0.001)

    def test_jupiter_period(self, jupiter_orbit: dict) -> None:
        period = orbital_period(jupiter_orbit["semi_major_axis_au"])
        assert period == pytest.approx(jupiter_orbit["period_years"], rel=0.01)

    def test_period_at_1_au_is_1_year(self) -> None:
        """By definition: a = 1 AU, M = 1 M☉ → T = 1 year."""
        assert orbital_period(1.0, 1.0) == pytest.approx(1.0, rel=1e-10)

    def test_period_scales_with_a_to_3_2(self) -> None:
        """T ∝ a^(3/2): doubling semi-major axis multiplies period by 2√2."""
        T1 = orbital_period(1.0)
        T2 = orbital_period(2.0)
        assert T2 / T1 == pytest.approx(2.0 ** 1.5, rel=1e-9)

    def test_zero_semi_major_axis_raises(self) -> None:
        with pytest.raises(ValueError):
            orbital_period(0.0)

    def test_zero_stellar_mass_raises(self) -> None:
        with pytest.raises(ValueError):
            orbital_period(1.0, mass_star_solar=0.0)

    @pytest.mark.parametrize("a_au, expected_years", [
        (0.387, 0.241),   # Mercury
        (0.723, 0.615),   # Venus
        (1.524, 1.881),   # Mars
        (9.537, 29.46),   # Saturn
    ])
    def test_solar_system_parametrised(
        self, a_au: float, expected_years: float
    ) -> None:
        period = orbital_period(a_au)
        assert period == pytest.approx(expected_years, rel=0.01)


# ===========================================================================
# Vis-viva — orbital_velocity
# ===========================================================================

class TestOrbitalVelocity:
    def test_earth_circular_velocity(self) -> None:
        """Earth in circular orbit at 1 AU ≈ 29.78 km/s."""
        v = orbital_velocity(1.0, 1.0)
        assert v == pytest.approx(29.78, rel=0.01)

    def test_circular_orbit_r_equals_a(self) -> None:
        """In a circular orbit (r = a), vis-viva simplifies to v = √(GM/a)."""
        a_m = 1.0 * AU_TO_M
        expected_ms = math.sqrt(G * M_SUN / a_m)
        expected_kms = expected_ms / 1_000.0
        v = orbital_velocity(1.0, 1.0)
        assert v == pytest.approx(expected_kms, rel=1e-6)

    def test_faster_at_perihelion(self) -> None:
        """Object at perihelion moves faster than at aphelion."""
        a, e = 1.5, 0.5
        q = a * (1 - e)   # perihelion
        Q = a * (1 + e)   # aphelion
        v_peri = orbital_velocity(a, q)
        v_aphe = orbital_velocity(a, Q)
        assert v_peri > v_aphe

    def test_unbound_orbit_raises(self) -> None:
        """r > 2a violates vis-viva for a bound orbit."""
        with pytest.raises(ValueError, match="[Uu]nbound"):
            orbital_velocity(1.0, 3.0)   # r = 3a > 2a

    def test_zero_distance_raises(self) -> None:
        with pytest.raises(ValueError):
            orbital_velocity(1.0, 0.0)


# ===========================================================================
# Escape velocity
# ===========================================================================

class TestEscapeVelocity:
    def test_earth_escape_velocity(self) -> None:
        """Earth escape velocity ≈ 11.19 km/s."""
        v = escape_velocity(M_EARTH, R_EARTH)
        assert v == pytest.approx(11.19, rel=0.01)

    def test_sun_escape_velocity(self) -> None:
        """Solar surface escape velocity ≈ 617.7 km/s."""
        v = escape_velocity(M_SUN, R_SUN)
        assert v == pytest.approx(617.7, rel=0.01)

    def test_escape_velocity_zero_mass_raises(self) -> None:
        with pytest.raises(ValueError):
            escape_velocity(0.0, R_EARTH)

    def test_escape_velocity_zero_radius_raises(self) -> None:
        with pytest.raises(ValueError):
            escape_velocity(M_EARTH, 0.0)

    def test_escape_velocity_larger_for_denser_body(self) -> None:
        """Same mass, smaller radius → higher escape velocity."""
        v_large = escape_velocity(M_EARTH, R_EARTH)
        v_small = escape_velocity(M_EARTH, R_EARTH / 2.0)
        assert v_small > v_large


# ===========================================================================
# Kepler's equation solver
# ===========================================================================

class TestSolveKeplerEquation:
    def test_circular_orbit_E_equals_M(self) -> None:
        """For e = 0, E = M exactly."""
        for M in [0.0, 45.0, 90.0, 180.0, 270.0]:
            E = solve_kepler_equation(M, 0.0)
            assert E == pytest.approx(M, abs=1e-9)

    def test_nearly_parabolic_convergence(self) -> None:
        """High-eccentricity orbit (Halley-like) should still converge."""
        E = solve_kepler_equation(mean_anomaly_deg=45.0, eccentricity=0.967)
        # Verify by back-substituting into Kepler's equation
        E_rad = math.radians(E)
        M_rad = math.radians(45.0)
        residual = M_rad - E_rad + 0.967 * math.sin(E_rad)
        assert abs(residual) < 1e-9

    @pytest.mark.parametrize("M_deg, e", [
        (0.0,    0.0),
        (90.0,   0.3),
        (180.0,  0.5),
        (270.0,  0.7),
        (360.0,  0.9),
    ])
    def test_kepler_solution_self_consistent(
        self, M_deg: float, e: float
    ) -> None:
        """E returned must satisfy M = E − e·sin(E) to numerical tolerance."""
        E_deg = solve_kepler_equation(M_deg, e)
        E_rad = math.radians(E_deg)
        M_rad = math.radians(M_deg)
        residual = abs(M_rad - E_rad + e * math.sin(E_rad))
        assert residual < 1e-9, f"Kepler residual {residual} for M={M_deg}, e={e}"


# ===========================================================================
# True anomaly
# ===========================================================================

class TestTrueAnomaly:
    def test_E0_gives_nu0(self) -> None:
        """E = 0 ↔ perihelion → ν = 0."""
        assert true_anomaly_from_eccentric(0.0, 0.5) == pytest.approx(0.0, abs=1e-10)

    def test_E180_gives_nu180(self) -> None:
        """E = 180° → ν = 180° (aphelion)."""
        nu = true_anomaly_from_eccentric(180.0, 0.5)
        assert nu == pytest.approx(180.0, abs=1e-8)

    def test_true_anomaly_in_valid_range(self) -> None:
        for E in range(0, 361, 15):
            nu = true_anomaly_from_eccentric(float(E), 0.3)
            assert 0.0 <= nu < 360.0


# ===========================================================================
# Orbital geometry and energy
# ===========================================================================

class TestOrbitalGeometry:
    def test_perihelion_distance(self) -> None:
        assert perihelion_distance(1.0, 0.5) == pytest.approx(0.5)

    def test_aphelion_distance(self) -> None:
        assert aphelion_distance(1.0, 0.5) == pytest.approx(1.5)

    def test_circular_orbit_peri_aphe_equal(self) -> None:
        assert perihelion_distance(5.0, 0.0) == pytest.approx(aphelion_distance(5.0, 0.0))

    def test_orbital_energy_negative(self) -> None:
        """Bound orbits must have negative specific orbital energy."""
        assert orbital_energy(1.0, 1.0) < 0.0

    def test_orbital_energy_larger_a_less_negative(self) -> None:
        """More distant orbit has less negative (higher) energy."""
        E1 = orbital_energy(1.0)
        E10 = orbital_energy(10.0)
        assert E10 > E1

    def test_mean_anomaly_at_time(self) -> None:
        """Mean anomaly advances linearly with time."""
        n = 0.985_647  # degrees/day (Earth)
        M0 = 0.0
        M_1yr = mean_anomaly_at_time(n, 365.25, M0)
        # After ~1 sidereal year, M should be ≈ 360° → normalised ≈ 0°
        assert M_1yr == pytest.approx(0.0, abs=1.0)   # within 1°

    def test_hill_sphere_earth(self) -> None:
        """Earth's Hill sphere radius ≈ 0.010 AU."""
        r_hill = hill_sphere_radius(1.0, 0.0167, M_EARTH, M_SUN)
        assert r_hill == pytest.approx(0.0100, rel=0.05)
