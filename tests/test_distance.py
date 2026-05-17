"""Unit tests for core.distance.

Demonstrates:
    * Scientific reference values (Hipparcos parallaxes, distance moduli)
    * Mathematical identity tests (modulus ↔ distance invertibility)
    * NumPy vectorised function tests
    * Negative/zero input validation
    * Parametrised edge-case scanning
"""

import math

import numpy as np
import pytest

from core.distance import (
    distance_from_modulus,
    distance_modulus,
    extinction_corrected_distance,
    modulus_from_distance,
    parallax_to_distance_ly,
    parallax_to_distance_parsec,
    photometric_distances,
    stellar_distance_3d,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def proxima_centauri() -> dict:
    """Hipparcos parallax and distance for Proxima Centauri."""
    return {
        "parallax_arcsec": 0.768_5,  # ~768.5 mas (HIP 70890)
        "distance_pc": 1.301,
        "distance_ly": 4.243,
    }


@pytest.fixture
def sirius() -> dict:
    """Approximate data for Sirius (α CMa)."""
    return {
        "apparent_mag": -1.46,
        "absolute_mag":  1.42,
        "distance_pc":   2.64,
    }


# ===========================================================================
# Parallax → distance
# ===========================================================================

class TestParallaxToDistance:
    def test_proxima_centauri_parsec(self, proxima_centauri: dict) -> None:
        dist = parallax_to_distance_parsec(proxima_centauri["parallax_arcsec"])
        assert dist == pytest.approx(proxima_centauri["distance_pc"], rel=0.01)

    def test_proxima_centauri_light_years(self, proxima_centauri: dict) -> None:
        dist = parallax_to_distance_ly(proxima_centauri["parallax_arcsec"])
        assert dist == pytest.approx(proxima_centauri["distance_ly"], rel=0.01)

    def test_one_arcsec_parallax_is_one_parsec(self) -> None:
        assert parallax_to_distance_parsec(1.0) == pytest.approx(1.0)

    def test_half_arcsec_is_two_parsecs(self) -> None:
        assert parallax_to_distance_parsec(0.5) == pytest.approx(2.0)

    def test_zero_parallax_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            parallax_to_distance_parsec(0.0)

    def test_negative_parallax_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            parallax_to_distance_parsec(-0.1)

    @pytest.mark.parametrize("parallax, expected_pc", [
        (1.0,    1.0),
        (0.1,   10.0),
        (0.01, 100.0),
        (2.0,    0.5),
    ])
    def test_parametrised_known_values(
        self, parallax: float, expected_pc: float
    ) -> None:
        assert parallax_to_distance_parsec(parallax) == pytest.approx(expected_pc)


# ===========================================================================
# Distance modulus
# ===========================================================================

class TestDistanceModulus:
    def test_sirius_distance_modulus(self, sirius: dict) -> None:
        mu = distance_modulus(sirius["apparent_mag"], sirius["absolute_mag"])
        assert mu == pytest.approx(sirius["apparent_mag"] - sirius["absolute_mag"])

    def test_modulus_at_10_pc_is_zero(self) -> None:
        """At the canonical 10 pc reference distance, μ = 0."""
        assert modulus_from_distance(10.0) == pytest.approx(0.0, abs=1e-10)

    def test_modulus_at_100_pc(self) -> None:
        """μ = 5 at 100 pc."""
        assert modulus_from_distance(100.0) == pytest.approx(5.0)

    def test_modulus_at_1000_pc(self) -> None:
        """μ = 10 at 1 000 pc."""
        assert modulus_from_distance(1_000.0) == pytest.approx(10.0)

    def test_distance_from_modulus_zero(self) -> None:
        """μ = 0 → d = 10 pc."""
        assert distance_from_modulus(0.0) == pytest.approx(10.0)

    def test_distance_from_modulus_five(self) -> None:
        """μ = 5 → d = 100 pc."""
        assert distance_from_modulus(5.0) == pytest.approx(100.0)

    def test_modulus_negative_for_nearby_stars(self) -> None:
        """Stars closer than 10 pc have a negative distance modulus."""
        mu = modulus_from_distance(1.0)
        assert mu < 0.0

    def test_modulus_from_distance_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            modulus_from_distance(0.0)

    def test_modulus_from_distance_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            modulus_from_distance(-5.0)

    @pytest.mark.parametrize("distance_pc", [1.0, 10.0, 100.0, 1000.0, 1e6])
    def test_round_trip(self, distance_pc: float) -> None:
        """distance → modulus → distance must be invertible."""
        mu = modulus_from_distance(distance_pc)
        assert distance_from_modulus(mu) == pytest.approx(distance_pc, rel=1e-9)


# ===========================================================================
# Extinction-corrected distance
# ===========================================================================

class TestExtinctionCorrectedDistance:
    def test_zero_extinction_equals_standard_modulus(self) -> None:
        m, M = 12.0, 7.0
        d_standard = distance_from_modulus(distance_modulus(m, M))
        d_corrected = extinction_corrected_distance(m, M, extinction=0.0)
        assert d_corrected == pytest.approx(d_standard, rel=1e-10)

    def test_extinction_decreases_distance(self) -> None:
        """Correcting for extinction reduces the inferred distance."""
        d_no_ext = extinction_corrected_distance(15.0, 5.0, extinction=0.0)
        d_ext = extinction_corrected_distance(15.0, 5.0, extinction=1.0)
        assert d_ext < d_no_ext

    def test_known_extinction_scenario(self) -> None:
        """m=15, M=5, A=2 → μ=8 → d=10^(13/5)≈398.1 pc."""
        d = extinction_corrected_distance(15.0, 5.0, extinction=2.0)
        assert d == pytest.approx(10.0 ** (13.0 / 5.0), rel=1e-6)


# ===========================================================================
# 3-D stellar distance
# ===========================================================================

class TestStellarDistance3D:
    def test_same_position_zero_distance(self) -> None:
        d = stellar_distance_3d(100.0, 20.0, 50.0, 100.0, 20.0, 50.0)
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_on_axis_same_ra_dec(self) -> None:
        """Two stars along the same line of sight: distance = |d2 − d1|."""
        d = stellar_distance_3d(0.0, 0.0, 5.0, 0.0, 0.0, 15.0)
        assert d == pytest.approx(10.0, rel=1e-10)

    def test_orthogonal_stars(self) -> None:
        """Stars at RA=0° and RA=90°, both on equator at distance 1 pc.
        Cartesian: p1=(1,0,0), p2=(0,1,0) → distance=√2."""
        d = stellar_distance_3d(0.0, 0.0, 1.0, 90.0, 0.0, 1.0)
        assert d == pytest.approx(math.sqrt(2.0), rel=1e-10)

    def test_distance_is_symmetric(self) -> None:
        """Distance from A→B must equal B→A."""
        d_ab = stellar_distance_3d(100.0, -20.0, 30.0, 200.0, 40.0, 80.0)
        d_ba = stellar_distance_3d(200.0, 40.0, 80.0, 100.0, -20.0, 30.0)
        assert d_ab == pytest.approx(d_ba, rel=1e-10)

    def test_returns_float(self) -> None:
        result = stellar_distance_3d(0.0, 0.0, 1.0, 90.0, 0.0, 1.0)
        assert isinstance(result, float)


# ===========================================================================
# Vectorised photometric distances (NumPy)
# ===========================================================================

class TestPhotometricDistances:
    def test_single_value_matches_scalar(self) -> None:
        arr = np.array([15.0])
        result = photometric_distances(arr, absolute_mag=5.0)
        expected = distance_from_modulus(15.0 - 5.0)
        assert result[0] == pytest.approx(expected, rel=1e-10)

    def test_returns_numpy_array(self) -> None:
        arr = np.array([10.0, 12.0, 14.0])
        result = photometric_distances(arr, absolute_mag=5.0)
        assert isinstance(result, np.ndarray)

    def test_shape_preserved(self) -> None:
        arr = np.linspace(5.0, 25.0, 100)
        result = photometric_distances(arr, absolute_mag=5.0)
        assert result.shape == arr.shape

    def test_monotonicity(self) -> None:
        """Fainter apparent magnitudes imply larger distances."""
        arr = np.array([10.0, 12.0, 14.0, 16.0])
        result = photometric_distances(arr, absolute_mag=5.0)
        assert np.all(np.diff(result) > 0), "Distances should be monotonically increasing"

    def test_extinction_shifts_distances(self) -> None:
        arr = np.array([15.0, 16.0, 17.0])
        d_no_ext = photometric_distances(arr, absolute_mag=5.0, extinction=0.0)
        d_ext = photometric_distances(arr, absolute_mag=5.0, extinction=1.0)
        assert np.all(d_ext < d_no_ext)
