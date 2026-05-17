"""Unit tests for core.magnitude.

Demonstrates:
    * Exact mathematical identities (Pogson's law)
    * Scientific reference values (Sirius, Vega, the Sun)
    * Parametrised tolerance tests across the magnitude range
    * NumPy array tests for vectorised functions
    * Edge-case validation: empty lists, zero flux, negative distance
"""

import math

import numpy as np
import pytest

from core.magnitude import (
    SUN_ABSOLUTE_MAG_V,
    ZERO_POINT_AB_JY,
    ZERO_POINT_FLUX_VEGA,
    absolute_magnitude,
    apparent_magnitude,
    combined_magnitude,
    flux_array_to_magnitudes,
    flux_to_magnitude,
    luminosity_solar,
    magnitude_difference_to_flux_ratio,
    magnitude_to_flux,
    magnitudes_to_flux_array,
    surface_brightness,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def sirius_data() -> dict:
    """Reference photometric data for Sirius (α CMa)."""
    return {
        "apparent_mag":  -1.46,
        "absolute_mag":   1.42,
        "distance_pc":    2.64,
    }


@pytest.fixture
def sun_data() -> dict:
    """Reference photometric data for the Sun."""
    return {
        "absolute_mag": SUN_ABSOLUTE_MAG_V,
        "distance_pc":  1.0 / 3.085_677_581e13 * 1.495_978_707e11,  # 1 AU in pc
    }


# ===========================================================================
# flux_to_magnitude / magnitude_to_flux
# ===========================================================================

class TestFluxMagnitudeConversion:
    """Verify Pogson's law:  m = −2.5 log₁₀(F/F₀)."""

    def test_zero_point_flux_gives_zero_magnitude(self) -> None:
        """A source at the zero-point flux has m = 0."""
        assert flux_to_magnitude(ZERO_POINT_AB_JY) == pytest.approx(0.0, abs=1e-10)

    def test_flux_ten_times_smaller_adds_2_5_mag(self) -> None:
        """Halving flux by 10× increases magnitude by 2.5."""
        m0 = flux_to_magnitude(1_000.0)
        m1 = flux_to_magnitude(100.0)
        assert m1 - m0 == pytest.approx(2.5, abs=1e-10)

    def test_flux_100x_smaller_adds_5_mag(self) -> None:
        delta = flux_to_magnitude(10.0) - flux_to_magnitude(1_000.0)
        assert delta == pytest.approx(5.0, abs=1e-10)

    def test_magnitude_to_flux_zero_mag(self) -> None:
        """m = 0 must return the zero-point flux."""
        assert magnitude_to_flux(0.0) == pytest.approx(ZERO_POINT_AB_JY, rel=1e-10)

    def test_magnitude_to_flux_positive_mag_lower_flux(self) -> None:
        assert magnitude_to_flux(5.0) < magnitude_to_flux(0.0)

    def test_round_trip_flux(self) -> None:
        flux = 1_234.567
        mag = flux_to_magnitude(flux)
        assert magnitude_to_flux(mag) == pytest.approx(flux, rel=1e-9)

    def test_round_trip_magnitude(self) -> None:
        mag = 12.345
        flux = magnitude_to_flux(mag)
        assert flux_to_magnitude(flux) == pytest.approx(mag, rel=1e-9)

    def test_zero_flux_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            flux_to_magnitude(0.0)

    def test_negative_flux_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            flux_to_magnitude(-1.0)

    @pytest.mark.parametrize("band, f0", ZERO_POINT_FLUX_VEGA.items())
    def test_vega_system_zero_point(self, band: str, f0: float) -> None:
        """Each Vega-system band's zero-point flux should give m=0."""
        assert flux_to_magnitude(f0, zero_point_flux=f0) == pytest.approx(0.0, abs=1e-10)


# ===========================================================================
# absolute_magnitude / apparent_magnitude
# ===========================================================================

class TestAbsoluteApparentMagnitude:
    def test_sirius_absolute_magnitude(self, sirius_data: dict) -> None:
        """Recover absolute magnitude of Sirius from known m and d."""
        M = absolute_magnitude(
            sirius_data["apparent_mag"], sirius_data["distance_pc"]
        )
        assert M == pytest.approx(sirius_data["absolute_mag"], abs=0.05)

    def test_sun_apparent_from_absolute(self) -> None:
        """Sun at 10 pc should have apparent mag equal to absolute mag."""
        m = apparent_magnitude(SUN_ABSOLUTE_MAG_V, 10.0)
        assert m == pytest.approx(SUN_ABSOLUTE_MAG_V, abs=1e-10)

    def test_apparent_brighter_at_closer_distance(self) -> None:
        m_near = apparent_magnitude(5.0, 10.0)
        m_far = apparent_magnitude(5.0, 100.0)
        assert m_near < m_far

    def test_absolute_brighter_further_away(self) -> None:
        """A star at greater distance appears fainter → inferred M is larger."""
        M_near = absolute_magnitude(10.0, 100.0)
        M_far = absolute_magnitude(10.0, 1_000.0)
        assert M_near > M_far

    def test_extinction_increases_apparent_magnitude(self) -> None:
        m_no_ext = apparent_magnitude(5.0, 100.0, extinction=0.0)
        m_ext = apparent_magnitude(5.0, 100.0, extinction=1.0)
        assert m_ext > m_no_ext

    def test_round_trip_distance(self) -> None:
        M, d = 3.0, 500.0
        m = apparent_magnitude(M, d)
        M_recovered = absolute_magnitude(m, d)
        assert M_recovered == pytest.approx(M, rel=1e-10)

    def test_zero_distance_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            absolute_magnitude(5.0, 0.0)

    def test_negative_distance_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            apparent_magnitude(5.0, -1.0)

    @pytest.mark.parametrize("distance_pc", [1.0, 10.0, 100.0, 1_000.0, 1e6])
    def test_apparent_absolute_round_trip_parametrised(
        self, distance_pc: float
    ) -> None:
        M = 5.0
        m = apparent_magnitude(M, distance_pc)
        assert absolute_magnitude(m, distance_pc) == pytest.approx(M, rel=1e-9)


# ===========================================================================
# Flux ratio and combined magnitude
# ===========================================================================

class TestFluxRatioCombinedMagnitude:
    def test_flux_ratio_zero_delta(self) -> None:
        """Δm = 0 → flux ratio = 1."""
        assert magnitude_difference_to_flux_ratio(0.0) == pytest.approx(1.0)

    def test_flux_ratio_2_5_mag(self) -> None:
        """Δm = 2.5 → ratio = 10."""
        assert magnitude_difference_to_flux_ratio(2.5) == pytest.approx(10.0, rel=1e-9)

    def test_flux_ratio_5_mag(self) -> None:
        """Δm = 5.0 → ratio = 100."""
        assert magnitude_difference_to_flux_ratio(5.0) == pytest.approx(100.0, rel=1e-9)

    def test_combined_single_source(self) -> None:
        """Combined magnitude of a single source equals that source's magnitude."""
        m = 8.5
        assert combined_magnitude([m]) == pytest.approx(m, abs=1e-10)

    def test_combined_two_equal_sources(self) -> None:
        """Two identical sources combine to a magnitude 2.5·log10(2) brighter."""
        m = 5.0
        expected = m - 2.5 * math.log10(2.0)
        assert combined_magnitude([m, m]) == pytest.approx(expected, rel=1e-9)

    def test_combined_dominated_by_bright_source(self) -> None:
        """Adding a faint companion barely changes the combined magnitude."""
        bright, faint = 5.0, 20.0
        combined = combined_magnitude([bright, faint])
        assert combined == pytest.approx(bright, abs=0.001)

    def test_combined_empty_list_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            combined_magnitude([])

    @pytest.mark.parametrize("mags", [
        [0.0, 0.0],
        [5.0, 5.0, 5.0],
        [-1.0, 2.0, 10.0],
    ])
    def test_combined_brighter_than_faintest(self, mags: list) -> None:
        """Combined magnitude must be brighter than the faintest component."""
        assert combined_magnitude(mags) < max(mags)


# ===========================================================================
# Luminosity
# ===========================================================================

class TestLuminositySolar:
    def test_sun_luminosity_is_one(self) -> None:
        """The Sun's own absolute magnitude should yield L = 1 L☉."""
        assert luminosity_solar(SUN_ABSOLUTE_MAG_V) == pytest.approx(1.0, rel=1e-9)

    def test_brighter_star_higher_luminosity(self) -> None:
        L_bright = luminosity_solar(-5.0)
        L_faint = luminosity_solar(10.0)
        assert L_bright > L_faint

    def test_luminosity_sirius(self) -> None:
        """Sirius M_V ≈ 1.42. V-band luminosity: 10^((4.83−1.42)/2.5) ≈ 23.1 L☉."""
        L = luminosity_solar(1.42)
        # Note: V-band luminosity differs from bolometric (~25 L☉) due to BC
        assert L == pytest.approx(23.1, abs=0.2)


# ===========================================================================
# Surface brightness
# ===========================================================================

class TestSurfaceBrightness:
    def test_unit_area(self) -> None:
        """For Ω = 1 arcsec², SB = m + 0."""
        assert surface_brightness(20.0, 1.0) == pytest.approx(20.0, abs=1e-10)

    def test_larger_area_brighter_sb(self) -> None:
        """A larger angular area gives a brighter (higher numeric) SB value."""
        sb1 = surface_brightness(20.0, 1.0)
        sb100 = surface_brightness(20.0, 100.0)
        assert sb100 > sb1

    def test_zero_area_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            surface_brightness(20.0, 0.0)

    def test_negative_area_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            surface_brightness(20.0, -1.0)


# ===========================================================================
# NumPy vectorised utilities
# ===========================================================================

class TestNumpyVectorised:
    def test_magnitudes_to_flux_array_shape(self) -> None:
        mags = np.array([5.0, 10.0, 15.0, 20.0])
        result = magnitudes_to_flux_array(mags)
        assert result.shape == mags.shape

    def test_magnitudes_to_flux_array_values(self) -> None:
        mags = np.array([0.0])
        result = magnitudes_to_flux_array(mags)
        assert result[0] == pytest.approx(ZERO_POINT_AB_JY, rel=1e-9)

    def test_flux_array_to_magnitudes_round_trip(self) -> None:
        mags = np.array([5.0, 10.0, 15.0])
        fluxes = magnitudes_to_flux_array(mags)
        recovered = flux_array_to_magnitudes(fluxes)
        np.testing.assert_allclose(recovered, mags, rtol=1e-9)

    def test_flux_array_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            flux_array_to_magnitudes(np.array([100.0, -1.0, 200.0]))

    def test_flux_array_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            flux_array_to_magnitudes(np.array([0.0]))

    def test_magnitudes_to_flux_monotone_decreasing(self) -> None:
        """Brighter magnitudes should produce higher fluxes."""
        mags = np.array([0.0, 5.0, 10.0, 15.0])
        fluxes = magnitudes_to_flux_array(mags)
        assert np.all(np.diff(fluxes) < 0), "Flux should decrease as magnitude increases"
