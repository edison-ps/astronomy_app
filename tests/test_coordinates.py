"""Unit tests for core.coordinates.

Demonstrates:
    * Known-value assertions with scientific reference data
    * Round-trip (invertibility) tests for coordinate transforms
    * Parametrised tests across multiple sky positions
    * Edge-case handling (poles, antipodal points)
    * pytest.approx with absolute tolerances for angular quantities
"""

import math

import pytest

from core.coordinates import (
    OBLIQUITY_J2000,
    EquatorialCoord,
    EclipticCoord,
    GalacticCoord,
    HorizontalCoord,
    angular_separation,
    ecliptic_to_equatorial,
    equatorial_to_ecliptic,
    equatorial_to_galactic,
    equatorial_to_horizontal,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def sirius() -> dict:
    """Approximate J2000.0 equatorial coordinates of Sirius (α CMa)."""
    return {"ra": 101.287, "dec": -16.716}


@pytest.fixture
def canopus() -> dict:
    """Approximate J2000.0 equatorial coordinates of Canopus (α Car)."""
    return {"ra": 95.988, "dec": -52.696}


@pytest.fixture
def vega() -> dict:
    """Approximate J2000.0 equatorial coordinates of Vega (α Lyr)."""
    return {"ra": 279.235, "dec": 38.784}


# ===========================================================================
# EquatorialCoord dataclass validation
# ===========================================================================

class TestEquatorialCoordValidation:
    def test_valid_equatorial_coord(self) -> None:
        coord = EquatorialCoord(ra=180.0, dec=0.0)
        assert coord.ra == 180.0
        assert coord.dec == 0.0

    def test_ra_zero_valid(self) -> None:
        coord = EquatorialCoord(ra=0.0, dec=45.0)
        assert coord.ra == pytest.approx(0.0)

    def test_ra_upper_boundary_invalid(self) -> None:
        with pytest.raises(ValueError, match="RA"):
            EquatorialCoord(ra=360.0, dec=0.0)

    def test_ra_negative_invalid(self) -> None:
        with pytest.raises(ValueError, match="RA"):
            EquatorialCoord(ra=-0.001, dec=0.0)

    @pytest.mark.parametrize("dec", [-90.0, -45.0, 0.0, 45.0, 90.0])
    def test_valid_declination_range(self, dec: float) -> None:
        coord = EquatorialCoord(ra=0.0, dec=dec)
        assert coord.dec == dec

    @pytest.mark.parametrize("dec", [-90.001, 90.001, -100.0, 100.0])
    def test_invalid_declination_range(self, dec: float) -> None:
        with pytest.raises(ValueError, match="Dec"):
            EquatorialCoord(ra=0.0, dec=dec)


# ===========================================================================
# Equatorial → Horizontal
# ===========================================================================

class TestEquatorialToHorizontal:
    """Altitude / azimuth transformation tests.

    Reference geometry:
        Hour Angle (HA) = LST − RA.
        When HA = 0 the object is on the meridian.
    """

    def test_object_transiting_meridian_azimuth_south(self) -> None:
        """An object on the meridian (HA=0) south of zenith has Az ≈ 180°."""
        ra, dec = 90.0, 10.0
        lat = 51.5   # London latitude
        lst = ra     # HA = LST − RA = 0
        result = equatorial_to_horizontal(ra, dec, lat, 0.0, lst)
        assert result.azimuth == pytest.approx(180.0, abs=1e-5)

    def test_zenith_altitude(self) -> None:
        """Object at exactly the zenith (HA=0, dec=latitude) has Alt=90°."""
        lat = 40.0
        ra, dec = 0.0, lat
        result = equatorial_to_horizontal(ra, dec, lat, 0.0, ra)
        assert result.altitude == pytest.approx(90.0, abs=1e-5)

    def test_altitude_equatorial_observer(self) -> None:
        """At the equator, dec=0, HA=0 ⟹ Alt=90°."""
        result = equatorial_to_horizontal(0.0, 0.0, 0.0, 0.0, 0.0)
        assert result.altitude == pytest.approx(90.0, abs=1e-5)

    def test_circumpolar_star_always_positive_altitude(self) -> None:
        """A circumpolar star (dec > 90° − lat) should have Alt > 0° at any HA."""
        lat = 52.0
        dec = 80.0   # well circumpolar for lat=52
        for ha in range(0, 360, 30):
            lst = ha  # HA = LST when RA = 0
            result = equatorial_to_horizontal(0.0, dec, lat, 0.0, float(lst))
            assert result.altitude > 0.0, f"Expected positive altitude at HA={ha}"

    def test_below_horizon_negative_altitude(self) -> None:
        """South celestial pole is below horizon for northern observers."""
        result = equatorial_to_horizontal(0.0, -90.0, 45.0, 0.0, 0.0)
        assert result.altitude < 0.0

    def test_azimuth_in_valid_range(self) -> None:
        """Azimuth must always be in [0, 360)."""
        for ra_val in range(0, 360, 45):
            result = equatorial_to_horizontal(
                float(ra_val), 20.0, 45.0, 0.0, 180.0
            )
            assert 0.0 <= result.azimuth < 360.0


# ===========================================================================
# Equatorial ↔ Ecliptic
# ===========================================================================

class TestEquatorialEclipticTransform:
    """Tests for equatorial ↔ ecliptic coordinate transformation."""

    def test_vernal_equinox_maps_to_ecliptic_origin(self) -> None:
        """Vernal equinox at (RA=0, Dec=0) must map to ecliptic (λ=0, β=0)."""
        ecl = equatorial_to_ecliptic(0.0, 0.0)
        assert ecl.longitude == pytest.approx(0.0, abs=1e-8)
        assert ecl.latitude == pytest.approx(0.0, abs=1e-8)

    def test_north_ecliptic_pole(self) -> None:
        """The north ecliptic pole (β=90°) maps back to a specific equatorial point."""
        obliquity = OBLIQUITY_J2000
        eq = ecliptic_to_equatorial(0.0, 90.0, obliquity)
        # Dec should equal 90° − obliquity = 66.56°
        assert eq.dec == pytest.approx(90.0 - obliquity, abs=1e-5)

    @pytest.mark.parametrize("ra, dec", [
        (0.0,   0.0),
        (90.0,  23.4),
        (180.0, -10.0),
        (270.0,  66.6),
        (45.0,   30.0),
        (315.0, -45.0),
    ])
    def test_round_trip_parametrised(self, ra: float, dec: float) -> None:
        """equatorial → ecliptic → equatorial must recover original coords."""
        ecl = equatorial_to_ecliptic(ra, dec)
        eq = ecliptic_to_equatorial(ecl.longitude, ecl.latitude)
        assert eq.ra == pytest.approx(ra, abs=1e-7)
        assert eq.dec == pytest.approx(dec, abs=1e-7)

    def test_obliquity_parameter_accepted(self) -> None:
        """Custom obliquity values should be accepted without error."""
        result = equatorial_to_ecliptic(90.0, 23.0, obliquity=23.5)
        assert isinstance(result, EclipticCoord)


# ===========================================================================
# Equatorial → Galactic
# ===========================================================================

class TestEquatorialToGalactic:
    """Tests for equatorial → galactic coordinate transformation."""

    def test_galactic_centre_coordinates(self) -> None:
        """Galactic centre at RA≈266.40°, Dec≈−28.94° maps to (l,b) ≈ (0°, 0°)."""
        result = equatorial_to_galactic(266.405, -28.936)
        assert result.l == pytest.approx(0.0, abs=0.5)
        assert result.b == pytest.approx(0.0, abs=0.5)

    def test_galactic_north_pole_region(self) -> None:
        """Galactic north pole at RA≈192.86°, Dec≈27.13° has b ≈ 90°."""
        result = equatorial_to_galactic(192.860, 27.128)
        assert result.b == pytest.approx(90.0, abs=0.5)

    def test_galactic_longitude_in_valid_range(self) -> None:
        """Galactic longitude must be in [0, 360)."""
        result = equatorial_to_galactic(100.0, 20.0)
        assert 0.0 <= result.l < 360.0

    def test_galactic_latitude_in_valid_range(self) -> None:
        """Galactic latitude must be in [-90, 90]."""
        result = equatorial_to_galactic(100.0, 20.0)
        assert -90.0 <= result.b <= 90.0

    def test_returns_galactic_coord(self) -> None:
        result = equatorial_to_galactic(0.0, 0.0)
        assert isinstance(result, GalacticCoord)


# ===========================================================================
# Angular separation
# ===========================================================================

class TestAngularSeparation:
    """Tests for the Vincenty angular separation formula."""

    def test_same_point_zero_separation(self) -> None:
        sep = angular_separation(45.0, 30.0, 45.0, 30.0)
        assert sep == pytest.approx(0.0, abs=1e-10)

    def test_quarter_circle_along_equator(self) -> None:
        sep = angular_separation(0.0, 0.0, 90.0, 0.0)
        assert sep == pytest.approx(90.0, abs=1e-8)

    def test_half_circle_along_equator(self) -> None:
        sep = angular_separation(0.0, 0.0, 180.0, 0.0)
        assert sep == pytest.approx(180.0, abs=1e-8)

    def test_antipodal_points(self) -> None:
        sep = angular_separation(0.0, 45.0, 180.0, -45.0)
        assert sep == pytest.approx(180.0, abs=1e-8)

    def test_north_south_poles(self) -> None:
        sep = angular_separation(0.0, 90.0, 0.0, -90.0)
        assert sep == pytest.approx(180.0, abs=1e-8)

    def test_sirius_canopus_separation(
        self, sirius: dict, canopus: dict
    ) -> None:
        """Sirius–Canopus angular separation is ~36.2°."""
        sep = angular_separation(
            sirius["ra"], sirius["dec"],
            canopus["ra"], canopus["dec"],
        )
        # Computed value from J2000.0 catalogue coordinates used above
        assert sep == pytest.approx(36.22, abs=0.1)

    def test_symmetry(self, sirius: dict, canopus: dict) -> None:
        """Angular separation must be symmetric: sep(A,B) == sep(B,A)."""
        sep_ab = angular_separation(
            sirius["ra"], sirius["dec"], canopus["ra"], canopus["dec"]
        )
        sep_ba = angular_separation(
            canopus["ra"], canopus["dec"], sirius["ra"], sirius["dec"]
        )
        assert sep_ab == pytest.approx(sep_ba, rel=1e-12)

    @pytest.mark.parametrize("ra1,dec1,ra2,dec2", [
        (0.0, 0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0, 0.0),
        (359.0, 0.0, 1.0, 0.0),   # across the 0h/360° boundary
    ])
    def test_small_separations(
        self, ra1: float, dec1: float, ra2: float, dec2: float
    ) -> None:
        """Small separations should be non-negative and ≤ 180°."""
        sep = angular_separation(ra1, dec1, ra2, dec2)
        assert 0.0 <= sep <= 180.0
