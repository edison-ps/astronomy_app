"""Unit tests for utils.conversions.

Demonstrates:
    * pytest.approx for floating-point tolerance
    * Parametrised test cases for systematic coverage
    * Edge-case validation (zero, negative, boundary values)
    * Round-trip (invertibility) tests
"""

import math

import pytest

from utils.conversions import (
    AU_TO_KM,
    PARSEC_TO_LY,
    au_to_km,
    degrees_to_dms,
    degrees_to_hms,
    degrees_to_radians,
    dms_to_degrees,
    hms_to_degrees,
    km_to_au,
    light_years_to_parsecs,
    normalize_angle,
    normalize_angle_signed,
    parsec_to_au,
    parsec_to_light_years,
    radians_to_degrees,
)


# ===========================================================================
# Angle: degrees ↔ radians
# ===========================================================================

class TestDegreesRadians:
    """Mutual-inverse and known-value tests for degree/radian conversions."""

    @pytest.mark.parametrize("deg, expected_rad", [
        (0.0,   0.0),
        (90.0,  math.pi / 2),
        (180.0, math.pi),
        (270.0, 3 * math.pi / 2),
        (360.0, 2 * math.pi),
        (-90.0, -math.pi / 2),
    ])
    def test_degrees_to_radians_known_values(self, deg: float, expected_rad: float) -> None:
        assert degrees_to_radians(deg) == pytest.approx(expected_rad)

    @pytest.mark.parametrize("rad, expected_deg", [
        (0.0,           0.0),
        (math.pi / 2,   90.0),
        (math.pi,       180.0),
        (2 * math.pi,   360.0),
        (-math.pi / 6, -30.0),
    ])
    def test_radians_to_degrees_known_values(self, rad: float, expected_deg: float) -> None:
        assert radians_to_degrees(rad) == pytest.approx(expected_deg)

    @pytest.mark.parametrize("deg", [0.0, 45.0, 90.0, 123.456, 360.0, -180.0])
    def test_round_trip(self, deg: float) -> None:
        """degrees → radians → degrees must recover the original value."""
        assert radians_to_degrees(degrees_to_radians(deg)) == pytest.approx(deg)


# ===========================================================================
# Angle: H:M:S ↔ degrees
# ===========================================================================

class TestHMSConversions:
    """Tests for Right Ascension H:M:S ↔ decimal degrees."""

    @pytest.mark.parametrize("h, m, s, expected_deg", [
        (0,  0,  0.0,   0.0),
        (6,  0,  0.0,  90.0),
        (12, 0,  0.0, 180.0),
        (18, 0,  0.0, 270.0),
        (23, 59, 59.0, pytest.approx(359.9958333, rel=1e-6)),
        (1,  30, 0.0,  22.5),
        (0,  0, 36.0,   0.15),
    ])
    def test_hms_to_degrees_known_values(
        self, h: int, m: int, s: float, expected_deg: float
    ) -> None:
        assert hms_to_degrees(h, m, s) == pytest.approx(expected_deg, rel=1e-6)

    def test_hms_to_degrees_invalid_hours(self) -> None:
        with pytest.raises(ValueError, match="Hours"):
            hms_to_degrees(24, 0, 0.0)

    def test_hms_to_degrees_negative_hours(self) -> None:
        with pytest.raises(ValueError, match="Hours"):
            hms_to_degrees(-1, 0, 0.0)

    def test_hms_to_degrees_invalid_minutes(self) -> None:
        with pytest.raises(ValueError, match="Minutes"):
            hms_to_degrees(1, 60, 0.0)

    def test_hms_to_degrees_invalid_seconds(self) -> None:
        with pytest.raises(ValueError, match="Seconds"):
            hms_to_degrees(1, 0, 60.0)

    @pytest.mark.parametrize("degrees", [0.0, 45.0, 90.0, 180.0, 270.0, 359.9])
    def test_round_trip(self, degrees: float) -> None:
        """degrees → H:M:S → degrees must recover the original value."""
        h, m, s = degrees_to_hms(degrees)
        result = hms_to_degrees(h, m, s)
        assert result == pytest.approx(degrees, rel=1e-9)


# ===========================================================================
# Angle: D:M:S ↔ degrees
# ===========================================================================

class TestDMSConversions:
    """Tests for Declination D:M:S ↔ decimal degrees."""

    @pytest.mark.parametrize("d, am, as_, expected", [
        (0,    0,  0.0,   0.0),
        (90,   0,  0.0,  90.0),
        (-90,  0,  0.0, -90.0),
        (45,  30,  0.0,  45.5),
        (-23, 26, 21.448, pytest.approx(-23.439291, rel=1e-5)),
        (0,    0, 36.0,   0.01),
    ])
    def test_dms_to_degrees_known_values(
        self, d: int, am: int, as_: float, expected: float
    ) -> None:
        assert dms_to_degrees(d, am, as_) == pytest.approx(expected, rel=1e-5)

    def test_dms_invalid_degrees(self) -> None:
        with pytest.raises(ValueError, match="Degrees"):
            dms_to_degrees(91, 0, 0.0)

    def test_dms_invalid_arcminutes(self) -> None:
        with pytest.raises(ValueError, match="Arcminutes"):
            dms_to_degrees(10, 60, 0.0)

    def test_dms_invalid_arcseconds(self) -> None:
        with pytest.raises(ValueError, match="Arcseconds"):
            dms_to_degrees(10, 0, 60.0)

    @pytest.mark.parametrize("degrees", [0.0, 15.0, -45.5, 89.999, -89.999])
    def test_round_trip(self, degrees: float) -> None:
        """degrees → D:M:S → degrees must recover the original value."""
        d, am, as_ = degrees_to_dms(degrees)
        result = dms_to_degrees(d, am, as_)
        assert result == pytest.approx(degrees, rel=1e-9)


# ===========================================================================
# Angle normalisation
# ===========================================================================

class TestNormalizeAngle:
    @pytest.mark.parametrize("angle, expected", [
        (0.0,    0.0),
        (360.0,  0.0),
        (361.0,  1.0),
        (-1.0,  359.0),
        (720.0,  0.0),
        (180.0, 180.0),
    ])
    def test_normalize_angle(self, angle: float, expected: float) -> None:
        assert normalize_angle(angle) == pytest.approx(expected)

    @pytest.mark.parametrize("angle, expected", [
        (0.0,      0.0),
        (90.0,    90.0),
        (180.0,  -180.0),    # range is [-180, 180), so 180 → -180
        (181.0,  -179.0),
        (360.0,    0.0),
        (-1.0,    -1.0),
        (-180.0, -180.0),    # -180 mod 360 = 180, which ≥ 180 → -180
    ])
    def test_normalize_angle_signed(self, angle: float, expected: float) -> None:
        assert normalize_angle_signed(angle) == pytest.approx(expected)


# ===========================================================================
# Distance conversions
# ===========================================================================

class TestDistanceConversions:
    """Tests for parsec, light-year, and AU conversions."""

    def test_parsec_to_ly_known_value(self) -> None:
        assert parsec_to_light_years(1.0) == pytest.approx(PARSEC_TO_LY, rel=1e-6)

    def test_parsec_to_ly_zero(self) -> None:
        assert parsec_to_light_years(0.0) == pytest.approx(0.0)

    def test_parsec_to_ly_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            parsec_to_light_years(-1.0)

    def test_light_years_to_parsec_inverse(self) -> None:
        pc = 10.0
        ly = parsec_to_light_years(pc)
        assert light_years_to_parsecs(ly) == pytest.approx(pc, rel=1e-10)

    def test_light_years_to_parsec_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            light_years_to_parsecs(-5.0)

    def test_au_to_km_earth_sun(self) -> None:
        """1 AU ≈ 149 597 870.7 km."""
        assert au_to_km(1.0) == pytest.approx(1.495_978_707e8, rel=1e-9)

    def test_km_to_au_round_trip(self) -> None:
        au = 5.2    # Jupiter's semi-major axis
        km = au_to_km(au)
        assert km_to_au(km) == pytest.approx(au, rel=1e-10)

    def test_km_to_au_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            km_to_au(-1.0)

    def test_parsec_to_au_known_value(self) -> None:
        """1 pc ≈ 206 264.806 AU."""
        assert parsec_to_au(1.0) == pytest.approx(206_264.806, rel=1e-5)

    @pytest.mark.parametrize("parsecs", [0.1, 1.0, 10.0, 100.0, 1000.0])
    def test_parsec_ly_round_trip_parametrised(self, parsecs: float) -> None:
        ly = parsec_to_light_years(parsecs)
        assert light_years_to_parsecs(ly) == pytest.approx(parsecs, rel=1e-10)
