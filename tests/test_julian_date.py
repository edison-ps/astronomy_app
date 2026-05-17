"""Unit tests for core.julian_date.

Demonstrates:
    * Deterministic tests against tabulated reference values
    * Round-trip tests for datetime ↔ JD conversions
    * pytest.approx with absolute tolerances in days/seconds
    * Fixture reuse across test classes
    * Edge cases: epoch boundaries, fractional days, leap years
"""

import math
from datetime import datetime, timezone

import pytest

from core.julian_date import (
    JulianDate,
    datetime_to_julian,
    gregorian_to_julian,
    julian_centuries_j2000,
    julian_to_datetime,
    julian_to_gregorian,
    local_sidereal_time,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def j2000_epoch() -> float:
    """JD for J2000.0 epoch: 2000 Jan 1.5 TT."""
    return 2_451_545.0


@pytest.fixture
def unix_epoch_jd() -> float:
    """Julian Date of Unix epoch: 1970 Jan 1.0 UTC."""
    return 2_440_587.5


# ===========================================================================
# JulianDate dataclass
# ===========================================================================

class TestJulianDateDataclass:
    def test_mjd_property(self, j2000_epoch: float) -> None:
        # MJD = JD − 2 400 000.5  →  2 451 545.0 − 2 400 000.5 = 51 544.5
        jd = JulianDate(jd=j2000_epoch)
        assert jd.mjd == pytest.approx(51_544.5)

    def test_j2000_offset_at_epoch(self, j2000_epoch: float) -> None:
        jd = JulianDate(jd=j2000_epoch)
        assert jd.j2000 == pytest.approx(0.0)

    def test_j2000_offset_one_century(self, j2000_epoch: float) -> None:
        jd = JulianDate(jd=j2000_epoch + 36_525.0)
        assert jd.j2000 == pytest.approx(36_525.0)

    def test_repr_contains_jd(self) -> None:
        jd = JulianDate(jd=2_451_545.0)
        assert "2451545" in repr(jd)


# ===========================================================================
# gregorian_to_julian — reference values from Meeus Table 7.a
# ===========================================================================

class TestGregorianToJulian:
    """Cross-check against values from Jean Meeus, *Astronomical Algorithms*,
    Table 7.a (p. 62)."""

    @pytest.mark.parametrize("year, month, day, expected_jd", [
        (2000,  1,  1.5,  2_451_545.0),      # J2000.0
        (1999,  1,  1.0,  2_451_179.5),
        (1987,  1, 27.0,  2_446_822.5),
        (1987,  6, 19.5,  2_446_966.0),
        (1988,  1, 27.0,  2_447_187.5),
        (1988,  6, 19.5,  2_447_332.0),
        (1900,  1,  1.0,  2_415_020.5),
        (1600,  1,  1.0,  2_305_447.5),
        (1600, 12, 31.0,  2_305_812.5),
        (  -4713, 11, 24.5,      0.0),   # Julian Day zero
    ])
    def test_known_values(
        self, year: int, month: int, day: float, expected_jd: float
    ) -> None:
        result = gregorian_to_julian(year, month, day)
        assert result == pytest.approx(expected_jd, abs=1e-5)

    def test_unix_epoch(self, unix_epoch_jd: float) -> None:
        result = gregorian_to_julian(1970, 1, 1.0)
        assert result == pytest.approx(unix_epoch_jd, abs=1e-5)

    def test_fractional_day_noon(self) -> None:
        """Day 1.5 means noon on the 1st (0.5 days = 12 h past midnight)."""
        jd_midnight = gregorian_to_julian(2000, 1, 1.0)
        jd_noon = gregorian_to_julian(2000, 1, 1.5)
        assert jd_noon - jd_midnight == pytest.approx(0.5, abs=1e-10)


# ===========================================================================
# julian_to_gregorian — round-trip and reference values
# ===========================================================================

class TestJulianToGregorian:
    @pytest.mark.parametrize("jd, expected_year, expected_month, expected_day", [
        (2_451_545.0, 2000,  1,  1.5),
        (2_446_822.5, 1987,  1, 27.0),
        (2_305_812.5, 1600, 12, 31.0),
    ])
    def test_known_values(
        self,
        jd: float,
        expected_year: int,
        expected_month: int,
        expected_day: float,
    ) -> None:
        year, month, day = julian_to_gregorian(jd)
        assert year == expected_year
        assert month == expected_month
        assert day == pytest.approx(expected_day, abs=1e-5)

    @pytest.mark.parametrize("year, month, day", [
        (2000, 1, 1.5),
        (1987, 6, 19.5),
        (2024, 3, 20.0),
        (1900, 1, 1.0),
        (2100, 12, 31.75),
    ])
    def test_round_trip(self, year: int, month: int, day: float) -> None:
        """gregorian_to_julian → julian_to_gregorian must recover input."""
        jd = gregorian_to_julian(year, month, day)
        y2, m2, d2 = julian_to_gregorian(jd)
        assert y2 == year
        assert m2 == month
        assert d2 == pytest.approx(day, abs=1e-5)


# ===========================================================================
# datetime ↔ Julian Date
# ===========================================================================

class TestDatetimeJulian:
    def test_datetime_to_julian_j2000(self) -> None:
        dt = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        jd = datetime_to_julian(dt)
        assert jd == pytest.approx(2_451_545.0, abs=1e-5)

    def test_datetime_to_julian_unix_epoch(self, unix_epoch_jd: float) -> None:
        dt = datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        jd = datetime_to_julian(dt)
        assert jd == pytest.approx(unix_epoch_jd, abs=1e-5)

    def test_julian_to_datetime_j2000(self) -> None:
        dt = julian_to_datetime(2_451_545.0)
        assert dt.year == 2000
        assert dt.month == 1
        assert dt.day == 1
        assert dt.hour == 12
        assert dt.minute == 0

    def test_julian_to_datetime_timezone_utc(self) -> None:
        dt = julian_to_datetime(2_451_545.0)
        assert dt.tzinfo == timezone.utc

    @pytest.mark.parametrize("dt_input", [
        datetime(2024, 5, 9, 0, 0, 0, tzinfo=timezone.utc),
        datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        datetime(1990, 6, 15, 6, 30, 0, tzinfo=timezone.utc),
        datetime(2023, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
    ])
    def test_round_trip_datetime(self, dt_input: datetime) -> None:
        """datetime → JD → datetime must agree to within 1 second."""
        jd = datetime_to_julian(dt_input)
        dt_out = julian_to_datetime(jd)
        delta_seconds = abs(
            (dt_out.replace(tzinfo=None) - dt_input.replace(tzinfo=None)).total_seconds()
        )
        assert delta_seconds < 1.0, (
            f"Round-trip failed for {dt_input}: delta = {delta_seconds:.3f}s"
        )


# ===========================================================================
# Local Sidereal Time
# ===========================================================================

class TestLocalSiderealTime:
    def test_greenwich_lst_j2000(self, j2000_epoch: float) -> None:
        """At J2000.0 (JD 2451545.0, 2000 Jan 1.5), GMST ≈ 280.46°."""
        lst = local_sidereal_time(j2000_epoch, longitude_deg=0.0)
        # USNO formula: 280.460618 + 360.985647 * 0 (days from J2000.0) = 280.46°
        assert lst == pytest.approx(280.46, abs=0.01)

    def test_lst_longitude_offset(self, j2000_epoch: float) -> None:
        """LST at 90°E should be GMST + 90°."""
        gmst = local_sidereal_time(j2000_epoch, longitude_deg=0.0)
        lst_east = local_sidereal_time(j2000_epoch, longitude_deg=90.0)
        diff = (lst_east - gmst) % 360.0
        assert diff == pytest.approx(90.0, abs=1e-6)

    def test_lst_in_valid_range(self) -> None:
        for jd in [2_400_000.5, 2_451_545.0, 2_460_000.0]:
            lst = local_sidereal_time(jd, longitude_deg=0.0)
            assert 0.0 <= lst < 360.0


# ===========================================================================
# Julian centuries
# ===========================================================================

class TestJulianCenturies:
    def test_at_j2000(self, j2000_epoch: float) -> None:
        assert julian_centuries_j2000(j2000_epoch) == pytest.approx(0.0)

    def test_one_century_after_j2000(self, j2000_epoch: float) -> None:
        assert julian_centuries_j2000(j2000_epoch + 36_525.0) == pytest.approx(1.0)

    def test_before_j2000(self, j2000_epoch: float) -> None:
        T = julian_centuries_j2000(j2000_epoch - 36_525.0)
        assert T == pytest.approx(-1.0)
