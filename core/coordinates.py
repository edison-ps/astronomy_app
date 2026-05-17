"""Astronomical coordinate system transformations.

Implements transformations between the principal coordinate systems
used in observational and positional astronomy:

    * Equatorial  (Right Ascension, Declination)
    * Horizontal  (Altitude, Azimuth)
    * Ecliptic    (longitude, latitude)
    * Galactic    (l, b)

All angles are in decimal degrees unless stated otherwise.
Reference epoch: J2000.0 for constants that are epoch-dependent.
"""

import math
from dataclasses import dataclass

from utils.conversions import degrees_to_radians, radians_to_degrees

# ---------------------------------------------------------------------------
# Epoch-dependent constants  (J2000.0)
# ---------------------------------------------------------------------------

OBLIQUITY_J2000: float = 23.439_291_111  # degrees – mean obliquity of ecliptic

# IAU galactic north-pole position in equatorial J2000.0
_RA_NGP: float = 192.859_508    # degrees
_DEC_NGP: float = 27.128_336    # degrees
_L_ASCENDING: float = 122.932   # degrees – galactic longitude of ascending node


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EquatorialCoord:
    """Equatorial coordinate pair (RA, Dec) in decimal degrees."""

    ra: float   # Right Ascension  [0, 360)  degrees
    dec: float  # Declination      [-90, 90] degrees

    def __post_init__(self) -> None:
        if not (0.0 <= self.ra < 360.0):
            raise ValueError(f"RA must be in [0, 360), got {self.ra}")
        if not (-90.0 <= self.dec <= 90.0):
            raise ValueError(f"Dec must be in [-90, 90], got {self.dec}")


@dataclass
class HorizontalCoord:
    """Horizontal coordinate pair (Altitude, Azimuth) in decimal degrees."""

    altitude: float  # [-90,  90]  degrees
    azimuth: float   # [  0, 360)  degrees


@dataclass
class EclipticCoord:
    """Ecliptic coordinate pair (longitude, latitude) in decimal degrees."""

    longitude: float  # [0, 360)  degrees
    latitude: float   # [-90, 90] degrees


@dataclass
class GalacticCoord:
    """Galactic coordinate pair (l, b) in decimal degrees."""

    l: float  # Galactic longitude  [0, 360)  degrees
    b: float  # Galactic latitude   [-90, 90] degrees


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------

def equatorial_to_horizontal(
    ra: float,
    dec: float,
    latitude: float,
    longitude: float,      # geographic longitude – included for API clarity
    lst: float,
) -> HorizontalCoord:
    """Convert equatorial coordinates to topocentric horizontal (Alt/Az).

    The Local Sidereal Time (LST) encodes both observer longitude and UT,
    so *longitude* is accepted for clarity but is not used in the formula.

    Args:
        ra:        Right Ascension in degrees.
        dec:       Declination in degrees.
        latitude:  Observer's geographic latitude in degrees.
        longitude: Observer's geographic longitude in degrees (informational).
        lst:       Local Sidereal Time in degrees.

    Returns:
        HorizontalCoord with altitude and azimuth in degrees.
    """
    ha_rad = degrees_to_radians((lst - ra) % 360.0)
    dec_rad = degrees_to_radians(dec)
    lat_rad = degrees_to_radians(latitude)

    sin_alt = (
        math.sin(dec_rad) * math.sin(lat_rad)
        + math.cos(dec_rad) * math.cos(lat_rad) * math.cos(ha_rad)
    )
    altitude = radians_to_degrees(math.asin(max(-1.0, min(1.0, sin_alt))))

    cos_az_num = math.sin(dec_rad) - math.sin(lat_rad) * sin_alt
    cos_az_den = math.cos(lat_rad) * math.cos(degrees_to_radians(altitude))

    # Guard against division by zero near the zenith/nadir
    if abs(cos_az_den) < 1e-12:
        azimuth = 0.0
    else:
        cos_az = max(-1.0, min(1.0, cos_az_num / cos_az_den))
        azimuth = radians_to_degrees(math.acos(cos_az))
        if math.sin(ha_rad) > 0.0:
            azimuth = 360.0 - azimuth

    azimuth = azimuth % 360.0
    return HorizontalCoord(altitude=altitude, azimuth=azimuth)


def equatorial_to_ecliptic(
    ra: float,
    dec: float,
    obliquity: float = OBLIQUITY_J2000,
) -> EclipticCoord:
    """Convert equatorial to ecliptic coordinates.

    Args:
        ra:         Right Ascension in degrees.
        dec:        Declination in degrees.
        obliquity:  Obliquity of the ecliptic in degrees (default J2000.0).

    Returns:
        EclipticCoord with ecliptic longitude and latitude.
    """
    ra_r = degrees_to_radians(ra)
    dec_r = degrees_to_radians(dec)
    eps_r = degrees_to_radians(obliquity)

    sin_beta = (
        math.sin(dec_r) * math.cos(eps_r)
        - math.cos(dec_r) * math.sin(eps_r) * math.sin(ra_r)
    )
    beta = radians_to_degrees(math.asin(max(-1.0, min(1.0, sin_beta))))

    y = math.sin(ra_r) * math.cos(eps_r) + math.tan(dec_r) * math.sin(eps_r)
    x = math.cos(ra_r)
    lam = radians_to_degrees(math.atan2(y, x)) % 360.0

    return EclipticCoord(longitude=lam, latitude=beta)


def ecliptic_to_equatorial(
    longitude: float,
    latitude: float,
    obliquity: float = OBLIQUITY_J2000,
) -> EquatorialCoord:
    """Convert ecliptic to equatorial coordinates.

    Args:
        longitude:  Ecliptic longitude in degrees.
        latitude:   Ecliptic latitude in degrees.
        obliquity:  Obliquity of the ecliptic in degrees (default J2000.0).

    Returns:
        EquatorialCoord with RA and Dec.
    """
    lam_r = degrees_to_radians(longitude)
    beta_r = degrees_to_radians(latitude)
    eps_r = degrees_to_radians(obliquity)

    sin_dec = (
        math.sin(beta_r) * math.cos(eps_r)
        + math.cos(beta_r) * math.sin(eps_r) * math.sin(lam_r)
    )
    dec = radians_to_degrees(math.asin(max(-1.0, min(1.0, sin_dec))))

    y = math.sin(lam_r) * math.cos(eps_r) - math.tan(beta_r) * math.sin(eps_r)
    x = math.cos(lam_r)
    ra = radians_to_degrees(math.atan2(y, x)) % 360.0

    return EquatorialCoord(ra=ra, dec=dec)


def equatorial_to_galactic(ra: float, dec: float) -> GalacticCoord:
    """Convert equatorial (J2000.0) to galactic coordinates.

    Uses the IAU-defined galactic north pole and zero-longitude direction
    (Liu et al. 2011 / original IAU 1958 definition in J2000.0 frame).

    Args:
        ra:  Right Ascension in degrees.
        dec: Declination in degrees.

    Returns:
        GalacticCoord with galactic longitude *l* and latitude *b*.
    """
    ra_r = degrees_to_radians(ra)
    dec_r = degrees_to_radians(dec)
    ngp_ra_r = degrees_to_radians(_RA_NGP)
    ngp_dec_r = degrees_to_radians(_DEC_NGP)

    sin_b = (
        math.sin(dec_r) * math.sin(ngp_dec_r)
        + math.cos(dec_r) * math.cos(ngp_dec_r) * math.cos(ra_r - ngp_ra_r)
    )
    b = radians_to_degrees(math.asin(max(-1.0, min(1.0, sin_b))))

    x = math.cos(dec_r) * math.sin(ra_r - ngp_ra_r)
    y = math.sin(dec_r) * math.cos(ngp_dec_r) - math.cos(dec_r) * math.sin(ngp_dec_r) * math.cos(ra_r - ngp_ra_r)
    l = (degrees_to_radians(_L_ASCENDING) - math.atan2(x, y))
    l = radians_to_degrees(l) % 360.0

    return GalacticCoord(l=l, b=b)


def angular_separation(
    ra1: float, dec1: float,
    ra2: float, dec2: float,
) -> float:
    """Calculate the angular separation between two celestial coordinates.

    Uses the Vincenty formula which is numerically stable for all angular
    separations, including very small and very large ones.

    Args:
        ra1, dec1: Coordinates of the first point in degrees.
        ra2, dec2: Coordinates of the second point in degrees.

    Returns:
        Angular separation in degrees [0, 180].
    """
    ra1_r = degrees_to_radians(ra1)
    ra2_r = degrees_to_radians(ra2)
    dec1_r = degrees_to_radians(dec1)
    dec2_r = degrees_to_radians(dec2)
    delta_ra = ra2_r - ra1_r

    numerator = math.sqrt(
        (math.cos(dec2_r) * math.sin(delta_ra)) ** 2
        + (math.cos(dec1_r) * math.sin(dec2_r)
           - math.sin(dec1_r) * math.cos(dec2_r) * math.cos(delta_ra)) ** 2
    )
    denominator = (
        math.sin(dec1_r) * math.sin(dec2_r)
        + math.cos(dec1_r) * math.cos(dec2_r) * math.cos(delta_ra)
    )
    return radians_to_degrees(math.atan2(numerator, denominator))
