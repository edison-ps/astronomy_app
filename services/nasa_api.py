"""NASA public API client.

Provides a clean, injectable interface to several NASA open data endpoints:

    * APOD  — Astronomy Picture of the Day
    * NeoWs — Near Earth Object Web Service (asteroid close approaches)

The :class:`NASAAPIClient` depends on an injected :class:`requests.Session`,
making it trivially testable with ``unittest.mock`` without any real HTTP
traffic.

Example usage::

    client = NASAAPIClient(api_key="YOUR_KEY")
    apod = client.get_apod("2024-01-01")
    print(apod.title)

    neos = client.get_near_earth_objects("2024-01-01", "2024-01-07")
    hazardous = [n for n in neos if n.is_hazardous]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class NASAAPIError(Exception):
    """Base exception for NASA API errors."""


class NASAAPIConnectionError(NASAAPIError):
    """Raised when the HTTP connection to NASA's servers fails."""


class NASAAPIHTTPError(NASAAPIError):
    """Raised when the API returns a non-2xx HTTP status code."""


class NASAAPIParseError(NASAAPIError):
    """Raised when the API response cannot be parsed into a data object."""


# ---------------------------------------------------------------------------
# Response data classes
# ---------------------------------------------------------------------------

@dataclass
class ApodData:
    """Astronomy Picture of the Day payload."""

    date: str
    title: str
    explanation: str
    url: str
    media_type: str
    hdurl: Optional[str] = None
    copyright: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApodData":
        """Construct an :class:`ApodData` from a raw API response dict.

        Raises:
            NASAAPIParseError: If required keys are missing.
        """
        required = ("date", "title", "explanation", "url", "media_type")
        missing = [k for k in required if k not in data]
        if missing:
            raise NASAAPIParseError(
                f"APOD response missing required fields: {missing}"
            )
        return cls(
            date=data["date"],
            title=data["title"],
            explanation=data["explanation"],
            url=data["url"],
            media_type=data["media_type"],
            hdurl=data.get("hdurl"),
            copyright=data.get("copyright"),
        )


@dataclass
class NearEarthObject:
    """Near Earth Object (asteroid) from the NeoWs feed."""

    id: str
    name: str
    absolute_magnitude: float
    diameter_min_km: float
    diameter_max_km: float
    is_hazardous: bool
    close_approach_date: str
    miss_distance_km: float
    relative_velocity_kmh: float

    @property
    def mean_diameter_km(self) -> float:
        """Arithmetic mean of the estimated diameter range."""
        return (self.diameter_min_km + self.diameter_max_km) / 2.0

    @classmethod
    def from_dict(cls, obj: Dict[str, Any], approach_idx: int = 0) -> "NearEarthObject":
        """Construct a :class:`NearEarthObject` from a NeoWs API dict.

        Raises:
            NASAAPIParseError: If expected keys are missing.
        """
        try:
            ca = obj["close_approach_data"][approach_idx]
            diam = obj["estimated_diameter"]["kilometers"]
            return cls(
                id=obj["id"],
                name=obj["name"],
                absolute_magnitude=float(obj["absolute_magnitude_h"]),
                diameter_min_km=float(diam["estimated_diameter_min"]),
                diameter_max_km=float(diam["estimated_diameter_max"]),
                is_hazardous=bool(obj["is_potentially_hazardous_asteroid"]),
                close_approach_date=ca["close_approach_date"],
                miss_distance_km=float(ca["miss_distance"]["kilometers"]),
                relative_velocity_kmh=float(
                    ca["relative_velocity"]["kilometers_per_hour"]
                ),
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise NASAAPIParseError(
                f"Failed to parse NearEarthObject: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

class NASAAPIClient:
    """HTTP client for NASA's public REST APIs.

    The optional *session* parameter enables dependency injection:
    pass a :class:`unittest.mock.Mock` or a :class:`requests.Session`
    configured with an adapter for deterministic tests.

    Args:
        api_key: NASA API key (defaults to the public ``DEMO_KEY``).
        session: Optional pre-configured :class:`requests.Session`.
    """

    BASE_URL: str = "https://api.nasa.gov"
    DEMO_KEY: str = "DEMO_KEY"

    def __init__(
        self,
        api_key: str = DEMO_KEY,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.api_key = api_key
        self._session: requests.Session = session or requests.Session()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: int = 10,
    ) -> Dict[str, Any]:
        """Execute a GET request and return the parsed JSON body.

        Args:
            endpoint: URL path relative to :attr:`BASE_URL`.
            params:   Query parameters (``api_key`` is appended automatically).
            timeout:  Request timeout in seconds.

        Returns:
            Parsed JSON response as a dict.

        Raises:
            NASAAPIConnectionError: On network-level errors.
            NASAAPIHTTPError:       On HTTP error status codes.
        """
        url = f"{self.BASE_URL}{endpoint}"
        query = dict(params or {})
        query["api_key"] = self.api_key

        try:
            response = self._session.get(url, params=query, timeout=timeout)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.ConnectionError as exc:
            raise NASAAPIConnectionError(
                f"Could not connect to NASA API: {exc}"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise NASAAPIConnectionError(
                f"NASA API request timed out after {timeout}s."
            ) from exc
        except requests.exceptions.HTTPError as exc:
            raise NASAAPIHTTPError(
                f"NASA API returned HTTP {response.status_code}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    def get_apod(self, apod_date: Optional[str] = None) -> ApodData:
        """Fetch the Astronomy Picture of the Day.

        Args:
            apod_date: Date string ``'YYYY-MM-DD'``.  Defaults to today.

        Returns:
            :class:`ApodData` instance.
        """
        params: Dict[str, Any] = {}
        if apod_date:
            params["date"] = apod_date

        data = self._get("/planetary/apod", params)
        return ApodData.from_dict(data)

    def get_near_earth_objects(
        self,
        start_date: str,
        end_date: str,
    ) -> List[NearEarthObject]:
        """Fetch Near Earth Objects from the NeoWs feed.

        Args:
            start_date: Start date ``'YYYY-MM-DD'`` (max 7-day window).
            end_date:   End date ``'YYYY-MM-DD'``.

        Returns:
            List of :class:`NearEarthObject` instances.
        """
        data = self._get(
            "/neo/rest/v1/feed",
            {"start_date": start_date, "end_date": end_date},
        )

        neos: List[NearEarthObject] = []
        for _date, objects in data.get("near_earth_objects", {}).items():
            for obj in objects:
                neos.append(NearEarthObject.from_dict(obj))

        return neos

    def get_asteroid_by_id(self, asteroid_id: str) -> Dict[str, Any]:
        """Fetch detailed data for a specific asteroid by NeoWs ID.

        Args:
            asteroid_id: NASA NeoWs asteroid ID string.

        Returns:
            Raw response dict (structure varies by object).
        """
        return self._get(f"/neo/rest/v1/neo/{asteroid_id}")
