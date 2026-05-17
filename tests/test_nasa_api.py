"""Unit tests for services.nasa_api.

Demonstrates:
    * Dependency injection via session parameter for pure unit tests
    * unittest.mock.Mock and MagicMock for HTTP session mocking
    * pytest fixtures for shared mock infrastructure
    * Testing error handling paths (connection errors, HTTP errors, parse errors)
    * Verifying correct URL construction and parameter passing
    * No real HTTP requests are made in any test
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
import requests

from services.nasa_api import (
    ApodData,
    NASAAPIClient,
    NASAAPIConnectionError,
    NASAAPIHTTPError,
    NASAAPIParseError,
    NearEarthObject,
)


# ===========================================================================
# Mock response builders
# ===========================================================================

def _make_response(json_data: dict, status_code: int = 200) -> Mock:
    """Create a mock requests.Response."""
    response = Mock(spec=requests.Response)
    response.status_code = status_code
    response.json.return_value = json_data
    if status_code >= 400:
        response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=response
        )
    else:
        response.raise_for_status.return_value = None
    return response


def _apod_payload(date: str = "2024-01-01") -> dict:
    return {
        "date": date,
        "title": "Andromeda Galaxy",
        "explanation": "The nearest major galaxy to the Milky Way.",
        "url": "https://apod.nasa.gov/apod/image/2401/Andromeda.jpg",
        "hdurl": "https://apod.nasa.gov/apod/image/2401/Andromeda_hd.jpg",
        "media_type": "image",
        "copyright": "Some Photographer",
    }


def _neo_feed_payload() -> dict:
    return {
        "near_earth_objects": {
            "2024-01-01": [
                {
                    "id": "2000433",
                    "name": "(433) Eros",
                    "absolute_magnitude_h": 10.8,
                    "estimated_diameter": {
                        "kilometers": {
                            "estimated_diameter_min": 10.21,
                            "estimated_diameter_max": 22.84,
                        }
                    },
                    "is_potentially_hazardous_asteroid": False,
                    "close_approach_data": [
                        {
                            "close_approach_date": "2024-01-01",
                            "miss_distance": {"kilometers": "17550000.0"},
                            "relative_velocity": {"kilometers_per_hour": "45000.0"},
                        }
                    ],
                }
            ]
        }
    }


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def mock_session() -> MagicMock:
    """A mock requests.Session that returns a 200 OK by default."""
    session = MagicMock(spec=requests.Session)
    return session


@pytest.fixture
def client(mock_session: MagicMock) -> NASAAPIClient:
    """NASAAPIClient wired with a mock session."""
    return NASAAPIClient(api_key="TEST_KEY", session=mock_session)


# ===========================================================================
# ApodData.from_dict
# ===========================================================================

class TestApodDataFromDict:
    def test_valid_payload(self) -> None:
        data = _apod_payload()
        apod = ApodData.from_dict(data)
        assert apod.title == "Andromeda Galaxy"
        assert apod.date == "2024-01-01"
        assert apod.media_type == "image"

    def test_optional_hdurl_present(self) -> None:
        apod = ApodData.from_dict(_apod_payload())
        assert apod.hdurl is not None

    def test_optional_copyright_present(self) -> None:
        apod = ApodData.from_dict(_apod_payload())
        assert apod.copyright == "Some Photographer"

    def test_missing_required_field_raises(self) -> None:
        payload = _apod_payload()
        del payload["title"]
        with pytest.raises(NASAAPIParseError, match="title"):
            ApodData.from_dict(payload)

    def test_missing_multiple_fields_raises(self) -> None:
        with pytest.raises(NASAAPIParseError):
            ApodData.from_dict({})


# ===========================================================================
# NearEarthObject.from_dict
# ===========================================================================

class TestNearEarthObjectFromDict:
    def test_valid_neo(self) -> None:
        obj = _neo_feed_payload()["near_earth_objects"]["2024-01-01"][0]
        neo = NearEarthObject.from_dict(obj)
        assert neo.name == "(433) Eros"
        assert neo.is_hazardous is False
        assert neo.miss_distance_km == pytest.approx(17_550_000.0)

    def test_mean_diameter_computed(self) -> None:
        obj = _neo_feed_payload()["near_earth_objects"]["2024-01-01"][0]
        neo = NearEarthObject.from_dict(obj)
        expected = (neo.diameter_min_km + neo.diameter_max_km) / 2.0
        assert neo.mean_diameter_km == pytest.approx(expected)

    def test_missing_close_approach_raises(self) -> None:
        obj = _neo_feed_payload()["near_earth_objects"]["2024-01-01"][0]
        del obj["close_approach_data"]
        with pytest.raises(NASAAPIParseError):
            NearEarthObject.from_dict(obj)

    def test_missing_diameter_raises(self) -> None:
        obj = _neo_feed_payload()["near_earth_objects"]["2024-01-01"][0]
        del obj["estimated_diameter"]
        with pytest.raises(NASAAPIParseError):
            NearEarthObject.from_dict(obj)


# ===========================================================================
# NASAAPIClient.get_apod
# ===========================================================================

class TestGetApod:
    def test_returns_apod_data(
        self, client: NASAAPIClient, mock_session: MagicMock
    ) -> None:
        mock_session.get.return_value = _make_response(_apod_payload("2024-01-15"))
        result = client.get_apod("2024-01-15")
        assert isinstance(result, ApodData)
        assert result.date == "2024-01-15"

    def test_correct_endpoint_called(
        self, client: NASAAPIClient, mock_session: MagicMock
    ) -> None:
        mock_session.get.return_value = _make_response(_apod_payload())
        client.get_apod("2024-01-01")
        call_args = mock_session.get.call_args
        assert "/planetary/apod" in call_args[0][0]

    def test_api_key_sent_in_params(
        self, client: NASAAPIClient, mock_session: MagicMock
    ) -> None:
        mock_session.get.return_value = _make_response(_apod_payload())
        client.get_apod("2024-01-01")
        params = mock_session.get.call_args[1]["params"]
        assert params.get("api_key") == "TEST_KEY"

    def test_date_param_sent_when_provided(
        self, client: NASAAPIClient, mock_session: MagicMock
    ) -> None:
        mock_session.get.return_value = _make_response(_apod_payload("2024-06-15"))
        client.get_apod("2024-06-15")
        params = mock_session.get.call_args[1]["params"]
        assert params.get("date") == "2024-06-15"

    def test_no_date_param_when_omitted(
        self, client: NASAAPIClient, mock_session: MagicMock
    ) -> None:
        mock_session.get.return_value = _make_response(_apod_payload())
        client.get_apod()
        params = mock_session.get.call_args[1]["params"]
        assert "date" not in params


# ===========================================================================
# NASAAPIClient.get_near_earth_objects
# ===========================================================================

class TestGetNearEarthObjects:
    def test_returns_list_of_neos(
        self, client: NASAAPIClient, mock_session: MagicMock
    ) -> None:
        mock_session.get.return_value = _make_response(_neo_feed_payload())
        result = client.get_near_earth_objects("2024-01-01", "2024-01-07")
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], NearEarthObject)

    def test_neo_fields_populated(
        self, client: NASAAPIClient, mock_session: MagicMock
    ) -> None:
        mock_session.get.return_value = _make_response(_neo_feed_payload())
        neos = client.get_near_earth_objects("2024-01-01", "2024-01-07")
        neo = neos[0]
        assert neo.id == "2000433"
        assert neo.absolute_magnitude == pytest.approx(10.8)
        assert neo.relative_velocity_kmh == pytest.approx(45_000.0)

    def test_empty_feed_returns_empty_list(
        self, client: NASAAPIClient, mock_session: MagicMock
    ) -> None:
        mock_session.get.return_value = _make_response({"near_earth_objects": {}})
        result = client.get_near_earth_objects("2024-01-01", "2024-01-01")
        assert result == []

    def test_date_params_sent(
        self, client: NASAAPIClient, mock_session: MagicMock
    ) -> None:
        mock_session.get.return_value = _make_response(_neo_feed_payload())
        client.get_near_earth_objects("2024-01-01", "2024-01-07")
        params = mock_session.get.call_args[1]["params"]
        assert params["start_date"] == "2024-01-01"
        assert params["end_date"] == "2024-01-07"

    def test_multiple_dates_aggregated(
        self, client: NASAAPIClient, mock_session: MagicMock
    ) -> None:
        """NEOs from multiple date keys should all be returned."""
        payload = {
            "near_earth_objects": {
                "2024-01-01": [
                    _neo_feed_payload()["near_earth_objects"]["2024-01-01"][0]
                ],
                "2024-01-02": [
                    {**_neo_feed_payload()["near_earth_objects"]["2024-01-01"][0],
                     "id": "9999999", "name": "Other Asteroid"}
                ],
            }
        }
        mock_session.get.return_value = _make_response(payload)
        result = client.get_near_earth_objects("2024-01-01", "2024-01-02")
        assert len(result) == 2


# ===========================================================================
# Error handling
# ===========================================================================

class TestAPIErrorHandling:
    def test_connection_error_raises_api_connection_error(
        self, client: NASAAPIClient, mock_session: MagicMock
    ) -> None:
        mock_session.get.side_effect = requests.exceptions.ConnectionError("no route")
        with pytest.raises(NASAAPIConnectionError, match="connect"):
            client.get_apod()

    def test_timeout_raises_api_connection_error(
        self, client: NASAAPIClient, mock_session: MagicMock
    ) -> None:
        mock_session.get.side_effect = requests.exceptions.Timeout()
        with pytest.raises(NASAAPIConnectionError, match="timed out"):
            client.get_apod()

    def test_http_403_raises_api_http_error(
        self, client: NASAAPIClient, mock_session: MagicMock
    ) -> None:
        mock_session.get.return_value = _make_response({}, status_code=403)
        with pytest.raises(NASAAPIHTTPError):
            client.get_apod()

    def test_http_429_too_many_requests(
        self, client: NASAAPIClient, mock_session: MagicMock
    ) -> None:
        mock_session.get.return_value = _make_response({}, status_code=429)
        with pytest.raises(NASAAPIHTTPError):
            client.get_near_earth_objects("2024-01-01", "2024-01-07")

    def test_http_500_raises_api_http_error(
        self, client: NASAAPIClient, mock_session: MagicMock
    ) -> None:
        mock_session.get.return_value = _make_response({}, status_code=500)
        with pytest.raises(NASAAPIHTTPError):
            client.get_apod()

    def test_malformed_apod_response_raises_parse_error(
        self, client: NASAAPIClient, mock_session: MagicMock
    ) -> None:
        """API returning unexpected JSON should raise NASAAPIParseError."""
        mock_session.get.return_value = _make_response({"unexpected": "data"})
        with pytest.raises(NASAAPIParseError):
            client.get_apod()


# ===========================================================================
# Client configuration
# ===========================================================================

class TestClientConfiguration:
    def test_demo_key_used_by_default(self) -> None:
        client = NASAAPIClient()
        assert client.api_key == NASAAPIClient.DEMO_KEY

    def test_custom_api_key_stored(self) -> None:
        client = NASAAPIClient(api_key="MY_KEY")
        assert client.api_key == "MY_KEY"

    def test_session_injected(self) -> None:
        session = MagicMock(spec=requests.Session)
        client = NASAAPIClient(session=session)
        assert client._session is session

    def test_default_session_created_when_not_injected(self) -> None:
        client = NASAAPIClient()
        assert isinstance(client._session, requests.Session)
