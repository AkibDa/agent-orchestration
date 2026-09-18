import pytest
from datetime import datetime, timezone
import json
from unittest.mock import patch, MagicMock

from agents.tide.agent import TideAgent
from schemas.contracts import QueryPlan, GeoLocation, AgentResult, QueryTime

@pytest.fixture
def mock_open_meteo_response():
    # Provide a mock response with a clear high and low tide
    data = {
        "utc_offset_seconds": 0,
        "hourly": {
            "time": [
                "2026-09-18T00:00",
                "2026-09-18T01:00",
                "2026-09-18T02:00",
                "2026-09-18T03:00",
                "2026-09-18T04:00",
                "2026-09-18T05:00",
                "2026-09-18T06:00",
                "2026-09-18T07:00"
            ],
            "sea_level_height_msl": [
                0.2, 
                0.5, 
                0.8, # Local max (High tide at 02:00)
                0.6, 
                0.3, 
                0.1, # Local min (Low tide at 05:00)
                0.4, 
                0.7
            ]
        }
    }
    return json.dumps(data).encode('utf-8')

@patch("urllib.request.urlopen")
def test_tide_agent_success(mock_urlopen, mock_open_meteo_response):
    # Setup mock
    cm = MagicMock()
    cm.getcode.return_value = 200
    cm.status = 200
    cm.read.return_value = mock_open_meteo_response
    cm.__enter__.return_value = cm
    mock_urlopen.return_value = cm

    agent = TideAgent()
    loc = GeoLocation(latitude=9.93, longitude=76.26, name="Kochi")
    # Plan target time at 00:00
    target_time = datetime(2026, 9, 18, 0, 0, tzinfo=timezone.utc)
    plan = QueryPlan(
        query="dummy",
        intent="marine_conditions",
        location=loc,
        time=QueryTime(exact=target_time),
        language="en"
    )

    result = agent.run(plan, {})

    assert result.status == "AVAILABLE"
    assert result.sources == ["OPEN_METEO"]
    assert "OPEN_METEO" in result.audit.sources
    assert result.audit.score_source == "REAL_EXTERNAL_DATA"
    assert "REAL_EXTERNAL_DATA" in result.audit.score_reason
    assert "ORCA_DERIVED" in result.audit.score_reason

    data = result.data
    assert data["current_height_m"] == 0.2
    assert data["current_phase"] == "rising"
    assert data["next_high"]["time"] == "02:00"
    assert data["next_high"]["height_m"] == 0.8
    assert data["next_low"]["time"] == "05:00"
    assert data["next_low"]["height_m"] == 0.1

@patch("urllib.request.urlopen")
def test_tide_agent_failure(mock_urlopen):
    # Setup mock to simulate failure
    mock_urlopen.side_effect = Exception("API Down")

    agent = TideAgent()
    loc = GeoLocation(latitude=9.93, longitude=76.26, name="Kochi")
    plan = QueryPlan(
        query="dummy",
        intent="marine_conditions",
        location=loc,
        language="en"
    )

    result = agent.run(plan, {})

    assert result.status == "DEGRADED"
    assert result.data["current_height_m"] is None
    assert result.data["next_high"]["time"] is None
    assert result.data["next_low"]["time"] is None
    assert result.confidence == 0.0

# Integration test against the real API (Requirement A, C, D, E)
def test_tide_agent_real_api():
    agent = TideAgent()
    loc = GeoLocation(latitude=9.93, longitude=76.26, name="Kochi")
    plan = QueryPlan(
        query="dummy",
        intent="marine_conditions",
        location=loc,
        language="en"
    )

    result = agent.run(plan, {})

    # Ensure it succeeds or fails cleanly, no mocked random values
    assert result.status in ["AVAILABLE", "DEGRADED"]
    
    if result.status == "AVAILABLE":
        assert result.data["current_height_m"] is not None
        # It must be a float
        assert isinstance(result.data["current_height_m"], float)
        assert result.sources == ["OPEN_METEO"]
