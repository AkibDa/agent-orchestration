# data_sources/weather_provider.py

import logging
import time
import concurrent.futures
from datetime import datetime, timezone
from schemas.contracts import UNKNOWN
from schemas.data_catalog import DATA_CATALOG

from config import IMD_ENABLED, IMD_DISABLED_REASON

import data_sources.imd as imd
import data_sources.open_meteo as open_meteo
import data_sources.incois as incois
import data_sources.mosdac as mosdac
import data_sources.cyclone as cyclone

logger = logging.getLogger(__name__)

IMD_AVAILABLE = True
LAST_IMD_FAILURE_TIME = 0.0
CIRCUIT_BREAKER_DURATION_SEC = 300.0

def _check_circuit_breaker():
    global IMD_AVAILABLE, LAST_IMD_FAILURE_TIME
    if not IMD_AVAILABLE:
        if time.time() - LAST_IMD_FAILURE_TIME > CIRCUIT_BREAKER_DURATION_SEC:
            IMD_AVAILABLE = True
        else:
            raise Exception("IMD API Circuit Breaker is active.")

def _record_failure():
    global IMD_AVAILABLE, LAST_IMD_FAILURE_TIME
    IMD_AVAILABLE = False
    LAST_IMD_FAILURE_TIME = time.time()

def get_weather_context(location: str, intent: str, lat: float = None, lon: float = None, time_left: float = 3.0) -> dict:
    """
    Facade for fetching weather and marine warnings.
    Walks DATA_CATALOG (primary -> fallback).
    """
    needs_warnings = intent in ["marine_safety", "hazard_alert", "pfz_search"]
    context = {}

    # Initialize lat/lon if not provided
    if lat is None or lon is None:
        lat, lon = open_meteo._get_coords(location)

    provenance = []

    if not IMD_ENABLED:
        provenance.append({"source": "IMD", "status": IMD_DISABLED_REASON, "reason": "Institutional sign-off pending", "latency_ms": 0})

        # Open-Meteo as primary
        t0 = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            om_future = executor.submit(open_meteo.get_primary_weather, lat, lon, time_left)
            cyclone_future = executor.submit(cyclone.get_active_cyclones, lat, lon) if needs_warnings else None
            
            primary = om_future.result()
            cyclone_data = cyclone_future.result() if cyclone_future else None

        om_latency = int((time.time() - t0) * 1000)

        context["weather"] = primary["weather"]
        context["coastal_bulletin"] = primary["coastal_bulletin"]
        provenance.append({"source": "OPEN_METEO", "status": "PRIMARY", "fields": ["wind", "pressure", "temperature", "precipitation"], "latency_ms": om_latency})

        from schemas.contracts import WarningContract
        context["official_marine_warning"] = WarningContract(
            source="IMD_UNAVAILABLE", quality="FORECAST", severity=UNKNOWN, warning_text=UNKNOWN
        )

        # Cyclone data from JTWC
        if needs_warnings and cyclone_data:
            context["cyclone_context"] = cyclone_data # Pass full dict to downstream agents
            provenance.append({
                "source": "JTWC",
                "status": "LIVE" if cyclone_data.get("cyclone_data_status") != "CYCLONE_DATA_UNAVAILABLE" else "CYCLONE_DATA_UNAVAILABLE",
                "retrieved_at": cyclone_data.get("retrieved_at")
            })

            # Still provide the old contract for compatibility if needed, but agents should use cyclone_context
            context["cyclone"] = WarningContract(
                source="JTWC", quality="OBSERVED", severity=UNKNOWN, warning_text=UNKNOWN
            )
        else:
            context["cyclone"] = WarningContract(
                source="NOT_REQUESTED", quality="OBSERVED", severity=UNKNOWN, warning_text=UNKNOWN
            )
            context["cyclone_context"] = None

        context["data_provenance"] = provenance
        return context

    # Try IMD Tier 1 (Weather, Official Warning, Coastal Bulletin)
    try:
        from backend.core.config import settings
        if settings.WEATHER_PROVIDER == "disabled":
            raise Exception("WEATHER_PROVIDER is disabled in configuration.")

        _check_circuit_breaker()
        tier1 = imd.get_tier1(location, location)
        context.update(tier1)
    except Exception as e:
        logger.warning(f"IMD Tier 1 failed: {e}")
        if "Circuit Breaker" not in str(e) and "NO_CREDENTIALS_CONFIGURED" not in str(e):
            _record_failure()

        # Fallback to Open-Meteo for weather/coastal, official warning stays UNKNOWN
        fallback = open_meteo.get_fallback_weather(location)
        context["weather"] = fallback["weather"]
        context["coastal_bulletin"] = fallback["coastal_bulletin"]

        # Ensure we explicitly return UNKNOWN for the official warning since OpenMeteo cannot provide it
        from schemas.contracts import WarningContract
        context["official_marine_warning"] = WarningContract(
            source="IMD_UNAVAILABLE", quality="FORECAST", severity=UNKNOWN, warning_text=UNKNOWN
        )

    # Try IMD Tier 2 (Cyclone) if warnings needed
    if needs_warnings:
        try:
            _check_circuit_breaker()
            tier2 = imd.get_tier2()
            context.update(tier2)
        except Exception as e:
            logger.warning(f"IMD Tier 2 failed: {e}")
            if "Circuit Breaker" not in str(e) and "NO_CREDENTIALS_CONFIGURED" not in str(e):
                _record_failure()
            from schemas.contracts import WarningContract
            context["cyclone"] = WarningContract(
                source="IMD_UNAVAILABLE", quality="OBSERVED", severity=UNKNOWN, warning_text=UNKNOWN
            )
    else:
        from schemas.contracts import WarningContract
        context["cyclone"] = WarningContract(
            source="NOT_REQUESTED", quality="OBSERVED", severity=UNKNOWN, warning_text=UNKNOWN
        )

    return context

def get_ocean_context(lat: float, lon: float, timestamp: datetime) -> dict:
    """
    Facade for fetching ocean state variables.
    Walks DATA_CATALOG (primary -> secondary).
    """
    context = incois.get_ocean_state_forecast(lat, lon, timestamp)

    # Check if we need to fallback/supplement with MOSDAC
    if context.get("surface_current") and context["surface_current"].value is UNKNOWN:
        mosdac_res = mosdac.get_mosdac_ocean_state(lat, lon, timestamp)
        context["surface_current"] = mosdac_res["surface_current"]

    if context.get("sst") and context["sst"].value is UNKNOWN:
        mosdac_res = mosdac.get_mosdac_ocean_state(lat, lon, timestamp)
        context["sst"] = mosdac_res["sst"]

    return context
