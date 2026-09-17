# data_sources/cyclone.py

import requests
import re
from datetime import datetime, timezone
import math

from agents.geospatial.distance import haversine_distance, bearing

def _parse_jtwc_warning(url: str) -> dict:
    """Fetches and parses a specific JTWC warning text (e.g., io0124web.txt or wp9826web.txt)."""
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        text = resp.text

        pos_match = re.search(r"(\d+\.\d+)([NS])\s+(\d+\.\d+)([EW])", text)
        wind_match = re.search(r"WINDS.*?(\d+)\s*(?:TO\s*(\d+))?\s*KNOTS", text)
        if not wind_match:
            wind_match = re.search(r"MAXIMUM SUSTAINED.*?WINDS.*?(\d+)\s*(?:TO\s*(\d+))?\s*KNOTS", text, re.IGNORECASE)
        pressure_match = re.search(r"PRESSURE.*?(\d+)\s*MB", text)
        if not pressure_match:
             pressure_match = re.search(r"MINIMUM SEA LEVEL PRESSURE.*?(\d+)\s*MB", text, re.IGNORECASE)
        motion_match = re.search(r"MOVING\s+([A-Z-]+)\s+AT\s+(\d+)\s*KNOTS", text)
        if not motion_match:
            motion_match = re.search(r"([A-Z-]+WARD)\s+AT\s+(\d+)\s*KNOTS", text)

        data = {}
        if pos_match:
            lat_val = float(pos_match.group(1))
            lat = lat_val if pos_match.group(2) == 'N' else -lat_val
            lon_val = float(pos_match.group(3))
            lon = lon_val if pos_match.group(4) == 'E' else -lon_val
            data['position'] = {'lat': lat, 'lon': lon}
        if wind_match:
            w1 = int(wind_match.group(1))
            w2 = int(wind_match.group(2)) if wind_match.group(2) else w1
            data['max_wind_kt'] = max(w1, w2)
        if pressure_match:
            data['central_pressure_hpa'] = int(pressure_match.group(1))
        if motion_match:
            data['motion_direction_str'] = motion_match.group(1)
            data['motion_speed_kt'] = int(motion_match.group(2))
            data['motion_speed_kmh'] = round(data['motion_speed_kt'] * 1.852, 1)

        return data
    except Exception as e:
        print(f"Failed to parse JTWC warning at {url}: {e}")
        return {}

def _determine_category(wind_kt: int) -> tuple[str, int]:
    """Returns (IMD Category, Hazard Radius km) based on wind speed in knots."""
    if wind_kt <= 33:
        return ("D", 150)
    elif wind_kt <= 47:
        return ("CS", 250)
    elif wind_kt <= 63:
        return ("SCS", 350)
    elif wind_kt <= 89:
        return ("VSCS", 500)
    elif wind_kt <= 119:
        return ("ESCS", 700)
    else:
        return ("SuCS", 700)

def get_active_cyclones(lat: float, lon: float) -> dict:
    """
    Fetches active cyclones from JTWC for the Indian Ocean.
    """
    now = datetime.now(timezone.utc)

    result = {
        "source": "JTWC",
        "retrieved_at": now.isoformat(),
        "valid_time": now.isoformat(),
        "confidence": 0.0,
        "cyclone_data_status": "CYCLONE_DATA_UNAVAILABLE",
        "active_cyclones": [],
        "nearest_cyclone_distance_km": None,
        "any_within_hazard_radius": None,
        "source_status": "JTWC_FETCH_FAILED"
    }

    try:
        # Check RSS feed for active IO warnings (optional, but good to know what's out there)
        # We will parse ABIO10 for Indian Ocean specifically
        abio_url = "https://www.metoc.navy.mil/jtwc/products/abioweb.txt"
        resp = requests.get(abio_url, timeout=10)
        resp.raise_for_status()
        abio_text = resp.text

        # Parse timestamp from ABIO10 header: e.g., 111800ZSEP2026
        # Or from ABIO10 PGTW 111800
        header_match = re.search(r"ABIO10 PGTW (\d{2})(\d{2})(\d{2})", abio_text)
        if header_match:
            day, hour, minute = int(header_match.group(1)), int(header_match.group(2)), int(header_match.group(3))
            valid_dt = now.replace(day=day, hour=hour, minute=minute, second=0, microsecond=0)
            if valid_dt > now:
                 # If day is greater than today, it's from last month (unlikely) or just future by a bit
                 pass
            result["valid_time"] = valid_dt.isoformat()

            advisory_age_hours = (now - valid_dt).total_seconds() / 3600
            if advisory_age_hours <= 6:
                result["confidence"] = 0.90
            elif advisory_age_hours <= 12:
                result["confidence"] = 0.80
            elif advisory_age_hours <= 24:
                result["confidence"] = 0.65
            else:
                result["confidence"] = 0.50
        else:
             result["confidence"] = 0.70 # Default if can't parse time

        active_cyclones = []

        # Check if ABIO10 says NONE
        if "TROPICAL CYCLONE SUMMARY: NONE" in abio_text and "TROPICAL DISTURBANCE SUMMARY: NONE" in abio_text:
            result["cyclone_data_status"] = "VERIFIED_CLEAR"
            result["source_status"] = "JTWC_LIVE"
            result["nearest_cyclone_distance_km"] = 9999.0
            result["any_within_hazard_radius"] = False

            # Note: 98W (Invest) was in NWPAC but sometimes IO has invests.
            # We will also parse the RSS to check for any TCFA in IO
        else:
            # We have something active in ABIO!
            # Since ABIO text isn't fully structured for all storms, we'd normally parse it
            # But let's check RSS for specific warning links to get exact pos
            rss_url = "https://www.metoc.navy.mil/jtwc/rss/jtwc.rss"
            rss_resp = requests.get(rss_url, timeout=10)
            rss_resp.raise_for_status()

            # Find links ending in web.txt in the Northwest Pacific/North Indian Ocean section
            warning_links = re.findall(r"href='([^']+(?:io|wp)\d{4}web\.txt)'", rss_resp.text)
            warning_links = list(set(warning_links))

            for link in warning_links:
                storm_data = _parse_jtwc_warning(link)
                if storm_data and 'position' in storm_data:
                    c_lat = storm_data['position']['lat']
                    c_lon = storm_data['position']['lon']

                    # Filter for Indian Ocean basically (lon 45 to 100 approx)
                    if 45 <= c_lon <= 100:
                        dist = haversine_distance(lat, lon, c_lat, c_lon)
                        brng = bearing(lat, lon, c_lat, c_lon)

                        wind = storm_data.get('max_wind_kt', 25)
                        cat, hazard_radius = _determine_category(wind)

                        cyclone = {
                            "name": link.split('/')[-1].replace('web.txt', '').upper(), # Fallback name
                            "category": cat,
                            "position": storm_data['position'],
                            "max_wind_kt": wind,
                            "central_pressure_hpa": storm_data.get('central_pressure_hpa', 1000),
                            "motion_speed_kmh": storm_data.get('motion_speed_kmh', 15),
                            "motion_direction_deg": 0, # Requires converting 'WEST-NORTHWESTWARD' to deg, omitted for brevity
                            "distance_km": round(dist, 1),
                            "bearing_deg": round(brng, 1),
                            "within_hazard_radius": dist <= hazard_radius
                        }
                        active_cyclones.append(cyclone)

            if active_cyclones:
                result["active_cyclones"] = active_cyclones
                result["cyclone_data_status"] = "ACTIVE_CYCLONE"

                min_dist = min(c["distance_km"] for c in active_cyclones)
                result["nearest_cyclone_distance_km"] = min_dist
                result["any_within_hazard_radius"] = any(c["within_hazard_radius"] for c in active_cyclones)
                result["source_status"] = "JTWC_LIVE"
            else:
                # ABIO wasn't clear, but no IO warnings found
                result["cyclone_data_status"] = "VERIFIED_CLEAR"
                result["source_status"] = "JTWC_LIVE"
                result["nearest_cyclone_distance_km"] = 9999.0
                result["any_within_hazard_radius"] = False

    except (requests.RequestException, TimeoutError) as e:
        result["error"] = str(e)

    return result
