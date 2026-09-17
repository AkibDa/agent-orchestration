# data_sources/incois.py

from datetime import datetime, timezone
from typing import Optional, List, Dict
import requests
import logging

from schemas.contracts import Observation, PFZCandidate, Advisory, UNKNOWN

logger = logging.getLogger(__name__)

def load_ocean_datasets():
    """No longer needed; real-time fetching is used."""
    pass

import os
import json
import hashlib
from datetime import datetime, timezone, timedelta
import pandas as pd
from bs4 import BeautifulSoup

def _parse_dms(dms_str: str) -> float:
    """Parse DMS string like '12 44 31 N' to decimal degrees."""
    parts = str(dms_str).strip().split()
    if len(parts) < 3:
        return 0.0
    try:
        d = float(parts[0])
        m = float(parts[1])
        s = float(parts[2])
        dec = d + (m / 60.0) + (s / 3600.0)
        if len(parts) == 4 and parts[3].upper() in ['S', 'W']:
            dec = -dec
        return dec
    except Exception:
        return 0.0

def _parse_incois_html(html_content: str, sector_id: str) -> List[PFZCandidate]:
    """Parse INCOIS HTML text data page to extract PFZ candidates."""
    candidates = []
    
    # Extract validity date if possible
    soup = BeautifulSoup(html_content, 'html.parser')
    valid_until = datetime.now(timezone.utc) + timedelta(hours=48) # Default
    for div in soup.find_all('div'):
        if div.string and 'valid' in div.string.lower():
            # Simplistic parse, could be improved. We just use default +48h for now.
            pass

    try:
        from io import StringIO
        dfs = pd.read_html(StringIO(html_content))
        if len(dfs) < 4:
            return []

            
        df = dfs[3] # The 4th table contains the PFZ data
        for _, row in df.iterrows():
            try:
                lat_dms = row.get('Latitude (dms)')
                lon_dms = row.get('Longitude (dms)')
                if pd.isna(lat_dms) or pd.isna(lon_dms):
                    continue
                    
                lat = _parse_dms(lat_dms)
                lon = _parse_dms(lon_dms)
                
                landmark = row.get('From the coast of', 'Unknown')
                direction = row.get('Direction', 'UNKNOWN')
                bearing_deg = None
                try:
                    bearing_deg = int(row.get('Bearing (deg)'))
                except:
                    pass
                    
                dist_range = str(row.get('Distance (km) From-To', ''))
                depth_range = str(row.get('Depth (mtr) From-To', ''))
                
                # Averages
                avg_dist = 0.0
                if dist_range and '-' in dist_range:
                    parts = dist_range.split('-')
                    avg_dist = (float(parts[0]) + float(parts[1])) / 2.0
                
                avg_depth = 50.0
                if depth_range and '-' in depth_range:
                    parts = depth_range.split('-')
                    avg_depth = (float(parts[0]) + float(parts[1])) / 2.0
                    
                # Generate stable ID
                pid_str = f"{landmark}_{lat}_{lon}"
                pfz_id = hashlib.md5(pid_str.encode('utf-8')).hexdigest()[:10]
                
                candidates.append(
                    PFZCandidate(
                        pfz_id=pfz_id,
                        source="INCOIS_LIVE",
                        quality="OBSERVED",
                        confidence=None,
                        latitude=round(lat, 4),
                        longitude=round(lon, 4),
                        depth=avg_depth,
                        bearing_from_landmark=direction,
                        distance_from_landmark=round(avg_dist, 1),
                        advisory_date=datetime.now(timezone.utc),
                        validity_window=None,
                        incois_distance_km_range=dist_range,
                        incois_depth_m_range=depth_range,
                        incois_bearing_deg=bearing_deg,
                        incois_direction=direction,
                        landing_center=landmark,
                        sector=sector_id,
                        incois_reference_latitude=str(lat_dms),
                        incois_reference_longitude=str(lon_dms)
                    )
                )
            except Exception as e:
                logger.warning(f"Error parsing PFZ row: {e}")
                continue
                
    except Exception as e:
        logger.warning(f"Error extracting tables from INCOIS HTML: {e}")
        
    return candidates

# In-memory cache for demo/production
_pfz_cache = {}

def fetch_live_incois_pfz(sector_id: str) -> List[PFZCandidate]:
    """Fetch live data from INCOIS using the established session logic."""
    import time
    
    # Check cache first (valid for 6 hours)
    if sector_id in _pfz_cache:
        cached_time, cached_data = _pfz_cache[sector_id]
        if time.time() - cached_time < 21600:
            return cached_data
            
    try:
        session = requests.Session()
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        # 1. Establish session
        session.get('https://incois.gov.in/MarineFisheries/TextDataHome?mfid=1&request_locale=en', headers=headers, verify=False, timeout=10)
        
        # 2. Fetch specific sector
        url = f'https://incois.gov.in/MarineFisheries/TextData?secid={sector_id}'
        res = session.get(url, headers=headers, verify=False, timeout=10)
        res.raise_for_status()
        
        candidates = _parse_incois_html(res.text, sector_id)
        if candidates:
            _pfz_cache[sector_id] = (time.time(), candidates)
        return candidates
    except Exception as e:
        logger.warning(f"Live fetch failed for {sector_id}: {e}")
        return []

def get_pfz_advisory(lat: float, lon: float, count: int = 1) -> List[PFZCandidate]:
    """
    Get PFZ advisory using live -> cache -> unavailable fallback.
    If ORCA_PFZ_SOURCE=fixture, read from the local synthetic GeoJSON.
    """
    from shapely.geometry import Point, shape
    from agents.geospatial.distance import haversine_distance, bearing, compass_direction
    
    source_mode = os.environ.get("ORCA_PFZ_SOURCE", "live").lower()
    
    candidates = []
    
    if source_mode == "fixture":
        # Fixture mode: read from static geojson
        geojson_path = os.path.join(os.path.dirname(__file__), "..", "data", "pfz_zones.geojson")
        if os.path.exists(geojson_path):
            with open(geojson_path, "r") as f:
                pfz_data = json.load(f)
            
            origin = Point(lon, lat)
            for feature in pfz_data.get("features", []):
                props = feature.get("properties", {})
                geom = shape(feature["geometry"])
                
                dist_deg = geom.distance(origin)
                if dist_deg < 2.0:
                    from shapely.ops import nearest_points
                    pt_origin, pt_nearest = nearest_points(origin, geom)
                    n_lat, n_lon = pt_nearest.y, pt_nearest.x
                    
                    dist_km = haversine_distance(lat, lon, n_lat, n_lon)
                    b_deg = bearing(lat, lon, n_lat, n_lon)
                    c_lat, c_lon = geom.centroid.y, geom.centroid.x
                    
                    candidates.append(
                        PFZCandidate(
                            pfz_id=props.get("pfz_id", "fixture_01"),
                            source="DEMO/SYNTHETIC",
                            quality="OBSERVED",
                            confidence=props.get("confidence", 0.8),
                            latitude=round(c_lat, 4),
                            longitude=round(c_lon, 4),
                            depth=props.get("depth_m", 50.0),
                            bearing_from_landmark=compass_direction(b_deg),
                            distance_from_landmark=round(dist_km, 1),
                            advisory_date=datetime.now(timezone.utc),
                            validity_window=props.get("validity_window", "48h")
                        )
                    )
    else:
        # Live mode
        # Determine sector based on rough coordinates (or user input logic)
        # For this prototype, we'll map rough bounding boxes to states, or default to a sector.
        # India states bounds roughly:
        sector = "SEC005" # Default to Kerala
        if lon > 85.0 and lat > 20.0:
            sector = "SEC011" # West Bengal
        elif lon > 84.0 and lat > 18.5:
            sector = "SEC010" # Odisha
        
        live_candidates = fetch_live_incois_pfz(sector)
        
        # Spatial filtering:
        for c in live_candidates:
            dist_km = haversine_distance(lat, lon, c.latitude, c.longitude)
            # ORCA calculates distance to the user
            # We preserve INCOIS's reported distance to the landing center in incois_distance_km_range
            if dist_km < 200.0: # Only consider PFZs within 200km
                c_copy = c.model_copy()
                c_copy.distance_from_landmark = round(dist_km, 1) # ORCA's distance
                c_copy.bearing_from_landmark = compass_direction(bearing(lat, lon, c.latitude, c.longitude)) # ORCA's bearing
                candidates.append(c_copy)
                
    if candidates:
        candidates.sort(key=lambda c: c.distance_from_landmark or 0.0)
        return candidates[:count]
        
    return []


def get_ocean_state_forecast(lat: float, lon: float, timestamp: datetime) -> Dict[str, Observation]:
    source = "INCOIS_OSF"

    res = {
        "wave_height": Observation(source=source, quality="FORECAST", value=UNKNOWN, unit="m"),
        "wave_period": Observation(source=source, quality="FORECAST", value=UNKNOWN, unit="s"),
        "wind_speed": Observation(source=source, quality="FORECAST", value=UNKNOWN, unit="m/s"),
        "sst": Observation(source=source, quality="FORECAST", value=UNKNOWN, unit="C"),
        "surface_current": Observation(source=source, quality="FORECAST", value=UNKNOWN, unit="m/s")
    }

    def fetch_marine():
        url = f"https://marine-api.open-meteo.com/v1/marine?latitude={lat}&longitude={lon}&current=wave_height,wave_period,ocean_current_velocity,ocean_current_direction"
        r = requests.get(url, timeout=3.0)
        r.raise_for_status()
        return r.json().get("current", {})

    def fetch_wx():
        url_wx = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m"
        r = requests.get(url_wx, timeout=3.0)
        r.raise_for_status()
        return r.json().get("current", {})

    import concurrent.futures
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(fetch_marine)
            f2 = executor.submit(fetch_wx)

            data = f1.result()
            data_wx = f2.result()

            wh = data.get("wave_height")
            wp = data.get("wave_period")

            if wh is not None:
                res["wave_height"].value = wh
            if wp is not None:
                res["wave_period"].value = wp

            temp = data_wx.get("temperature_2m")
            wind = data_wx.get("wind_speed_10m")

            if temp is not None:
                res["sst"].value = temp
            if wind is not None:
                res["wind_speed"].value = round(wind * (1000/3600), 2)

            current_vel = data.get("ocean_current_velocity")
            current_dir = data.get("ocean_current_direction")
            
            if current_vel is not None:
                res["surface_current"].value = round(current_vel * (1000/3600), 2) # Assume km/h to m/s

    except Exception as e:
        logger.warning(f"Error extracting OSF data: {e}")

    for k in res.values():
        k.valid_time = timestamp

    return res

