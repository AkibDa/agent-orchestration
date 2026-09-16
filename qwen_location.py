from schemas.contracts import GeoLocation


GAZETTEER = {
    "kochi": (9.9312, 76.2673, "Kochi"),
    "kchke": (9.9312, 76.2673, "Kochi"),
    "chennai": (13.0827, 80.2707, "Chennai"),
    "mumbai": (19.0760, 72.8777, "Mumbai"),
    "goa": (15.2993, 74.1240, "Goa"),
    "vizag": (17.6868, 83.2185, "Vizag"),
    "puri": (19.8135, 85.8312, "Puri"),
    "digha": (21.6274, 87.5088, "Digha"),
    "coxs_bazar": (21.4272, 92.0058, "Coxs_Bazar"),
    "coxs": (21.4272, 92.0058, "Coxs_Bazar"),
    "sundarbans": (21.9497, 88.9337, "Sundarbans"),
    "haldia": (22.0257, 88.0583, "Haldia"),
    "bakkhali": (21.2618, 88.2638, "Bakkhali"),
    "mandarmani": (21.6640, 87.6713, "Mandarmani"),
    "sagar island": (21.7960, 88.1384, "Sagar_Island"),
    "sagar": (21.7960, 88.1384, "Sagar_Island"),
    "kakdwip": (21.8764, 88.1878, "Kakdwip"),
    "fraserganj": (21.5833, 88.2500, "Fraserganj"),
    "veraval": (20.9022, 70.3686, "Veraval"),
    "porbandar": (21.6417, 69.6293, "Porbandar"),
    "kakinada": (16.9891, 82.2475, "Kakinada"),
    "mangalore": (12.9141, 74.8560, "Mangalore"),
}


INLAND_LOCATIONS = {
    "madhya pradesh": "Madhya Pradesh",
    "mp": "Madhya Pradesh",
    "ranchi": "Ranchi",
    "delhi": "Delhi",
    "rajasthan": "Rajasthan",
    "punjab": "Punjab",
    "haryana": "Haryana",
    "uttar pradesh": "Uttar Pradesh",
    "up": "Uttar Pradesh",
    "bihar": "Bihar",
    "jharkhand": "Jharkhand",
    "chhattisgarh": "Chhattisgarh",
    "telangana": "Telangana",
    "wb": "West Bengal",
    "west bengal": "West Bengal",
}


class LocationResolution:
    def __init__(
        self,
        geo_location=None,
        location_type="unknown",
        inland_name=None,
        candidate_names=None,
    ):
        self.geo_location = geo_location
        self.location_type = location_type
        self.inland_name = inland_name
        self.candidate_names = candidate_names or []


def extract_location(query_lower: str):
    """Gazetteer lookup used by the Qwen CUDA benchmark."""

    if not query_lower:
        return LocationResolution(None, "unknown", None, [])

    clean_text = query_lower.lower().strip()

    # Strip common Banglish / Hinglish postpositions.
    for suffix in [
        "-r kache",
        "-r",
        "-er",
        " ke paas",
        " ke",
        " kache",
        " kachhe",
    ]:
        if clean_text.endswith(suffix):
            clean_text = clean_text[:-len(suffix)].strip()

    # 1. Coastal matches.
    coastal_matches = []
    seen_names = set()

    for key, (lat, lon, name) in GAZETTEER.items():
        if key in clean_text or clean_text in key:
            if name not in seen_names:
                coastal_matches.append((key, lat, lon, name))
                seen_names.add(name)

    # 2. Inland matches.
    inland_matches = []

    for key, name in INLAND_LOCATIONS.items():
        if key in clean_text or clean_text in key:
            if name not in seen_names:
                inland_matches.append((key, name))
                seen_names.add(name)

    if coastal_matches:
        primary_key, lat, lon, primary_name = coastal_matches[0]

        geo = GeoLocation(
            latitude=lat,
            longitude=lon,
            name=primary_name,
        )

        candidates = (
            [m[3] for m in coastal_matches[1:]]
            + [m[1] for m in inland_matches]
        )

        return LocationResolution(
            geo_location=geo,
            location_type="coastal",
            inland_name=None,
            candidate_names=candidates,
        )

    if inland_matches:
        primary_key, primary_name = inland_matches[0]

        candidates = [m[1] for m in inland_matches[1:]]

        return LocationResolution(
            geo_location=None,
            location_type="inland",
            inland_name=primary_name,
            candidate_names=candidates,
        )

    return LocationResolution(
        geo_location=None,
        location_type="unknown",
        inland_name=None,
        candidate_names=[],
    )