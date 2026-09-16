from config import IMD_ENABLED

DATA_CATALOG = {
    "pfz":                    {"primary": "INCOIS_PFZ",  "fallback": None,        "required": True},
    "wave_height":            {"primary": "INCOIS_OSF",  "fallback": None,        "required": True},
    "wave_period":            {"primary": "INCOIS_OSF",  "fallback": None,        "required": False},
    "surface_current":        {"primary": "INCOIS_OSF",  "secondary": "MOSDAC",   "required": False},
    "sst":                    {"primary": "INCOIS_OSF",  "secondary": "MOSDAC",   "required": True},
    "weather":                {"primary": "IMD" if IMD_ENABLED else "OPEN_METEO", "fallback": "OPEN_METEO" if IMD_ENABLED else None, "required": True},
    "official_marine_warning":{"primary": "IMD" if IMD_ENABLED else "IMD_UNAVAILABLE", "fallback": None, "required": True},
    "cyclone":                {"primary": "IMD" if IMD_ENABLED else "JTWC", "fallback": None, "required": True},
}
