from location.resolver import extract_location
res = extract_location("fishing area")
print(hasattr(res, "geo_location"))
print(getattr(res, "geo_location", None))
