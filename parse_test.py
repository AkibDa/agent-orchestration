from conversation.router import parse_relative_spatial_constraint
from schemas.location import GeoLocation
loc = GeoLocation(latitude=9.9, longitude=76.2, name="Kochi")
res = parse_relative_spatial_constraint("to the fishing area", loc, None)
print(res)
