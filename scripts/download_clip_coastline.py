import urllib.request
import json
import os
from shapely.geometry import box, shape, mapping, MultiLineString

url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_coastline.geojson"
print("Downloading Natural Earth 1:10m coastline...")
response = urllib.request.urlopen(url)
data = json.loads(response.read().decode('utf-8'))

print("Clipping to ORCA domain (5N-25N, 65E-95E)...")
orca_bbox = box(65.0, 5.0, 95.0, 25.0)

clipped_features = []
for feature in data.get("features", []):
    geom = shape(feature["geometry"])
    if geom.intersects(orca_bbox):
        clipped_geom = geom.intersection(orca_bbox)
        if not clipped_geom.is_empty:
            clipped_features.append({
                "type": "Feature",
                "properties": feature.get("properties", {}),
                "geometry": mapping(clipped_geom)
            })

out_data = {
    "type": "FeatureCollection",
    "features": clipped_features
}

out_path = os.path.join(os.path.dirname(__file__), "..", "data", "coastline_clipped.geojson")
with open(out_path, "w") as f:
    json.dump(out_data, f)
print(f"Saved to {out_path}. File size: {os.path.getsize(out_path)} bytes.")
