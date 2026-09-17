import sys, os
import pandas as pd
import math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
from data_sources.incois import _parse_dms

fixture_path = os.path.join(os.path.dirname(__file__), 'tests', 'fixtures', 'incois_kerala_pfz.html')
with open(fixture_path, 'r') as f:
    html_content = f.read()

dfs = pd.read_html(html_content)
df = dfs[3]
print("Columns:", df.columns.tolist())
for i, row in df.head(3).iterrows():
    lat_dms = row.get('Latitude (dms)')
    lon_dms = row.get('Longitude (dms)')
    print(f"Row {i} Lat: {repr(lat_dms)} Lon: {repr(lon_dms)}")
    if pd.isna(lat_dms):
        print("IS NA!")
    
    lat = _parse_dms(lat_dms)
    print("Parsed lat:", lat)
    
    bearing = row.get('Bearing (deg)')
    print("Bearing:", repr(bearing))
    
    dist_range = str(row.get('Distance (km) From-To', ''))
    print("Dist:", repr(dist_range))

