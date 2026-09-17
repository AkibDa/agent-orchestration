import requests
import xml.etree.ElementTree as ET

url = "https://incois.gov.in/geoserver/wfs?service=WFS&version=1.1.0&request=GetCapabilities"
headers = {'User-Agent': 'Mozilla/5.0'}
response = requests.get(url, headers=headers, verify=False)
try:
    root = ET.fromstring(response.content)
    for feature in root.findall('.//{http://www.opengis.net/wfs}FeatureType'):
        name = feature.find('{http://www.opengis.net/wfs}Name').text
        print(f"Layer: {name}")
except Exception as e:
    print(f"Error parsing WFS: {e}")

