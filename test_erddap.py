import requests
import json

url = "https://incois.gov.in/erddap/search/index.json?page=1&itemsPerPage=1000&searchFor=pfz"
headers = {'User-Agent': 'Mozilla/5.0'}
response = requests.get(url, headers=headers, verify=False)
try:
    data = response.json()
    print("Found ERDDAP datasets:")
    for row in data['table']['rows']:
        print(row[0])
except Exception as e:
    print(f"Error parsing json: {e}")
    print(response.text[:500])

