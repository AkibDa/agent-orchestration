import urllib.request
import json
import datetime

# Kochi: 9.93, 76.26
url = "https://marine-api.open-meteo.com/v1/marine?latitude=9.93&longitude=76.26&hourly=sea_level_height_msl&timezone=Asia/Kolkata"
req = urllib.request.Request(url)
try:
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        print("KOCHI RESPONSE (first 10 hours):")
        times = data["hourly"]["time"]
        levels = data["hourly"]["sea_level_height_msl"]
        for i in range(10):
            print(f"  {times[i]} -> {levels[i]} m")
except Exception as e:
    print("Error:", e)

# Mandarmani: 21.66, 87.73
url2 = "https://marine-api.open-meteo.com/v1/marine?latitude=21.66&longitude=87.73&hourly=sea_level_height_msl&timezone=Asia/Kolkata"
try:
    with urllib.request.urlopen(url2) as response:
        data = json.loads(response.read().decode())
        print("\nMANDARMANI RESPONSE (first 10 hours):")
        times = data["hourly"]["time"]
        levels = data["hourly"]["sea_level_height_msl"]
        for i in range(10):
            print(f"  {times[i]} -> {levels[i]} m")
except Exception as e:
    print("Error:", e)
