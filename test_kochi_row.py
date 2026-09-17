import requests
import pandas as pd
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

sector_id = "SEC005"
session = requests.Session()
headers = {'User-Agent': 'Mozilla/5.0'}
session.get('https://incois.gov.in/MarineFisheries/TextDataHome?mfid=1&request_locale=en', headers=headers, verify=False, timeout=10)
url = f'https://incois.gov.in/MarineFisheries/TextData?secid={sector_id}'
res = session.get(url, headers=headers, verify=False, timeout=10)
res.raise_for_status()

dfs = pd.read_html(StringIO(res.text))
df = dfs[3]
print("Columns:", list(df.columns))

# Find Kunzhathur
for _, row in df.iterrows():
    landmark = row.get('From the coast of', 'Unknown')
    if 'Kunzhathur' in str(landmark):
        print("\n=== Kunzhathur Row ===")
        print(row.to_dict())
        break
