import requests
import pandas as pd
from bs4 import BeautifulSoup

session = requests.Session()
headers = {'User-Agent': 'Mozilla/5.0'}

# Initialize session
session.get('https://incois.gov.in/MarineFisheries/TextDataHome?mfid=1&request_locale=en', headers=headers, verify=False)

def get_pfz_data(secid, state_name):
    print(f"Fetching {state_name} ({secid})...")
    url = f'https://incois.gov.in/MarineFisheries/TextData?secid={secid}'
    res = session.get(url, headers=headers, verify=False)
    
    # Get date/validity from the page
    soup = BeautifulSoup(res.text, 'html.parser')
    validity_text = "Unknown"
    for div in soup.find_all('div'):
        if div.string and 'valid' in div.string.lower():
            validity_text = div.string.strip()
            break
            
    try:
        dfs = pd.read_html(res.text)
        if len(dfs) >= 4:
            df = dfs[3]
            print(f"Validity: {validity_text}")
            print(f"Records found: {len(df)}")
            print(df.head(1).to_dict('records'))
        else:
            print("Table not found!")
    except Exception as e:
        print("Error parsing:", e)

get_pfz_data('SEC010', 'Odisha')
get_pfz_data('SEC011', 'West Bengal')

