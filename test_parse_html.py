import sys
import os
import pandas as pd
from bs4 import BeautifulSoup
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

fixture_path = os.path.join(os.path.dirname(__file__), 'tests', 'fixtures', 'incois_kerala_pfz.html')
with open(fixture_path, 'r') as f:
    html_content = f.read()

try:
    dfs = pd.read_html(html_content)
    print(f"Number of dfs: {len(dfs)}")
    if len(dfs) > 0:
        print(dfs[3].head())
except Exception as e:
    print(f"Error: {e}")
