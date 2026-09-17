import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
from data_sources.incois import _parse_incois_html
import logging
logging.basicConfig(level=logging.WARNING)

fixture_path = os.path.join(os.path.dirname(__file__), 'tests', 'fixtures', 'incois_kerala_pfz.html')
with open(fixture_path, 'r') as f:
    html_content = f.read()

c = _parse_incois_html(html_content, sector_id="SEC005")
print(f"Candidates: {len(c)}")
