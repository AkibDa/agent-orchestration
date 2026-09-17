import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
from data_sources.incois import _parse_dms, _parse_incois_html

print(_parse_dms("12 44 31 N"))
print(_parse_dms("74 31 29 E"))

fixture_path = os.path.join(os.path.dirname(__file__), 'tests', 'fixtures', 'incois_kerala_pfz.html')
with open(fixture_path, 'r') as f:
    html_content = f.read()

candidates = _parse_incois_html(html_content, sector_id="SEC005")
print(f"Candidates found: {len(candidates)}")
