import pytest
from datetime import datetime
import os
import sys

# add parent dir to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_sources.incois import _parse_dms, _parse_incois_html

def test_parse_dms():
    assert _parse_dms("12 44 31 N") == pytest.approx(12.741944444444445, abs=1e-6)
    assert _parse_dms("74 52 29 E") == pytest.approx(74.87472222222223, abs=1e-6)
    assert _parse_dms("10 0 0 S") == -10.0
    assert _parse_dms("20 30 0 W") == -20.5

def test_parse_html():
    fixture_path = os.path.join(os.path.dirname(__file__), 'fixtures', 'incois_kerala_pfz.html')
    with open(fixture_path, 'r') as f:
        html_content = f.read()
    
    candidates = _parse_incois_html(html_content, sector_id="SEC005")
    
    assert len(candidates) > 0
    c0 = candidates[0]
    assert c0.pfz_id is not None
    assert c0.source == "INCOIS_LIVE"
    assert c0.incois_direction == "W"
    assert c0.incois_bearing_deg == 270
    assert c0.incois_distance_km_range == "35-40"
    assert c0.incois_depth_m_range == "49-54"
    assert abs(c0.latitude - 12.7419) < 0.001
    assert abs(c0.longitude - 74.5247) < 0.001
    assert c0.distance_from_landmark == 37.5
    assert c0.depth == 51.5

if __name__ == "__main__":
    pytest.main(["-v", __file__])
