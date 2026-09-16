# Proto/location/models.py

from dataclasses import dataclass, field
from typing import Optional, List
from schemas.contracts import GeoLocation

@dataclass
class ResolvedLocation:
    canonical_name: str
    latitude: Optional[float]
    longitude: Optional[float]
    location_class: str = "coastal_point"  # coastal_point, coastal_city, inland_city, inland_region, unknown
    marine_access: str = "direct"          # direct, nearby, none
    coastal_access: bool = True
    region: Optional[str] = None
    role: str = "TARGET"                   # TARGET or REFERENCE
    confidence: float = 1.0
    source: str = "GAZETTEER"              # GAZETTEER or METADATA
    resolution_status: str = "EXACT"       # EXACT, FUZZY_HIGH_CONFIDENCE, AMBIGUOUS, NOT_FOUND
    matched_text: Optional[str] = None
    candidate_names: List[str] = field(default_factory=list)

    @property
    def geo_location(self) -> Optional[GeoLocation]:
        if self.latitude is not None and self.longitude is not None:
            return GeoLocation(latitude=self.latitude, longitude=self.longitude, name=self.canonical_name)
        return None

    @property
    def location_type(self) -> str:
        return "coastal" if self.coastal_access else "inland"
