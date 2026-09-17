import unittest
from conversation.response_generator import generate_multilingual_response

class TestMultilingualPFZ(unittest.TestCase):
    def setUp(self):
        self.recommendation_result = {
            "result_type": "PFZ_RESULT",
            "location": {"name": "Digha"},
            "found": True,
            "qualified": True,
            "candidate": {
                "distance_km": 25.6,
                "bearing": "S",
                "latitude": 21.4119,
                "longitude": 87.5983,
                "probability": None,
                "validity_window": None,
                "incois_distance_km_range": "22-27",
                "incois_depth_m_range": "6-11",
                "incois_direction": "SE"
            },
            "source": "INCOIS_LIVE",
            "decision_threshold": 0.85
        }

    def test_english_response(self):
        text, segs = generate_multilingual_response(self.recommendation_result, language="en", context={"pfz": {}})
        self.assertIn("Nearest PFZ near Digha", text)
        self.assertIn("25.6 km", text)
        self.assertIn("INCOIS: 22-27 km", text)
        self.assertNotIn("Probability", text)
        
    def test_bengali_response(self):
        text, segs = generate_multilingual_response(self.recommendation_result, language="bn", context={"pfz": {}})
        self.assertIn("Digha-এর কাছাকাছি nearest PFZ", text)
        self.assertIn("25.6 km", text)
        self.assertIn("INCOIS: 22-27 km", text)
        
    def test_bengalish_response(self):
        text, segs = generate_multilingual_response(self.recommendation_result, language="bn_en", context={"pfz": {}})
        self.assertIn("Digha-r kachakachi nearest PFZ", text)
        self.assertIn("25.6 km", text)
        self.assertIn("INCOIS: 22-27 km", text)

    def test_hinglish_response(self):
        text, segs = generate_multilingual_response(self.recommendation_result, language="hi-Latn", context={"pfz": {}})
        self.assertIn("Digha ke paas nearest PFZ", text)
        self.assertIn("25.6 km", text)
        self.assertIn("INCOIS: 22-27 km", text)

if __name__ == "__main__":
    unittest.main()
