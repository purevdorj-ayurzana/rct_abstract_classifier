import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from verify_data import verify


class StudyDataTests(unittest.TestCase):
    def test_released_data_and_all_recorded_requests(self):
        summary = verify()
        self.assertEqual(summary, {"input_publications": 7552, "recorded_requests_verified": 9730, "validation_publications": 50, "registry_associations": 1890})
