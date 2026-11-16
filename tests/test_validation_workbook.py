import sys
import unittest
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from validation import read_csv


class ValidationWorkbookTests(unittest.TestCase):
    def test_every_saved_excel_annotation_matches_the_csv(self):
        ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        with zipfile.ZipFile(ROOT / "data/validation/annotations.xlsx") as archive:
            shared = ["".join(n.itertext()) for n in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("m:si", ns)] if "xl/sharedStrings.xml" in archive.namelist() else []
            sheets = []
            for number in [1, 2]:
                rows = []
                tree = ET.fromstring(archive.read(f"xl/worksheets/sheet{number}.xml"))
                for row in tree.findall("m:sheetData/m:row", ns):
                    values = {}
                    for cell in row.findall("m:c", ns):
                        node = cell.find("m:v", ns)
                        value = (node.text or "") if node is not None else ""
                        if cell.get("t") == "s" and value:
                            value = shared[int(value)]
                        if cell.get("t") == "inlineStr":
                            value = "".join(cell.find("m:is", ns).itertext())
                        values["".join(c for c in cell.attrib["r"] if c.isalpha())] = value
                    rows.append(values)
                sheets.append(rows[1:])
        annotations = read_csv(ROOT / "data/validation/annotations.csv")
        rct = {"PRIMARY_RCT_REPORT": "Yes", "NOT_PRIMARY_RCT_REPORT": "No", "UNCLEAR": "Unclear", "": ""}
        endpoint = {"POSITIVE": "Positive", "NEGATIVE": "Negative", "UNCLEAR": "Unclear", "NOT_APPLICABLE": "Not applicable", "": ""}
        self.assertEqual(len(sheets[0]), 50)
        for saved, source in zip(sheets[0], annotations):
            expected = [source["pmid"], source["title"], rct[source["pw_rct"]], rct[source["ap_rct"]], rct[source["llm_rct"]], endpoint[source["pw_primary_endpoint"]], endpoint[source["ap_primary_endpoint"]], endpoint[source["llm_primary_endpoint"]]]
            self.assertEqual([saved.get(c, "") for c in "ABCDEFGH"], expected)
        source_rows = read_csv(ROOT / "data/validation/source_annotations.csv")
        self.assertEqual(len(sheets[1]), 150)
        for saved, source in zip(sheets[1], source_rows):
            self.assertEqual([saved.get(c, "") for c in "ABCD"], [source[k] for k in ["pmid", "annotator", "rct_label", "primary_endpoint_success"]])
