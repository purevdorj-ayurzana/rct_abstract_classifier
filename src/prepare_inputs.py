"""Rebuild the study input from the original PubMed CSV and abstract export."""
from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from pathlib import Path

from common import atomic_write_json, sha256_file


COLUMNS = ["pmid", "title", "abstract", "doi", "journal", "publication_year", "publication_date"]
PRELUDE = re.compile(
    r"^(?:author information:|collaborators?:|group information:|erratum (?:in|for)|"
    r"retraction (?:in|of)|expression of concern in|comment (?:in|on)|update (?:in|of)|"
    r"correction (?:in|to)|corrected and republished in|republished in|"
    r"summary for patients in|review in|editorial comment in|associated data:)", re.I,
)
TAIL = re.compile(
    r"^(?:copyright\b|©|doi:|pmid:|pmcid:|conflict of interest statement:|"
    r"publisher(?:'s)? disclaimer:|publication types?:|mesh terms?:|substances?:|"
    r"grant support:|figures?:|references?:)", re.I,
)


def flatten(paragraph: str) -> str:
    # Preserve Unicode paragraph separators present in the original study text.
    return " ".join(line.strip() for line in re.split(r"\r?\n", paragraph) if line.strip())


def normalized_title(text: str) -> str:
    normalized = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()
    return re.sub(r"[.。]$", "", normalized).lower()


def extract_abstract(record: str, title: str) -> str:
    paragraphs = [p.strip() for p in re.split(r"(?:\r?\n){2}", record.strip()) if p.strip()]
    matches = [i for i, p in enumerate(paragraphs) if normalized_title(flatten(p)) == normalized_title(title)]
    if not matches:
        raise ValueError("The PubMed record has no matching publication title.")
    cursor = matches[0] + 2  # Title, followed by authors.
    while cursor < len(paragraphs) and PRELUDE.match(flatten(paragraphs[cursor])):
        cursor += 1
    abstract = []
    while cursor < len(paragraphs) and not TAIL.match(flatten(paragraphs[cursor])):
        if not PRELUDE.match(flatten(paragraphs[cursor])):
            abstract.append(flatten(paragraphs[cursor]))
        cursor += 1
    return "\n\n".join(abstract).strip()


def rebuild(publications: Path, abstracts: Path) -> tuple[list[dict], list[dict]]:
    with publications.open(encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    required = {"PMID", "Title", "DOI", "Journal/Book", "Publication Year", "Create Date"}
    if not records or not required.issubset(records[0]):
        raise ValueError("The publication list must be a PubMed CSV export.")
    pmids = [r["PMID"] for r in records]
    if any(not p.strip() for p in pmids) or len(pmids) != len(set(pmids)):
        raise ValueError("The publication list contains missing or duplicate PMIDs.")
    indexed = {r["PMID"]: r for r in records}
    found = {}
    text = abstracts.read_text(encoding="utf-8-sig")
    for record in re.split(r"(?:\r?\n){3,}(?=\d+\.\s)", text.strip()):
        match = re.search(r"^PMID:\s*(\d+)\b", record, re.M)
        if not match:
            raise ValueError("An abstract-export record has no PMID.")
        pmid = match.group(1)
        if pmid in found or pmid not in indexed:
            raise ValueError(f"Duplicate or unexpected abstract-export PMID: {pmid}")
        found[pmid] = extract_abstract(record, indexed[pmid]["Title"])
    ready, missing = [], []
    for record in records:
        pmid = record["PMID"]
        row = dict(zip(COLUMNS, [pmid, record["Title"], found.get(pmid, ""), record["DOI"], record["Journal/Book"], record["Publication Year"], record["Create Date"]]))
        (ready if row["abstract"].strip() else missing).append(row)
    return ready, missing


def write_csv(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publications", default="data/raw/pubmed_publications.csv")
    parser.add_argument("--abstracts", default="data/raw/pubmed_abstracts.txt")
    parser.add_argument("--output-dir", default="outputs/prepared_input")
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    publications, abstracts = Path(args.publications), Path(args.abstracts)
    ready, missing = rebuild(publications, abstracts)
    write_csv(output / "publications_ready.csv", ready)
    write_csv(output / "publications_missing_abstract.csv", missing)
    atomic_write_json(output / "manifest.json", {
        "source_publications_sha256": sha256_file(publications),
        "source_abstracts_sha256": sha256_file(abstracts),
        "ready_rows": len(ready), "missing_abstract_rows": len(missing),
        "date_field_source": "PubMed Create Date",
    })
    print(f"Prepared {len(ready)} publications; {len(missing)} have no abstract.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
