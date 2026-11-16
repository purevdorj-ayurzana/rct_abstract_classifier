# Data guide

All tables join by `pmid`, the PubMed publication identifier.

| File | Contents |
| --- | --- |
| `raw/pubmed_publications.csv` | Original PubMed publication-list download |
| `raw/pubmed_abstracts.txt` | Original PubMed abstract-text download |
| `processed/publications_ready.csv` | The 7,552 records supplied to the classifier |
| `validation/annotations.csv` | Combined PW, AP, and LLM annotations for 50 publications |
| `validation/annotations.xlsx` | The same annotations in a readable workbook |
| `validation/source_annotations.csv` | Original annotation labels, one row per publication and annotator |
| `validation/source_manifest.json` | Annotation-source details and checksums |
| `validation/sampling_manifest.json` | Recorded random-sample method, seed, and selected PMIDs |

The processed input preserves titles, abstracts, DOI, journal, publication year,
and date. `publication_date` contains PubMed's **Create Date**, as supplied in the
study input. Publication years range from 2020 to 2026; 52 records have year 2026.
All 7,552 records have an abstract. The exact original search query and download
date are not recorded in the supplied source files.

## Results

`results/rct_classifications.csv` contains one result per input publication:

- `final_label`: `PRIMARY_RCT_REPORT`, `NOT_PRIMARY_RCT_REPORT`, or `UNCLEAR`.
- `underlying_trial_randomized`: whether randomized allocation was identified.
- `publication_role`: primary results, secondary/follow-up, or another publication type.
- Trial name, reported registration identifiers, and a brief rationale.
- `nct_abstract_regex_found` and `nct_abstract_registry_numbers`: deterministic
  abstract-only NCT detection, populated only for eligible reports.

The positive label includes any eligible randomized-group results comparison,
not just the original primary-results publication. A missing NCT identifier does
not make a publication ineligible.

`results/primary_endpoint_assessments.csv` contains the 2,178 eligible publications.
It records `POSITIVE`, `NEGATIVE`, or `UNCLEAR`, the explicitly identified primary
endpoint, trial framework, rationale, and reason for uncertainty. Secondary
endpoints cannot make the primary endpoint positive.

`results/provenance` records model settings, request and input checksums, token
usage, and response identifiers. `results/study.json` gives study counts and
checks. `results/manifest.json` records released-file checksums.

`results/registry_handoff` preserves all 2,178 eligible publications, including
371 without an accepted NCT association. It contains 1,890 PMID–NCT associations
covering 1,759 distinct candidate NCT identifiers. Two source-cited manual
corrections supplement the abstract regex; the original regex flags remain unchanged.
Candidate identifiers are not proof of trial eligibility or registry availability.

## Validation labels

The combined CSV uses the RCT labels defined above and primary-endpoint labels
`POSITIVE`, `NEGATIVE`, `UNCLEAR`, and `NOT_APPLICABLE`. The workbook displays
RCT eligibility as Yes/No/Unclear and endpoint labels in plain language.

PW's original RCT labels are retained directly. AP and LLM `TRUE`/`FALSE` map to
eligible/excluded RCT reports. For endpoint success, `TRUE`/`FALSE` map to
`POSITIVE`/`NEGATIVE`; `UNCLEAR` is unchanged. Source `NA`, and blank LLM endpoints
for excluded records, map to `NOT_APPLICABLE`. Other missing labels remain blank.
Original values are preserved in the source table. No consensus label is added.
