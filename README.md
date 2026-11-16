# rct_abstract_classifier

Data and code associated with the manuscript **“An LLM-assisted Pipeline for
Identifying ClinicalTrials.gov Registry Changes and Their Association with
Primary Endpoint Positivity.”**

This pipeline identifies publications reporting results that directly compare
randomized trial groups. It then assesses whether an explicitly identified
prespecified primary endpoint was successful, using the title and abstract.
ClinicalTrials.gov registry-history analysis is provided in the
[companion repository](https://github.com/p-ayurzana/clinicaltrials_registry_pipeline/tree/main).

## Data

The study includes **7,552 publications** from JAMA, The Lancet, The BMJ, and
The New England Journal of Medicine. The original PubMed downloads and the
processed abstracts are included in [data](data/README.md).

The [results](results) contain all RCT classifications, **2,178 primary-endpoint
assessments**, and the publication–registry links used for the registry-history
project. Records with uncertain classifications are retained.

## Workflow

1. Match the downloaded publication list and abstracts by PMID.
2. Identify reports that compare randomized groups.
3. Assess primary-endpoint success in the eligible reports.
4. Extract abstract NCT identifiers and prepare the registry handoff.

Both classification steps use **gpt-5.6-sol with medium reasoning** and the
included [prompts](prompts). Eligible reports include secondary and follow-up
publications when they preserve the randomized comparison. The label
`PRIMARY_RCT_REPORT` includes all such reports.

## Validation

PW and AP annotated a random sample of **50 publications** for RCT classification
and primary-endpoint success. Their annotations and the LLM results are available
in the [validation table](data/validation/annotations.csv).
Individual disagreements are preserved.

## Use

Python 3.12 was used for the study. From the repository root:

```text
python -m pip install -r requirements.txt
python src/verify_data.py
python -m unittest discover -s tests
```

The saved results can be examined without an API key. To run new classifications,
set `OPENAI_API_KEY` in your environment and follow [Running the pipeline](RUNNING.md).
API calls incur charges; a rerun may produce different labels.

Code is released under the [MIT license](LICENSE). PubMed abstracts retain their
existing rights; see [Data rights](DATA_RIGHTS.md).
