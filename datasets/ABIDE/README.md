# ABIDE

Homepage: https://fcon_1000.projects.nitrc.org/indi/abide/

## 1. Download source data

Download source imaging and phenotypic data from the fcp-indi S3 bucket

```bash
aws s3 sync --no-sign-request \
  s3://fcp-indi/data/Projects/ABIDE/RawDataBIDS data/RawDataBIDS \
  --exclude 'sidecards*'
```

```bash
aws s3 cp --no-sign-request 's3://fcp-indi/data/Projects/ABIDE/Phenotypic_V1_0b_preprocessed.csv' metadata/
```

## 2. Preprocessing

We preprocess the data with [fMRIPrep](https://fmriprep.org) (v25.2.3).

First get a list of all subjects

```bash
bash scripts/find_subjects.sh
```

This should produce a list of 1112 subjects in [`metadata/ABIDE_subjects.txt`](metadata/ABIDE_subjects.txt)

Then preprocess each subject independently by running [`scripts/preprocess.sh`](scripts/preprocess.sh). For example

```bash
parallel -j 64 ./scripts/preprocess.sh {} ::: {1..1112}
```

See also [`scripts/run_preprocess.sh`](scripts/run_preprocess.sh).

## 3. Curation

Construct a curated subset of complete data. The inclusion criteria are:

- runs with complete preprocessing outputs in CIFTI 91k and MNI 152 (NLin6Asym)
- runs with at least 5 min data
- subjects with at least one valid run
- subjects with mean FD < 0.2 mm (using the official FD values released by ABIDE)
- subjects passing QC from rater 1
- sites with at least 20 valid subjects

We split subjects into train, validation, and test sets with a ratio of 70:15:15, stratifying by age (three bins), sex, and diagnosis.

| split      |   Total |   Autism |   Control |   Male |   Female |   Age 6-12 |   Age 13-17 |   Age 18-64 |
|:-----------|--------:|---------:|----------:|-------:|---------:|-----------:|------------:|------------:|
| train      |     578 |      260 |       318 |    485 |       93 |        198 |         198 |         182 |
| validation |     124 |       54 |        70 |    105 |       19 |         43 |          42 |          39 |
| test       |     124 |       57 |        67 |    103 |       21 |         42 |          43 |          39 |

See [`scripts/abide_curation.ipynb`](scripts/abide_curation.ipynb).

```bash
uv run jupyter execute --inplace scripts/abide_curation.ipynb
```

The output table of curated subjects with phenotypic labels is in [`metadata/ABIDE_curated.csv`](metadata/ABIDE_curated.csv).


## 4. Generate Arrow datasets

Finally, generate output Arrow datasets for all target standard spaces

```bash
uv run python scripts/make_abide_dataset.py --space schaefer400
```

The script standardizes all runs to TR = 2.0s across sites, and selects the first 150 TRs (5 min) for each included run.
