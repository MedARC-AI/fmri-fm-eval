# ABIDE

Download source imaging and phenotypic data

```bash
aws s3 sync --no-sign-request s3://fcp-indi/data/Projects/ABIDE/RawDataBIDS data/sourcedata/RawDataBIDS
```

```bash
aws s3 sync --no-sign-request s3://fcp-indi/data/Projects/ABIDE/PhenotypicData data/metadata/PhenotypicData
aws s3 cp --no-sign-request 's3://fcp-indi/data/Projects/ABIDE/Phenotypic_V1_0b.csv' data/metadata/
aws s3 cp --no-sign-request 's3://fcp-indi/data/Projects/ABIDE/Phenotypic_V1_0b_preprocessed.csv' data/metadata/
aws s3 cp --no-sign-request 's3://fcp-indi/data/Projects/ABIDE/Phenotypic_V1_0b_preprocessed1.csv' data/metadata/
```
