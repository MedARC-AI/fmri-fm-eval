# ADHD200

Download source data

```bash
aws s3 sync --no-sign-request s3://fcp-indi/data/Projects/ADHD200/RawDataBIDS sourcedata/RawDataBIDS
```

```bash
wget https://fcon_1000.projects.nitrc.org/indi/adhd200/general/allSubs_testSet_phenotypic_dx.csv -P sourcedata/
```

Get list of datasets and subjects

```bash
bash scripts/find_subjects.sh
```

Run preprocessing

```bash
parallel -j 64 ./scripts/preprocess.sh {} ::: {1..961}
```

Inspect fmriprep output figures

```bash
python ../utils/img_viewer.py preprocessed/fmriprep
```
