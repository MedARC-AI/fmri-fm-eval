# ADHD200

Download source data

```bash
aws s3 sync --no-sign-request s3://fcp-indi/data/Projects/ADHD200/RawDataBIDS data/sourcedata/RawDataBIDS
```

```bash
wget https://fcon_1000.projects.nitrc.org/indi/adhd200/general/allSubs_testSet_phenotypic_dx.csv -P data/metadata/
```

Get list of datasets and subjects

```bash
bash scripts/find_subjects.sh
```

Run preprocessing

```bash
parallel -j 64 ./scripts/preprocess.sh {} ::: {1..961}
```

Cleanup any abandoned running docker containers

```bash
docker ps -aq --filter ancestor=nipreps/fmriprep:25.2.3 | xargs -r docker rm -f
```

Inspect fmriprep output figures

```bash
python ../scripts/img_viewer.py data/fmriprep
```

Upload fmriprep outputs to r2

```bash
aws s3 sync data/fmriprep s3://medarc/fmri-fm-eval/ADHD200/fmriprep
```
