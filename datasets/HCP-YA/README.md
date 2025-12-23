# HCP-YA

Homepage: https://www.humanconnectome.org/study/hcp-young-adult/overview

## Dataset creation

### Download preprocessed fMRI data

Download minimally preprocessed outputs in MNI152 NLin6Asym (FSL) space and fsLR 91k CIFTI space. Note downloading HCP data form S3 requires signed access.

```bash
aws s3 sync s3://hcp-openaccess/HCP_1200 data/sourcedata/HCP_1200 \
  --exclude "*" \
  --include "*/MNINonLinear/Results/?fMRI_*/?fMRI_*_[LRAP][LRAP].nii.gz" \
  --include "*/MNINonLinear/Results/?fMRI_*/?fMRI_*_Atlas_MSMAll.dtseries.nii" \
  --include "*/MNINonLinear/Results/tfMRI_*/EVs/*"
```

> *Note*: alternatively, the data can be streamed directly from s3 instead of downloading locally.

### Download phenotypic data

Download the unrestricted and restricted phenotypic data from [BALSA](https://balsa.wustl.edu/) and copy to [`metadata/hcpya_unrestricted.csv`](metadata/hcpya_unrestricted.csv) and [`metadata/hcpya_restricted.csv`](metadata/hcpya_restricted.csv) respectively.

We only use the restricted sheet for generating subject splits. Specifically, we use the family ID for generating splits of unrelated subjects. Phenotypic prediction targets are constructed from unrestricted data.

### Create subject splits

Define 20 random subject splits ("batches") of independent unrelated subjects.

```bash
uv run python scripts/make_hcpya_subject_batch_splits.py
```

The splits are saved in [`metadata/hcpya_subject_batch_splits.json`](metadata/hcpya_subject_batch_splits.json).

The standard subject splits are:

- train: batches `[0, 1, ..., 16]`
- validation: batches `[17, 18]`
- test: batches `[19, 20]`

> *Note:* in a previous version of the dataset, the first 18 batches (train + validation) were used for pretraining.

### Generate metadata table

Generate a table including all HCP-YA image metadata. This will make generating derived datasets easier.

```bash
uv run python scripts/make_hcpya_metadata.py
```

The output metadata is saved in [`metadata/hcpya_metadata.parquet`](metadata/hcpya_metadata.parquet).

### Generate phenotypic prediction targets

Generate discrete coded phenotypic target variables.

```bash
uv run python scripts/make_hcpya_targets.py
```

The targets are saved in [`metadata/targets`](metadata/targets/) as JSON files mapping subject ID to target variable.

### Generate pretraining webdataset

We reserve ~80% of the full HCP-YA dataset for pretraining fMRI foundation models. Generate the fixed pretraining data in [webdataset](https://github.com/webdataset/webdataset) format for all target spaces.

```bash
bash scripts/make_hcpya_all_wds.sh
```

The script uploads shards automatically to the MedARC S3 bucket (provided your env variables are set up correctly).

### Generate `miniclips` eval dataset

To evaluate model reconstruction performance, we generate a small eval dataset of ~2K short fMRI clips sampled uniformly from 100 subjects in each of the train, validation, and test HCP-YA splits.

```bash
bash scripts/make_hcpya_miniclips_arrow.sh
```

The dataset is saved in [`data/processed`](data/processed/) in multiple target output spaces (e.g. parcellated, flat map, MNI) in huggingface arrow format.

### Generate `taskclips` eval dataset

> **_TODO_**

### Generate `rest1lr` eval dataset

To evaluate phenotypic prediction perofmrnace, we generate an eval dataset consisting of single resting state runs (`REST1_LR`) truncated to 500 TRs per run from ~600 subjects.

| split | subjects | frames |
| --- | --- | --- |
| train | 440 | 220K |
| validation | 98 | 49K |
| test | 115 | 58K |

```bash
bash scripts/make_hcpya_rest1lr_arrow.sh
```

### Upload processed datasets to r2

Sync any locally saved datasets to our remote MedARC R2 bucket.

```bash
bash scripts/upload_hcpya_r2.sh
```
