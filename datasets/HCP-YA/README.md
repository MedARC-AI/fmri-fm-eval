# HCP-YA

Homepage: https://www.humanconnectome.org/study/hcp-young-adult/overview

## 1. Download source data

Download minimally preprocessed outputs in MNI152 NLin6Asym (FSL) space and fsLR 91k CIFTI space. Note downloading HCP data form S3 requires signed access.

```bash
aws s3 sync s3://hcp-openaccess/HCP_1200 data/sourcedata/HCP_1200 \
  --exclude "*" \
  --include "*/MNINonLinear/Results/?fMRI_*/?fMRI_*_[LRAP][LRAP].nii.gz" \
  --include "*/MNINonLinear/Results/?fMRI_*/?fMRI_*_Atlas_MSMAll.dtseries.nii" \
  --include "*/MNINonLinear/Results/tfMRI_*/EVs/*"
```

> *Note*: alternatively, the data can be streamed directly from s3 instead of downloading locally.

Download the unrestricted and restricted phenotypic data from [BALSA](https://balsa.wustl.edu/) and copy to [`metadata/hcpya_unrestricted.csv`](metadata/hcpya_unrestricted.csv) and [`metadata/hcpya_restricted.csv`](metadata/hcpya_restricted.csv) respectively.

We only use the restricted sheet for generating subject splits. Specifically, we use the family ID for generating splits of unrelated subjects. Phenotypic prediction targets are constructed from unrestricted data.

## 2. Generate static metadata

### Subject splits

Define 20 random subject splits ("batches") of independent unrelated subjects.

```bash
uv run python scripts/make_hcpya_subject_batch_splits.py
```

The splits are saved in [`metadata/hcpya_subject_batch_splits.json`](metadata/hcpya_subject_batch_splits.json).

The standard subject splits are:

- train: batches `[0, 1, ..., 16]` (or a subset thereof for smaller training data)
- validation: batches `[17, 18]`
- test: batches `[19, 20]`

> *Note:* for pretraining, usually the first 18 batches (train + validation) are used.

### Image metadata

Generate a table including all HCP-YA image metadata. This will make generating derived datasets easier.

```bash
uv run python scripts/make_hcpya_metadata.py
```

The output metadata is saved in [`metadata/hcpya_metadata.parquet`](metadata/hcpya_metadata.parquet).

### Phenotypic prediction targets

Generate discrete coded phenotypic target variables.

```bash
uv run python scripts/make_hcpya_targets.py
```

The targets are saved in [`metadata/hcp_pheno_targets.csv`](metadata/hcp_pheno_targets.csv).

## 3. Generate full webdataset datasets for pretraining

We render *all* HCP-YA data into [webdataset](https://github.com/webdataset/webdataset) shards to use for pretraining.

```bash
bash scripts/make_hcpya_all_wds.sh
```

The script uploads shards automatically to the MedARC S3 bucket (provided your env variables are set up correctly).

```bash
# HCP authorized AWS key
AWS_ACCESS_KEY_ID=XXXX
AWS_SECRET_ACCESS_KEY=XXXX
AWS_ENDPOINT_URL_S3=

# MedARC R2 key
R2_ACCESS_KEY_ID=XXXX
R2_SECRET_ACCESS_KEY=XXXX
R2_ENDPOINT_URL_S3="https://XXXX.r2.cloudflarestorage.com"
```

## 4. Generate eval datasets

All eval datasets are saved in [`data/processed`](data/processed/) in multiple target output spaces (e.g. `schaefer400`, `flat`, `mni`) in huggingface arrow format. All datasets use the same batches of subjects for splits:

- `train`: `{0..7}` (~400 subjects)
- `validation`: `{16..17}` (~100 subjects)
- `test`: `{18..19}` (~100 subjects)

Though the exact number of subjects per split varies across datasets due to differences in data curation.

### `clips` eval dataset

To evaluate model reconstruction performance, we generate an eval dataset of short fMRI clips sampled uniformly from all fMRI runs.

```bash
uv run python scripts/make_hcpya_clips_arrow.py --space schaefer400
```

### `task21` eval dataset

To evaluate cognitive state prediction, we generate an eval dataset of trial-locked fMRI clips sampling from 6 fMRI tasks and 21 task conditions.

```bash
uv run python scripts/make_hcpya_task21_arrow.py --space schaefer400
```

### `rest1lr` eval dataset

To evaluate phenotypic prediction performance, we generate an eval dataset consisting of single resting state runs (`REST1_LR`) truncated to 500 TRs per run.

```bash
uv run python scripts/make_hcpya_rest1lr_arrow.py --space schaefer400
```

### Upload processed datasets to r2

Sync any locally saved datasets to our remote MedARC R2 bucket.

```bash
# args="--dryrun"
args=

for ds_dir in data/processed/*; do
    ds_name=${ds_dir##*/}
    aws s3 sync $args $ds_dir s3://medarc/fmri-datasets/eval/${ds_name}
done
```
