# HCP-YA

Homepage: https://www.humanconnectome.org/study/hcp-young-adult/overview

## Dataset creation

### 1. Download preprocessed fMRI data

Download minimally preprocessed outputs in MNI152 NLin6Asym (FSL) space and fsLR 91k CIFTI space. Note downloading HCP data form S3 requires signed access.

```bash
aws s3 sync s3://hcp-openaccess/HCP_1200 data/sourcedata/HCP_1200 \
  --exclude "*" \
  --include "*/MNINonLinear/Results/?fMRI_*/?fMRI_*_[LRAP][LRAP].nii.gz" \
  --include "*/MNINonLinear/Results/?fMRI_*/?fMRI_*_Atlas_MSMAll.dtseries.nii" \
  --include "*/MNINonLinear/Results/tfMRI_*/EVs/*"
```

### 2. Download phenotypic data

Download the unrestricted and restricted phenotypic data from [BALSA](https://balsa.wustl.edu/) and copy to [`metadata/hcpya_unrestricted.csv`](metadata/hcpya_unrestricted.csv) and [`metadata/hcpya_restricted.csv`](metadata/hcpya_restricted.csv) respectively.

We only use the restricted sheet for generating subject splits. Specifically, we use the family ID for generating splits of unrelated subjects. Phenotypic prediction targets are constructed from unrestricted data.

### 3. Create subject splits

Define 20 random subject splits ("batches") of independent unrelated subjects.

```bash
uv run python scripts/make_hcpya_subject_batch_splits.py
```

The splits are saved in [`metadata/hcpya_subject_batch_splits.json`](metadata/hcpya_subject_batch_splits.json).


### 4. Generate phenotypic prediction targets

Generate discrete coded phenotypic target variables.

```bash
uv run python scripts/make_hcpya_targets.py
```

The targets are saved in [`metadata/targets`](metadata/targets/) as JSON files mapping subject ID to target variable.

### 5. Generate `REST1_LR` eval dataset

Generate eval dataset consisting of single resting state runs (`REST1_LR`) truncated to 500 TRs per run from ~600 subjects

| split | subjects | frames |
| --- | --- | --- |
| train | 440 | 220K |
| validation | 98 | 49K |
| test | 115 | 58K |

```bash
bash scripts/make_hcpya_rest1lr_dataset.sh
```

The dataset is saved in [`data/processed`](data/processed/) in multiple target output spaces (e.g. parcellated, flat map, MNI) in huggingface arrow format
