# PPMI 

This folder contains all steps and scripts needed to convert the **PPMI** dataset
from the original DICOM ZIP files into a **BIDS** structure using `dcm2bids`.

These instructions assume the PPMI zip files (`PPMI_01.zip.*`, `PPMI_02.zip.*`, …) are already available in a local directory such as:

```
datasets/PPMI/dicom/
```


## Prerequisites
* Ensure `dcm2niix` and `dcm2bids` are installed.
* Ensure `bash` is available


## Reconstruct and Extract the DICOM data
Run the `unzip_ppmi.sh` script in the `datasets/PPMI/scripts/` folder:

```bash
./datasets/PPMI/scripts/unzip_ppmi.sh [base_dir]
```
`[base_dir]` is optional, default: `datasets/PPMI`.

After extraction, the directory structure will look like:
``` 
/datasets/PPMI/unzipped/PPMI/<subject_id>/...
```
`<subject_id>` folders come from the zip contents.


## DICOM to BIDS Conversion

### Create the BIDS directory structure
```bash
mkdir -p datasets/PPMI/bids
```

### Generate BIDS metadata files
```bash
dcm2bids_scaffold -o datasets/PPMI/bids
```

### Copy the configuration file 
The conversion requires a configuration file, provided at: 

`datasets/PPMI/scripts/dcm2bids_config.json`. 

Copy it into the BIDS `code` folder: 
```bash 
cp datasets/PPMI/scripts/dcm2bids_config.json datasets/PPMI/bids/code/dcm2bids_config.json
```

### Run the conversion script
```bash
./datasets/PPMI/scripts/run_dcm2bids.sh [base_dir] [bids_dir] [config_file]
```

* All arguments are optional. 
* Defaults:
* base_dir: `./datasets/PPMI/unzipped/PPMI`
* bids_dir: `./datasets/PPMI/bids`
* config_file: `./datasets/PPMI/bids/code/dcm2bids_config.json`

### Expected output and logging
After conversion, the BIDS dataset will look like:

```
datasets/PPMI/bids/sub-<ID>/ses-<ID>/
```
with:
* ```anat/sub-<ID>_ses-<ID>_T1w.nii.gz```
* ```func/sub-<ID>_ses-<ID>_task-rest_dir-<phase>_bold.nii.gz```

The script creates a log file at:
```
datasets/PPMI/bids/dcm2bids_conversion.log
```


## Generate Filtered Subject List
Run the subject list notebook to generate a filtered list of valid subjects:

```bash
datasets/PPMI/notebooks/ppmi_bids_subject_list.ipynb
```

* This notebook selects subjects based on the defined criteria (e.g., at least 1 T1w and 1 fMRI BOLD scan with >=200 TRs)
* Output CSV: `datasets/PPMI/metadata/PPMI_T1+fMRI_gt200TR_valid_subjects.csv` 
* This file will be used as input for the next curation step.


## Curate Complete PPMI Dataset
Run the curation script:

```bash
python datasets/PPMI/scripts/curate_ppmi.py
```

* Reads the filtered subject list and all clinical metadata.
* Applies the hybrid filtering criteria:
    * Strict filtering for PD/Prodromal patients (all required clinical columns present)
    * Lenient filtering for Healthy Controls (impute missing clinical scores)
* Output CSV: `datasets/PPMI/metadata/PPMI_Hybrid_Cases.csv`


## Create 500-Subject Subset
Run the 500-subject subset script:

```bash
python datasets/PPMI/scripts/subset_500.py
```

* Stratified selection maintaining diagnosis proportions: Healthy Control, Parkinson's Disease, and Prodromal.
* Splits data into train/test/validation (70/15/15%)
* Outputs:

```bash
datasets/PPMI/metadata/PPMI_500_Curated.csv (session-level)
datasets/PPMI/metadata/PPMI_500_Split_Reference.csv (subject-level)
```
