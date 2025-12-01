# PPMI 

This folder contains all steps and scripts needed to convert the **PPMI** dataset
from the original DICOM ZIP files into a **BIDS** structure using `dcm2niix` and `dcm2bids`.

These instructions assume the PPMI zip files `PPMI_01.zip.*`, `PPMI_02.zip.*`, …
are already available in a local directory such as:

```
PPMI/dicom/
```

## Reconstruct and Extract ZIP Files (PPMI_01, PPMI_02, ...)
PPMI splits large ZIP archives into multiple chunks: 

```
PPMI_01.zip.aa
PPMI_01.zip.ab
PPMI_01.zip.ac
...
```

Reconstruct the full ZIP file
```bash
cat PPMI/dicom/PPMI_01.zip.* > PPMI/dicom/PPMI_01.zip
``` 

Extract the DICOM data

```bash
mkdir -p PPMI/unzipped/PPMI_01
unzip PPMI/dicom/PPMI_01.zip -d PPMI/unzipped/PPMI_01
```

After extraction, the directory will look like:
``` 
PPMI/unzipped/PPMI_01/PPMI/<subject_id>/...
```

## DICOM to BIDS Conversion
Ensure `dcm2niix` and `dcm2bids` are installed.

Create BIDS directory structure
```bash
mkdir -p PPMI/bids
```

Create template BIDS metadata files
```bash
dcm2bids_scaffold -o PPMI/bids
```

Copy configuration file 

The conversion requires the configuration file, which is provided at `PPMI/scripts/dcm2bids_config.json`. **Copy this file** into the BIDS metadata location: 
```bash 
cp PPMI/scripts/dcm2bids_config.json PPMI/bids/code/dcm2bids_config.json
```

Run the conversion
``` bash
bash PPMI/scripts/run_dcm2bids.sh
```

Expected output structure

```
PPMI/bids/sub-<ID>/
```
with:
* ```anat/sub-<ID>_T1w.nii.gz```
* ```func/sub-<ID>_task-rest_dir-<phase>_bold.nii.gz```

## Notes
* Some subjects may be skipped if their fMRI runs are extremely short or malformed.
* The resulting BIDS folder is ready for preprocessing with `fMRIPrep`.