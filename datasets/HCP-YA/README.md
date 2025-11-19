# HCP-YA

Homepage: https://www.humanconnectome.org/study/hcp-young-adult/overview

Download minimally preprocessed outputs

```bash
aws s3 sync --dryrun s3://hcp-openaccess/HCP_1200 data/sourcedata/HCP_1200 \
  --exclude "*" \
  --include "*/MNINonLinear/Results/?fMRI_*/?fMRI_*_[LRAP][LRAP].nii.gz" \
  --include "*/MNINonLinear/Results/?fMRI_*/?fMRI_*_Atlas_MSMAll.dtseries.nii" \
  --include "*/MNINonLinear/Results/tfMRI_*/EVs/*"
```

Download the [HCP restricted behavioral data](https://www.humanconnectome.org/study/hcp-young-adult/document/restricted-data-usage) and copy to [`data/metadata/hcpya_restricted.csv`](data/metadata/hcpya_restricted.csv).
