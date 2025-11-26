#!/bin/bash

# all target spaces required by different models
spaces=(
    schaefer400
    schaefer400_tians3
    flat
    a424
    mni
)

# nb, volume data not currently stored locally
# but remote is fine since the script is not blocked waiting for download
roots=(
    data/sourcedata/HCP_1200
    data/sourcedata/HCP_1200
    data/sourcedata/HCP_1200
    s3://hcp-openaccess/HCP_1200
    s3://hcp-openaccess/HCP_1200
)

log_path="logs/make_hcpya_rest1lr_dataset.log"

for ii in {0..4}; do
    space=${spaces[ii]}
    root=${roots[ii]}
    uv run python scripts/make_hcpya_rest1lr_dataset.py \
        --space "${space}" \
        --root "${root}" \
        2>&1 | tee -a "${log_path}"
done
