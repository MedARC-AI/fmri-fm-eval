#!/bin/bash

spaces=(
    flat
    schaefer400
    mni_cortex
)

# nb, volume data not currently stored locally
# but remote is fine since the script is not blocked waiting for download
roots=(
    data/sourcedata/HCP_1200
    data/sourcedata/HCP_1200
    s3://hcp-openaccess/HCP_1200
)

outdir="s3://medarc/fmri-fm-eval/processed"

log_path="logs/make_hcpya_all_wds.log"

spaceids="0 1 2"

for ii in $spaceids; do
    space=${spaces[ii]}
    root=${roots[ii]}

    parallel --jobs 16 \
        uv run --no-sync \
        python scripts/make_hcpya_all_wds.py \
        --space "${space}" \
        --root "${root}" \
        --outdir "${outdir}" \
        --shard-id {} ::: {0..1599} \
        2>&1 | tee -a "${log_path}"
done
