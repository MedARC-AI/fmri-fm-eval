#!/bin/bash

# args="--dryrun"
args=

for ds_dir in data/processed/*; do
    ds_name=${ds_dir##*/}
    aws s3 sync $args $ds_dir s3://medarc/fmri-fm-eval/processed/${ds_name}
done

aws s3 sync $args metadata/targets s3://medarc/fmri-fm-eval/processed/targets
