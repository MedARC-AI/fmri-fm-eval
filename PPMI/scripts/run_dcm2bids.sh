#!/bin/bash

# Path to DICOM subjects
SRC_DIR=../unzipped/PPMI_01/PPMI

# Path to BIDS output directory
BIDS_DIR=../bids

# Path to config file
CONFIG=../bids/code/dcm2bids_config.json

mkdir -p "$BIDS_DIR"

# Loop through each subject folder
for SUB in $SRC_DIR/*; do
    # Extract subject ID (folder name only)
    SUB_ID=$(basename "$SUB")

    echo " Converting subject: $SUB_ID"

    # Run dcm2bids for that subject
    dcm2bids \
        -d "$SUB" \
        -p "$SUB_ID" \
        -c "$CONFIG" \
        -o "$BIDS_DIR"

    echo "Finished subject: $SUB_ID"
    echo
done
