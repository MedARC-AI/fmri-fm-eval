#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Parent dir (PPMI folder) is one level up from scripts folder
PARENT_DIR="$SCRIPT_DIR/.."

# Path to DICOM subjects
SRC_DIR="$PARENT_DIR/unzipped/PPMI_01/PPMI"

# Path to BIDS output directory
BIDS_DIR="$PARENT_DIR/bids"

# Path to config file
CONFIG="$BIDS_DIR/code/dcm2bids_config.json"

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
