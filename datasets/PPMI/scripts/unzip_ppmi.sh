#!/bin/bash
set -euo pipefail

# BASE_DIR can be any directory where the DICOM zip files are present
BASE_DIR="${1:-$(dirname "$0")/..}"
DICOM_DIR="$BASE_DIR/dicom"
UNZIPPED_DIR="$BASE_DIR/unzipped"

mkdir -p "$UNZIPPED_DIR"
shopt -s nullglob

# Automatically detect the number of PPMI parts to process
NUM_PARTS=$(ls "$DICOM_DIR"/PPMI_*.zip.* 2>/dev/null | \
            sed -E 's/.*PPMI_([0-9]{2})\.zip\..*/\1/' | \
            sort -nu | tail -n1)

# Exit if no zip fragments found
if [ -z "$NUM_PARTS" ]; then
    echo "No PPMI zip fragments found in $DICOM_DIR"
    exit 1
fi

# Loop over all zip parts automatically
for i in $(seq -w 1 $NUM_PARTS); do
    FULL_ZIP="$DICOM_DIR/PPMI_${i}.zip"

    echo "Reconstructing PPMI_${i}.zip..."
    cat "$DICOM_DIR"/PPMI_${i}.zip.* > "$FULL_ZIP"

    echo "Unzipping PPMI_${i} into $UNZIPPED_DIR..."
    unzip -q "$FULL_ZIP" -d "$UNZIPPED_DIR"

    echo "Finished PPMI_${i}."
done

echo "All PPMI zips reconstructed and extracted successfully."
