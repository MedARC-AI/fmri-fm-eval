#!/bin/bash

# Base directories
BASE_DIR="${1:-./datasets/PPMI/unzipped/PPMI}" # Original unzipped PPMI DICOMs
BIDS_DIR="${2:-./datasets/PPMI/bids}" # Output BIDS directory
CONFIG="${3:-$BIDS_DIR/code/dcm2bids_config.json}" # Path to dcm2bids config file

echo "Using directories:"
echo "  Base: $BASE_DIR"
echo "  BIDS output:  $BIDS_DIR"
echo "  Config file:  $CONFIG"
echo ""

# Log file
LOG_FILE="$BIDS_DIR/dcm2bids_conversion.log"
echo "Logging to $LOG_FILE"
echo "Conversion started at $(date)" > "$LOG_FILE"

# Check base directory
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Base directory not found at $BASE_DIR" | tee -a "$LOG_FILE"
    exit 1
fi

# Check config
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found at $CONFIG" | tee -a "$LOG_FILE"
    exit 1
fi

# Loop over all subjects
for SUB_DIR in "$BASE_DIR"/*; do
    [ -d "$SUB_DIR" ] || continue 

    SUB=$(basename "$SUB_DIR")
    echo "Processing subject $SUB..." | tee -a "$LOG_FILE"

    # Loop over all run folders (e.g., T1 or rsfMRI)
    for RUN_DIR in "$SUB_DIR"/*; do
        [ -d "$RUN_DIR" ] || continue
        RUN_NAME=$(basename "$RUN_DIR")
        
        # Loop over all session folders
        for SESFOLDER in "$RUN_DIR"/*; do
            [ -d "$SESFOLDER" ] || continue
            # Extract session date from folder name: 2022-09-21_13_41_02.0 -> 20220921
            SESNAME=$(basename "$SESFOLDER" | cut -d'_' -f1 | tr -d '-')
            
            # Loop over all DICOM folders
            for DICOM_DIR in "$SESFOLDER"/*; do
                [ -d "$DICOM_DIR" ] || continue
                DICOM_NAME=$(basename "$DICOM_DIR")
                
                echo "Subject: $SUB, Run: $RUN_NAME, Session: $SESNAME, DICOM: $DICOM_NAME" >> "$LOG_FILE"
                
                # Run dcm2bids
                dcm2bids \
                    -d "$DICOM_DIR" \
                    -p "$SUB" \
                    -c "$CONFIG" \
                    -o "$BIDS_DIR" \
                    -s "ses-$SESNAME" \
                    --auto_extract_entities 2>> "$LOG_FILE"

                if [ $? -ne 0 ]; then
                    echo "WARNING: dcm2bids failed for Subject: $SUB, DICOM: $DICOM_NAME" >> "$LOG_FILE"
                fi
            done
        done
    done

    echo "Finished subject $SUB" | tee -a "$LOG_FILE"
done

echo "All subjects processed at $(date)" >> "$LOG_FILE"
