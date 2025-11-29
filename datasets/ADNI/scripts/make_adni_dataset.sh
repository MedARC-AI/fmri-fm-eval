#!/bin/bash
spaces=(
    schaefer400
    schaefer400_tians3
    flat
    a424
    mni
)

log_path="logs/make_adni_dataset.log"
mkdir -p logs

for space in "${spaces[@]}"; do
    echo "Generating space: ${space}"
    uv run python scripts/make_adni_dataset.py \
        --space "${space}" \
        2>&1 | tee -a "${log_path}"
done

echo "Done generating all datasets"
