#!/usr/bin/env bash

base_dir="${PWD}/data/fmriprep"
#skip_list=("Caltech" "CMU_a")

for dir in "$base_dir"/*/; do
    [[ -d "$dir" ]] || continue   # skip if it isn't a directory

    dataset=$(basename "$dir")
    # skip=false
    # for s in "${skip_list[@]}"; do
    #     if [[ "$dataset" == "$s" ]]; then
    #         skip=true
    #         echo "Skipping $dataset"
    #         break
    #     fi
    # done
    # $skip && continue

    echo "Processing: $dataset"
    # Do work here, e.g.:
    rclone copy --progress --copy-links --transfers=16 "$dir" "r2://medarc/fmri-fm-eval/ABIDE/fmriprep/${dataset}"
    #rclone copy --progress --copy-links --transfers=16 /teamspace/studios/this_studio/fmri-fm-eval/ABIDE/preprocessed/fmriprep/CMU_a r2://medarc/fmri-fm-eval/ABIDE/fmriprep/CMU_a/
done

# set -euo pipefail
# dataset=ABIDE
# subject_list="${PWD}/sourcedata/ABIDE_subjects.txt"

# while IFS=' ' read -r dataset subject; do
#     [[ -z "$dataset" || -z "$subject" ]] && continue

#     src_dir="${PWD}/preprocessed/fmriprep/${dataset}/${subject}"
#     dst_dir="r2:medarc/fmri-fm-eval/ABIDE/fmriprep/${dataset}/${subject}"

#     if [[ ! -d "$src_dir" ]]; then
#         echo "Skip: missing ${src_dir}"
#         continue
#     fi

#     echo "Uploading ${src_dir} → ${dst_dir}"
#     rclone copy --progress "$src_dir" "$dst_dir"
# done < "$subject_list"

# rclone copy --progress --copy-links --transfers=16 /teamspace/studios/this_studio/fmri-fm-eval/ABIDE/preprocessed/fmriprep/CMU_a r2://medarc/fmri-fm-eval/ABIDE/fmriprep/CMU_a/
