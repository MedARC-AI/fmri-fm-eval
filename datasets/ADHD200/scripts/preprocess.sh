#!/bin/bash

if [[ -z $1 ]]; then
    echo "preprocess.sh SUBIDX"
    exit
fi

subidx=$1
subid=$(sed -n ${subidx}p "${PWD}/data/metadata/adhd200_subject_list.txt")
dataset=$(echo $subid | cut -d " " -f 1)
subid=$(echo $subid | cut -d " " -f 2)

datadir="${PWD}/data/sourcedata/RawDataBIDS/${dataset}"
outdir="${PWD}/data/fmriprep/${dataset}"
logdir="${PWD}/logs/fmriprep"

fs_license=$(readlink -f ../resources/license.txt)
# we need to separately mount a shared fsaverage directory, otherwise there is a race
# https://github.com/nipreps/fmriprep/issues/3492
fsavgdir=$(readlink -f ../resources/fsaverage)

mkdir -p $outdir 2>/dev/null
mkdir -p $logdir 2>/dev/null

docker run --rm \
    -v "${datadir}:/data:ro" \
    -v "${outdir}:/out" \
    -v "${fsavgdir}:/out/sourcedata/freesurfer/fsaverage:ro" \
    -v "${fs_license}:/opt/freesurfer/license.txt:ro" \
    nipreps/fmriprep:25.2.3 \
    /data /out participant \
    --participant-label $subid \
    --skip_bids_validation \
    --fs-license-file /opt/freesurfer/license.txt \
    --ignore fieldmaps slicetiming sbref t2w flair fmap-jacobian \
    --output-spaces MNI152NLin6Asym:res-2 \
    --cifti-output 91k \
    --nprocs 1 \
    --omp-nthreads 1 \
    2>&1 | tee -a ${logdir}/${dataset}_${subid}.txt
