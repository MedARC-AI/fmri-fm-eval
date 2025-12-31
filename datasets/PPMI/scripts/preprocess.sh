#!/bin/bash

if [[ -z $1 ]]; then
    echo "preprocess.sh SUBIDX"
    exit
fi

set -a
source ${HOME}/.env.medarc.r2
set +a

subidx=$1
subid=$(sed -n $((subidx + 1))p "${PWD}/metadata/PPMI_BIDS_complete_subs.txt")

# download data for target subject
datadir=/tmp/PPMI/bids_complete
for retry in {1..3}; do
    aws s3 sync \
        s3://medarc/fmri-fm-eval/PPMI/bids_complete \
        ${datadir} \
        --exclude '*' \
        --include 'dataset_description.json' \
        --include '*sub-'${subid}'*'
done

outdir="${PWD}/fmriprep"
logdir="${PWD}/logs"

logname="fmriprep_${subid}.txt"
logpath="${logdir}/$logname"

mkdir -p $outdir 2>/dev/null
mkdir -p $logdir 2>/dev/null

fs_license=$(readlink -f ../../resources/license.txt)
# we need to separately mount a shared fsaverage directory, otherwise there is a race
# https://github.com/nipreps/fmriprep/issues/3492
fsavgdir=$(readlink -f ../../resources/fsaverage)

# using first-lex for sub anatomical reference so that we only run freesurfer once per
# subject. for adni, we used 'sessionwise', but I'm not sure I want to spend the extra
# compute time on this. first-lex is the default, so it should be good enough. typically
# the sessions are not very far apart.
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
    --output-spaces T1w MNI152NLin6Asym:res-2 \
    --cifti-output 91k \
    --nprocs 1 \
    --omp-nthreads 1 \
    --subject-anatomical-reference first-lex \
    --stop-on-first-crash \
    2>&1 | tee -a $logpath

for retry in {1..3}; do
    aws s3 sync \
        ${outdir} \
        s3://medarc/fmri-fm-eval/PPMI/fmriprep \
        --exclude '*' \
        --include '*sub-'${subid}'*'
done

aws s3 cp $logpath "s3://medarc/fmri-fm-eval/PPMI/fmriprep/logs/${logname}"
