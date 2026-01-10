#!/bin/bash

start=$1
stop=$2

cd ${HOME}/fmri-fm-eval/datasets/PPMI

seq $start $(( stop - 1 )) | parallel --delay 30 ./scripts/preprocess.sh {}
