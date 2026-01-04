#!/bin/bash

# sudo mkdir /output 2>/dev/null
# sudo chmod a+w /output 2>/dev/null
# mkdir /output/preprocessed 2>/dev/null

set -a
#source .env.medarc.r2
set +a

#cd ${HOME}/ABIDE

## ABIDE has 1112 subjects (Refer ABIDE_subjects.txt)
start=1
stop=96
#stop=1112

seq $start $stop | parallel --linebuffer -j 96 ../scripts/preprocess.sh {}
