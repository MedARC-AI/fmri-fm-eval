#!/bin/bash

subdirs=$(find data/fmriprep -maxdepth 2 -name 'sub-*' -type d | sort)

for dir in $subdirs; do
    if [[ ! -f ${dir}.html ]]; then
        sub=${dir##*/}
        parent=${dir%/*}
        echo $parent $sub
        # sudo rm -rf $dir
        # sudo rm -rf ${parent}/sourcedata/freesurfer/${sub}_*
    fi
done
