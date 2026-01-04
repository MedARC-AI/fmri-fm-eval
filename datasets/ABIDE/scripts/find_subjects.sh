#!/bin/bash

rm metadata/ABIDE_subjects.txt 2>/dev/null

while read subdir; do
  dataset=$(echo $subdir | cut -d / -f 3)
  sub=$(echo $subdir | cut -d / -f 4)
  sub=${sub#sub-}
  echo $dataset $sub >> metadata/ABIDE_subjects.txt
done < <(find data/RawDataBIDS -type d -name 'sub-*' | sort)
