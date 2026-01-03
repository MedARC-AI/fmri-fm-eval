#!/bin/bash

rm metadata/subject_list.txt 2>/dev/null

while read subdir; do
  dataset=$(echo $subdir | cut -d / -f 3)
  sub=$(echo $subdir | cut -d / -f 4)
  sub=${sub#sub-}
  echo $dataset $sub >> metadata/subject_list.txt
done < <(find data/RawDataBIDS -type d -name 'sub-*' | sort)
