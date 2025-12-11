#!/bin/bash

rm sourcedata/subject_list.txt 2>/dev/null

while read subdir; do
  dataset=$(echo $subdir | cut -d / -f 3)
  sub=$(echo $subdir | cut -d / -f 4)
  sub=${sub#sub-}
  echo $dataset $sub >> sourcedata/subject_list.txt
done < <(find sourcedata/RawDataBIDS -type d -name 'sub-*' | sort)
