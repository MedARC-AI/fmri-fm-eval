#!/bin/bash

sub_list_path="data/metadata/adhd200_subject_list.txt"
rm $sub_list_path 2>/dev/null

while read subdir; do
  dataset=$(echo $subdir | cut -d / -f 3)
  sub=$(echo $subdir | cut -d / -f 4)
  sub=${sub#sub-}
  echo $dataset $sub >> $sub_list_path
done < <(find data/sourcedata/RawDataBIDS -type d -name 'sub-*' | sort)
