#!/bin/bash

docker create --name temp_fmriprep_container nipreps/fmriprep:25.2.3
docker cp temp_fmriprep_container:/opt/freesurfer/subjects/fsaverage /path/to/your/host/directory
docker rm temp_fmriprep_container