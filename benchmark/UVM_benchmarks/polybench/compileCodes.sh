#!/bin/bash

# set PATH and LD_LIBRARY_PATH for CUDA/OpenCL installation (may need to be adjusted)

for currDir in *
do
    echo $currDir
    if [ -d $currDir ]
	then
		cd $currDir
		pwd
		make clean
		make
		cd ..
    fi
done