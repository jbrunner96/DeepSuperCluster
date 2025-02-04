#!/bin/bash -e

echo "Starting"
source /cvmfs/sft.cern.ch/lcg/views/LCG_105a_cuda/x86_64-el9-gcc11-opt/setup.sh

if [ "$4" !=  "None" ]; then
	source $4
fi

echo "Training"

if [ "$6" == "True" ]; then
	python trainer_awk.py --config $1 --model $2 --name $3 --apikey $5 --comet
else
	python trainer_awk.py --config $1 --model $2 --name $3 --apikey $5
fi
