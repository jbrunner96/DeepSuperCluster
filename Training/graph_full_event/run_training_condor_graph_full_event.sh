#!/bin/bash -e

echo "Starting"
source /cvmfs/sft.cern.ch/lcg/views/LCG_106a_cuda/x86_64-el9-gcc11-opt/setup.sh 

source $4

echo "Training"

python trainer_graph_full_event.py --config $1 --model $2 --name $3 --apikey $5

