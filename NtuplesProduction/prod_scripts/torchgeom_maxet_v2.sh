#!/bin/sh

source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc12-opt/setup.sh

BASEDIR="/eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/windows_data/"


python condor_torchgeom_dataset.py -i ${BASEDIR}/electrons/run3_126X_2023/ndjson_126X_mcRun3_2023_forPU65_byevent_maxet_v2/raw/ \
       -nef 51200 -o ${BASEDIR}/electrons/run3_126X_2023/ndjson_126X_mcRun3_2023_forPU65_byevent_max_et_v2/processed \
       -q microcentury -cf condor_torchgeom_v1_singleele_maxet_v2 -nfg 3


python condor_torchgeom_dataset.py -i ${BASEDIR}/electrons/run3_126X_2023_overlapTraining_double/ndjson_126X_mcRun3_2023_forPU65_byevent_max_et_v2/raw/ \
       -nef 51200 -o ${BASEDIR}/electrons/run3_126X_2023_overlapTraining_double/ndjson_126X_mcRun3_2023_forPU65_byevent_max_et_v2/processed \
       -q microcentury -cf condor_torchgeom_v1_doubleele_maxet_v2 -nfg 3



python condor_torchgeom_dataset.py -i ${BASEDIR}/gammas/run3_126X_2023/ndjson_126X_mcRun3_2023_forPU65_byevent_max_et_v2/raw/ \
       -nef 51200 -o ${BASEDIR}/gammas/run3_126X_2023/ndjson_126X_mcRun3_2023_forPU65_byevent_max_et_v2/processed \
       -q microcentury -cf condor_torchgeom_v1_singlepho_maxet_v2 -nfg 3



python condor_torchgeom_dataset.py -i ${BASEDIR}/gammas/run3_126X_2023_overlapTraining_double/ndjson_126X_mcRun3_2023_forPU65_byevent_max_et_v2/raw/ \
       -nef 51200 -o ${BASEDIR}/gammas/run3_126X_2023_overlapTraining_double/ndjson_126X_mcRun3_2023_forPU65_byevent_max_et_v2/processed \
       -q microcentury -cf condor_torchgeom_v1_doublepho_maxet_v2 -nfg 3
