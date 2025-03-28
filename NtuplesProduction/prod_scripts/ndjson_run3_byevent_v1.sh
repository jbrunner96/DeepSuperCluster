#!/bin/sh

source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc12-opt/setup.sh

BASEDIR="/eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering"

## Electrons
python condor_ndjson_byevent.py -i ${BASEDIR}/FourElectronsGunPt1-500_pythia8_PremixRun3_bsRealistic2022_13p6TeV_126X_mcRun3_2023_forPU65_v4_DumperTraining \
       -nfg 25  \
       -o /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/windows_data/electrons/run3_126X_2023/ndjson_126X_mcRun3_2023_forPU65_byevent_v1\
       -a sim_fraction --wp-file simfraction_electron_run3_PU65.json \
       -q espresso -e user --nocalomatched-nmax 5 \
       --min-et-seed 1  --pu-limit 1e7 -c -cf condor_njdson_electron_run3_2023_byevent

## Photons
python condor_ndjson_byevent.py -i ${BASEDIR}/FourGammasGunPt1-500_pythia8_PremixRun3_bsRealistic2022_13p6TeV_126X_mcRun3_2023_forPU65_v4_DumperTraining \
       -nfg 25  \
       -o /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/windows_data/gammas/run3_126X_2023/ndjson_126X_mcRun3_2023_forPU65_byevent_v1\
       -a sim_fraction --wp-file simfraction_photon_run3_PU65.json \
       -q espresso -e user --nocalomatched-nmax 5 \
       --min-et-seed 1  --pu-limit 1e7 -c -cf condor_njdson_photon_run3_2023_byevent


# overalpping
# ## Electrons
python condor_ndjson_byevent.py -i ${BASEDIR}/FourElectronsGunPt1-500_withOverLap_pythia8_PremixRun3_bsRealistic2022_13p6TeV_126X_mcRun3_2023_forPU65_v4_DumperTraining \
       -nfg 25  \
       -o /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/windows_data/electrons/run3_126X_2023_overlapTraining_double/ndjson_126X_mcRun3_2023_forPU65_byevent_v1\
       -a sim_fraction --wp-file simfraction_electron_run3_PU65.json \
       -q espresso -e user --nocalomatched-nmax 5 \
       --min-et-seed 1  --pu-limit 1e7 -c -cf condor_njdson_electron_run3_2023_overlap_double_byevent

## Photons
python condor_ndjson_byevent.py -i ${BASEDIR}/FourGammasGunPt1-500_withOverLap_pythia8_PremixRun3_bsRealistic2022_13p6TeV_126X_mcRun3_2023_forPU65_v4_DumperTraining \
       -nfg 25  \
       -o /eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/windows_data/gammas/run3_126X_2023_overlapTraining_double/ndjson_126X_mcRun3_2023_forPU65_byevent_v1\
       -a sim_fraction --wp-file simfraction_photon_run3_PU65.json \
       -q espresso -e user  --nocalomatched-nmax 5\
       --min-et-seed 1  --pu-limit 1e7 -c -cf condor_njdson_photon_run3_2023_overlap_double_byevent
