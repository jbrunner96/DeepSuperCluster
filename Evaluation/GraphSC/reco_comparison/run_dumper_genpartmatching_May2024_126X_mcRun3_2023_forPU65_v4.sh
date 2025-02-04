#!/bin/sh -e

ver=$1
options=" --reco-collection genparticle "

basedir="/eos/cms/store/group/dpg_ecal/alca_ecalcalib/bmarzocc/Clustering/"
outdir="/eos/user/r/rdfexp/ecal/cluster/output_deepcluster_dumper/reco_comparison/genmatching_efficiency_studies/"

#Take an option --dry-run to test the script without submitting the jobs
dryrun=$2


### Electrons, DeepSC_algoA

python condor_reco_dumper.py -i ${basedir}/FourElectronsGunPt1-500_pythia8_PremixRun3_bsRealistic2022_13p6TeV_126X_mcRun3_2023_forPU65_v4_DeepscAlgoA_DumperTraining \
       -o ${outdir}electrons_genmatching/DeepSC_algoA_126X_mcRun3_2023_v${ver}/ -a sim_fraction --wp-file simfraction_electron_235_235.json -nfg 50 -q espresso --compress ${options} -cf condor_dumper_genmatching_ele_deepsc_126X

cd condor_dumper_genmatching_ele_deepsc_126X;

if [ -z $dryrun ]; then
    condor_submit condor_job.txt
fi
cd ..

### Electrons, Mustache
python condor_reco_dumper.py -i ${basedir}/FourElectronsGunPt1-500_pythia8_PremixRun3_bsRealistic2022_13p6TeV_126X_mcRun3_2023_forPU65_v4_Mustache_DumperTraining \
       -o ${outdir}electrons_genmatching/Mustache_126X_mcRun3_2023_v${ver}/ -a sim_fraction --wp-file simfraction_electron_235_235.json -nfg 50 -q espresso --compress ${options} -cf condor_dumper_genmatching_ele_mustache_126X

cd condor_dumper_genmatching_ele_mustache_126X;
if [ -z $dryrun ]; then
    condor_submit condor_job.txt
fi
cd ..


### Photons, DeepSC_algoA
python condor_reco_dumper.py -i ${basedir}/FourGammasGunPt1-500_pythia8_PremixRun3_bsRealistic2022_13p6TeV_126X_mcRun3_2023_forPU65_v4_DeepscAlgoA_DumperTraining \
       -o ${outdir}photons_genmatching/DeepSC_algoA_126X_mcRun3_2023_v${ver}/ -a sim_fraction --wp-file simfraction_photon_235_235.json -nfg 50 -q espresso --compress ${options} -cf condor_dumper_genmatching_photon_deepsc_126X

cd condor_dumper_genmatching_photon_deepsc_126X;
if [ -z $dryrun ]; then
    condor_submit condor_job.txt
fi
cd ..

### Photons, Mustache
python condor_reco_dumper.py -i ${basedir}/FourGammasGunPt1-500_pythia8_PremixRun3_bsRealistic2022_13p6TeV_126X_mcRun3_2023_forPU65_v4_Mustache_DumperTraining \
       -o ${outdir}photons_genmatching/Mustache_126X_mcRun3_2023_v${ver}/ -a sim_fraction --wp-file simfraction_photon_235_235.json -nfg 50 -q espresso --compress ${options} -cf condor_dumper_genmatching_photon_mustache_126X

cd condor_dumper_genmatching_photon_mustache_126X;
if [ -z $dryrun ]; then
    condor_submit condor_job.txt
fi
cd ..

### Electrons, Overlapping objects, DeepSC_algoA
python condor_reco_dumper.py -i ${basedir}/FourElectronsGunPt1-500_withOverLap_pythia8_PremixRun3_bsRealistic2022_13p6TeV_126X_mcRun3_2023_forPU65_v4_DeepscAlgoA_DumperTraining \
       -o ${outdir}electrons_genmatching/DeepSC_algoA_126X_mcRun3_2023_overlapping_v${ver}/ -a sim_fraction --wp-file simfraction_electron_235_235.json -nfg 50 -q espresso --compress ${options} -cf condor_dumper_genmatching_ele_deepsc_126X_overlapping

cd condor_dumper_genmatching_ele_deepsc_126X_overlapping;
if [ -z $dryrun ]; then
    condor_submit condor_job.txt
fi
    
cd ..

### Electrons, Overlapping objects, Mustache
python condor_reco_dumper.py -i ${basedir}/FourElectronsGunPt1-500_withOverLap_pythia8_PremixRun3_bsRealistic2022_13p6TeV_126X_mcRun3_2023_forPU65_v4_Mustache_DumperTraining \
       -o ${outdir}electrons_genmatching/Mustache_126X_mcRun3_2023_overlapping_v${ver}/ -a sim_fraction --wp-file simfraction_electron_235_235.json -nfg 50 -q espresso --compress ${options} -cf condor_dumper_genmatching_ele_mustache_126X_overlapping

cd condor_dumper_genmatching_ele_mustache_126X_overlapping;
if [ -z $dryrun ]; then
    condor_submit condor_job.txt
fi
cd ..

### Photons, Overlapping objects, DeepSC_algoA
python condor_reco_dumper.py -i ${basedir}/FourGammasGunPt1-500_withOverLap_pythia8_PremixRun3_bsRealistic2022_13p6TeV_126X_mcRun3_2023_forPU65_v4_DeepscAlgoA_DumperTraining \
       -o ${outdir}photons_genmatching/DeepSC_algoA_126X_mcRun3_2023_overlapping_v${ver}/ -a sim_fraction --wp-file simfraction_photon_235_235.json -nfg 50 -q espresso --compress ${options} -cf condor_dumper_genmatching_photon_deepsc_126X_overlapping

cd condor_dumper_genmatching_photon_deepsc_126X_overlapping;
if [ -z $dryrun ]; then
    condor_submit condor_job.txt
fi
cd ..

### Photons, Overlapping objects, Mustache
python condor_reco_dumper.py -i ${basedir}/FourGammasGunPt1-500_withOverLap_pythia8_PremixRun3_bsRealistic2022_13p6TeV_126X_mcRun3_2023_forPU65_v4_Mustache_DumperTraining \
       -o ${outdir}photons_genmatching/Mustache_126X_mcRun3_2023_overlapping_v${ver}/ -a sim_fraction --wp-file simfraction_photon_235_235.json -nfg 50 -q espresso --compress ${options} -cf condor_dumper_genmatching_photon_mustache_126X_overlapping

cd condor_dumper_genmatching_photon_mustache_126X_overlapping;
if [ -z $dryrun ]; then
    condor_submit condor_job.txt
fi
cd ..
