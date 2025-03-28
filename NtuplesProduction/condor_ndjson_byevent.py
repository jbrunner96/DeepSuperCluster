import sys
import os
import argparse
import random
from math import *
from glob import glob

with open("command.txt", "w") as of:
    of.write(" ".join(["python"]+sys.argv))

'''
This scripts runs hadd on single crystal files to 
group them in strips reading a DOF file
'''
parser = argparse.ArgumentParser()

parser.add_argument("-i", "--inputdir", type=str, help="Inputdir", required=True)
parser.add_argument("-nfg", "--nfile-group", type=int, help="How many files per numpy file", required=True)
# parser.add_argument("-tf", "--test-fraction", type=float, help="Fraction of files for testing", required=True)
parser.add_argument("-o", "--outputdir", type=str, help="Outputdir", required=True)
parser.add_argument("-a","--assoc-strategy", type=str, help="Association strategy", required=True, default="sim_fraction")
parser.add_argument("--wp-file", type=str,  help="File with sim fraction thresholds")
parser.add_argument("-q", "--queue", type=str, help="Condor queue", default="longlunch", required=True)
parser.add_argument("-e", "--eos", type=str, default="user", help="EOS instance user/cms", required=False)
parser.add_argument("--min-et-seed", type=float,  help="Min Et of the seeds", default=1)
parser.add_argument("--max-et-seed", type=float,  help="Max Et of the seeds", default=1e6)
parser.add_argument("--max-et-isolated-cl", type=float,  help="Max Et of the isolated cluster", default=1e6)
parser.add_argument("-ov","--overlap", action="store_true",  help="Overlapping window mode", default=False)
parser.add_argument("--pu-limit", type=float,  help="SimEnergy PU limit", default=1e6)
parser.add_argument('-c', "--compress", action="store_true",  help="Compress output")
parser.add_argument("--redo", action="store_true", default=False, help="Redo all files")
parser.add_argument("--nocalomatched-nmax", type=int,  help="Max number of calomatched clusters", default=10)
parser.add_argument("-d","--debug", action="store_true",  help="debug", default=False)
parser.add_argument("-cf","--condor-folder", type=str,  help="Condor folder", default="condor_ndjson")
args = parser.parse_args()

# Create output folder for jobs configuration
os.makedirs(args.condor_folder, exist_ok=True)
os.makedirs(args.condor_folder+"/error", exist_ok=True)
os.makedirs(args.condor_folder+"/output", exist_ok=True)
os.makedirs(args.condor_folder+"/log", exist_ok=True)

# Prepare condor jobs
condor = '''executable              = run_ndjson_byevent_script.sh
output                  = output/strips.$(ClusterId).$(ProcId).out
error                   = error/strips.$(ClusterId).$(ProcId).err
log                     = log/strips.$(ClusterId).log
transfer_input_files    = ../cluster_ndjson_byevent.py, ../windows_creator_graph_byevent.py, ../calo_association.py, ../simScore_WP/{wp_file}, ../Mustache.C

+JobFlavour             = "{queue}"
queue arguments from arguments.txt

+AccountingGroup = "group_u_CMS.CAF.COMM"
'''

condor = condor.replace("{queue}", args.queue)
condor = condor.replace("{wp_file}", args.wp_file)

script = '''#!/bin/sh -e

source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc12-opt/setup.sh

JOBID=$1;  
INPUTFILE=$2;
OUTPUTDIR=$3;
ASSOC=$4;
WPFILE=$5;
ET_SEED=$6;
PULIM=$7;


echo -e "Running ndjson dumper.."

python cluster_ndjson_byevent.py -i ${INPUTFILE} -o output.ndjson \
            -a ${ASSOC} --wp-file ${WPFILE} --min-et-seed ${ET_SEED} \
            --max-et-seed {max_et_seed} --max-et-isolated-cl {max_et_isolated_cl}  {overlap} \
            --nocalomatched-nmax {max-nocalomatched} --pu-limit ${PULIM} {debug};

{compress}
echo -e "Copying result to: $OUTPUTDIR";
xrdcp -f --nopbar  output.{output_ext} root://eos{eosinstance}.cern.ch/${OUTPUTDIR}/clusters_data_${JOBID}.{output_ext};
xrdcp -f --nopbar  output.meta.csv root://eos{eosinstance}.cern.ch/${OUTPUTDIR}/clusters_data_${JOBID}.meta.csv;

echo -e "DONE";
'''

script = script.replace("{eosinstance}", args.eos)
if args.compress:
    script = script.replace("{compress}", 'tar -zcf output.ndjson.tar.gz output.ndjson')
    script = script.replace("{output_ext}", 'ndjson.tar.gz')
else:
    script = script.replace("{compress}", '')
    script = script.replace("{output_ext}", 'ndjson')
if args.debug:
    script = script.replace("{debug}", "--debug")
else: 
    script = script.replace("{debug}", "")
if args.overlap:
    script = script.replace("{overlap}", "--overlap")
else:
    script = script.replace("{overlap}", "")
if args.nocalomatched_nmax:
    script = script.replace("{max-nocalomatched}", str(args.nocalomatched_nmax))
else:
    script = script.replace("{max-nocalomatched}", "10000")
    
script = script.replace("{max_et_seed}", str(args.max_et_seed))
script = script.replace("{max_et_isolated_cl}", str(args.max_et_isolated_cl))
    
arguments= []
if not os.path.exists(args.outputdir):
    os.makedirs(args.outputdir)

inputfiles = glob(args.inputdir + "/**/**.root", recursive=True)
ninputfiles = len(inputfiles)
# template_inputfile = "cluster_job{}_step2_output.root"

wp_file = os.path.split(args.wp_file)[1]

print("N input files: ", ninputfiles)


jobid = 0
files_groups = []
ifile_used = 0
ifile_curr = 0


for file in inputfiles:
    files_groups.append(file)
    ifile_used +=1 
    ifile_curr +=1

    if len(files_groups) == args.nfile_group:
        jobid +=1
        #join input files by ;
        arguments.append("{} {} {} {} {} {} {}".format(
                jobid,"#_#".join(files_groups), args.outputdir, args.assoc_strategy, wp_file,
                args.min_et_seed, args.pu_limit))
        files_groups = []
        ifile_group = 0

# Join also the last group
if len(files_groups):
    arguments.append("{} {} {} {} {} {} {}".format(
                    jobid+1,"#_#".join(files_groups), args.outputdir, args.assoc_strategy,wp_file,
                    args.min_et_seed, args.pu_limit))

print("N. jobs ", len(arguments))


with open(args.condor_folder + "/condor_job.txt", "w") as cnd_out:
    cnd_out.write(condor)

with open(args.condor_folder + "/arguments.txt", "w") as arg:
    arg.write("\n".join(arguments))

with open(args.condor_folder + "/run_ndjson_byevent_script.sh", "w") as rs:
    rs.write(script)

#os.system("condor_submit condor_job.txt")




