import sys
import os
import argparse
from glob import glob
# source /cvmfs/sft.cern.ch/lcg/views/LCG_102/x86_64-centos7-gcc11-opt/setup.hs

with open("command.txt", "w") as of:
    of.write(" ".join(["python"]+sys.argv))

'''
This scripts runs hadd on single crystal files to 
group them in strips reading a DOF file
'''
parser = argparse.ArgumentParser()

parser.add_argument("-i", "--inputdir", type=str, help="Inputdir", required=True)
parser.add_argument("-nfg", "--nfile-group", type=int, help="How many files per tfrecord file", required=True)
parser.add_argument("-nef", "--nevents-file", type=int, help="How many files per tfrecord file", required=True)
parser.add_argument("-o", "--outputdir", type=str, help="Outputdir", required=True)
parser.add_argument("-q", "--queue", type=str, help="Condor queue", default="longlunch", required=True)
parser.add_argument("-cf","--condor-folder", type=str,  help="Condor folder", default="condor_ndjson")
args = parser.parse_args()


# Create output folder for jobs configuration
os.makedirs(args.condor_folder+"/error", exist_ok=True)
os.makedirs(args.condor_folder+"/output", exist_ok=True)
os.makedirs(args.condor_folder+"/log", exist_ok=True)



# Prepare condor jobs
condor = '''executable              = run_torchgeom.sh
output                  = output/strips.$(ClusterId).$(ProcId).out
error                   = error/strips.$(ClusterId).$(ProcId).err
log                     = log/strips.$(ClusterId).log
transfer_input_files    = ../convert_torchgeom_dataset.py

+JobFlavour             = "{queue}"
queue arguments from arguments.txt

+AccountingGroup = "group_u_CMS.CAF.COMM"
'''

condor = condor.replace("{queue}", args.queue)

script = '''#!/bin/sh -e

source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh

JOBID=$1;  
INPUTFILE=$2;
OUTPUTDIR=$3;
NFILES=$4;


echo -e "Running torch geom dumper.."

mkdir output;
python convert_torchgeom_dataset.py  ${INPUTFILE} ./output ${JOBID} ${NFILES}
            
echo -e "Copying result to: $OUTPUTDIR";
rsync -avz output/ ${OUTPUTDIR}

echo -e "DONE";
'''


arguments= []
if not os.path.exists(args.outputdir):
    os.makedirs(args.outputdir)

inputfiles = glob(args.inputdir + "/**.ndjson.tar.gz", recursive=True)
ninputfiles = len(inputfiles)

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
        arguments.append("{} {} {} {}".format(
                jobid,"#_#".join(files_groups), args.outputdir, args.nevents_file))
        files_groups = []
        ifile_group = 0


if len(files_groups)>0:
# Join also the last group
    arguments.append("{} {} {} {}".format(
                    jobid+1,"#_#".join(files_groups), args.outputdir,args.nevents_file))



print("Njobs: ", len(arguments))
    
with open(args.condor_folder + "/condor_job.txt", "w") as cnd_out:
    cnd_out.write(condor)

with open(args.condor_folder + "/arguments.txt", "w") as arg:
    arg.write("\n".join(arguments))

with open(args.condor_folder + "/run_torchgeom.sh", "w") as rs:
    rs.write(script)

#os.system("condor_submit condor_job.txt")




