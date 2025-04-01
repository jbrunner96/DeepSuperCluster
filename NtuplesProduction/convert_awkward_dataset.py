import json
import os
import collections
import sys
import glob
import gzip
import argparse 
import awkward as ak
from glob import glob

# source /cvmfs/sft.cern.ch/lcg/views/LCG_101/x86_64-centos7-gcc11-opt/setup.sh

parser = argparse.ArgumentParser()
parser.add_argument("-n","--name", type=str, help="Job name", required=True)
parser.add_argument("-i","--input", type=str, help="input dir or files list", required=True)
parser.add_argument("-o","--outputdir", type=str, help="Outputdirectory",required=True)
parser.add_argument("-f","--features-def", type=str, help="Features definition file", default="features_definition.json")
parser.add_argument("-g","--groupfiles", type=int, help="N. input file for each output file",default=1)
parser.add_argument("-s","--standalone", action="store_true", help="Run without condor")
parser.add_argument("--flavour", type=int, help="PdgID flavor to add to the dataset", default=11)
args = parser.parse_args()

features_dict = json.load(open(args.features_def))["features_dict"]

def load_iter(files):
    for filename in files:
        with gzip.open(filename, "rt") as file:
            lines = []
            for line in file.readlines():
                line = line[line.rfind('{"run_id"'):]
                if "}" in line:
                    line = line.replace("NaN", "-999")
                    lines.append(line)
            yield "[{}]".format(",".join(lines))
                                   

def convert_to_features_awkward(file, flavour):
    df = ak.from_json(file)
    out = {}
    for k,v in features_dict.items():
        if k != "hits_indices":
            if k.startswith("cl_"):
                out[k] = df.clusters[v]
            else:
                out[k] = df[v]
    # Getting a subset of the hits array
    hits_index = ak.local_index(df.clusters.cl_hits)
    mask_hits_index = (hits_index < 0) # all false
    for i in features_dict["hits_indices"]:
        mask_hits_index = mask_hits_index | (hits_index == i)
    out["cl_h"] = df.clusters.cl_hits[mask_hits_index]
    # add the flavour info to each line    
    flavour = ak.ones_like(df.en_seed) * flavour
    out["window_metadata"] = ak.with_field(out["window_metadata"], flavour, "flavour")
    return ak.Array(out)

def finalize_output(arrays, directory, name):
    totA = ak.concatenate(arrays)
    ak.to_parquet(totA, name)
    #ak.to_parquet.dataset(directory) --> to be done externally at the end
    
if __name__ == "__main__":
    
    outputdir =args.outputdir

    os.makedirs(outputdir, exist_ok=True)

    if args.standalone:
        inputfiles = glob(args.input + "/**.ndjson.tar.gz", recursive=True)
    else:
        if "#_#" in args.input: 
            inputfiles = args.input.split("#_#")
        else:
            inputfiles = [args.input]
    ninputfiles = len(inputfiles)
        
    print("Start reading files")
    ig = 0
    iG = 1
    arrays = []
    for file in load_iter(inputfiles):
        print("Processing file {}, in group {}".format(ig, iG))
        arrays.append(convert_to_features_awkward(file, args.flavour))
        ig +=1
        if ig == args.groupfiles:
            if args.standalone:
                finalize_output(arrays, args.outputdir, args.outputdir + "/"+args.name + ".{}.parquet".format(iG) )
            else:
                finalize_output(arrays, args.outputdir, args.outputdir + "/"+args.name )
            iG+=1
            ig = 0
            arrays.clear()
        

    if len(arrays):
        if args.standalone:
            finalize_output(arrays, args.outputdir, args.outputdir + "/"+args.name + ".{}.parquet".format(iG) )
        else:
            finalize_output(arrays, args.outputdir, args.outputdir + "/"+args.name )
        arrays.clear()
        
    print("DONE!")
