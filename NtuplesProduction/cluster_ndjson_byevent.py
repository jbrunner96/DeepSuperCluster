from __future__ import print_function
import ROOT as R 
R.gROOT.SetBatch(True)
#R.PyConfig.IgnoreCommandLineOptions = True

import sys 
import os
from collections import defaultdict
from itertools import islice, chain
from numpy import mean
from array import array
from math import sqrt
from pprint import pprint as pp
import numpy as np
import argparse
import pickle
import pandas as pd
from windows_creator_graph_byevent import WindowCreator

parser = argparse.ArgumentParser()
parser.add_argument("-i","--inputfile", type=str, help="inputfile", required=True)
parser.add_argument("-o","--outputfile", type=str, help="outputfile", default="clusters_data.pkl")
parser.add_argument("-a","--assoc-strategy", type=str, help="Association strategy", default="sim_fraction")
parser.add_argument("--wp-file", type=str,  help="File with sim fraction thresholds")
parser.add_argument("-n","--nevents", type=int,nargs="+", help="n events iterator", required=False)
parser.add_argument("-d","--debug", action="store_true",  help="debug", default=False)
parser.add_argument("--min-et-seed", type=float,  help="Min Et of the seeds", default=1.)
parser.add_argument("--max-et-seed", type=float,  help="Max Et of the seeds", default=1e6)
parser.add_argument("--max-et-isolated-cl", type=float,  help="Max Et of the isolated clusters", default=1e6)
parser.add_argument("--nocalomatched-nmax", type=int,  help="Max number of calomatched clusters", default=10)
parser.add_argument("--pu-limit", type=float,  help="SimEnergy PU limit", default=1e6)
args = parser.parse_args()

if "#_#" in args.inputfile: 
    inputfiles = args.inputfile.split("#_#")
else:
    inputfiles = [args.inputfile]

# simfraction_thresholds_file = R.TFile(args.wp_file)
# simfraction_thresholds = simfraction_thresholds_file.Get("h2_Minimum_simScore_seedBins")

# Parameters controlling the creation of the window
# min simFraction for the seed with a signal caloparticle
SEED_MIN_FRACTION=1e-2
# min simFraction for the cluster to be associated with the caloparticle
CL_MIN_FRACION=1e-4
# threshold of simEnergy PU / simEnergy signal for each cluster and seed to be matched with a caloparticle
SIMENERGY_PU_LIMIT= args.pu_limit

windows_creator = WindowCreator(args.wp_file, SEED_MIN_FRACTION,
                                cl_min_fraction=CL_MIN_FRACION,
                                simenergy_pu_limit = SIMENERGY_PU_LIMIT,
                                min_et_seed=args.min_et_seed,
                                max_et_seed=args.max_et_seed,
                                max_et_isolated_cl=args.max_et_isolated_cl,
                                assoc_strategy=args.assoc_strategy,
                                nocalomatchedNmax=args.nocalomatched_nmax)

debug = args.debug

metadata = []
windows_files = open(args.outputfile, "w")

all_metadata = [ ] 

for inputfile in inputfiles:
    f = R.TFile(inputfile);
    tree = f.Get("recosimdumper/caloTree")

    print ("Starting")
    for iev, event in enumerate(tree):
        if iev % 10 == 0: print(".",end="")
        windows_data, debug_metadata = windows_creator.get_windows(event, debug= args.debug )
        all_metadata.append(debug_metadata)
        for wd in windows_data:
            windows_files.write(wd + '\n') # 1 event==graph for call
    f.Close()

windows_files.close()
meta = pd.DataFrame(all_metadata)
meta.to_csv("output.meta.csv", sep=';', index=False)
        
