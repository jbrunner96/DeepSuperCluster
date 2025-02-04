import os
import pandas as pd 
import glob
import gzip
import argparse
import ast

parser = argparse.ArgumentParser()
parser.add_argument("-i","--inputdir", type=str, help="inputfolder", required=True)
parser.add_argument("-o","--outputfile", type=str, help="outputfile",required=True)
parser.add_argument("-n","--number", type=int, help="number of files to merge", default=-1)
args = parser.parse_args()

                    
for ty in ["object"]:
    ifile = 0
     # Open the HDF5 store
    store = pd.HDFStore(args.outputfile.replace("{type}", ty))
    
    for f in glob.glob(args.inputdir+"/{}_data*.tar.gz".format(ty)):
        if args.number > 0 and ifile >= args.number: break
        ifile += 1
        print(f)
        with gzip.open(f, "rt") as file:
            def convert_to_list(val):
                if val == "":
                    return []
                else:
                    return ast.literal_eval(val)

            # Load the DataFrame, applying the conversion function to the 'Col3' column
            df = pd.read_csv(file, sep=";", converters={
                'ele_clsAdded_eta': convert_to_list,
                'ele_clsAdded_phi': convert_to_list,
                'ele_clsAdded_energy': convert_to_list,
                'ele_clsRemoved_eta': convert_to_list,
                'ele_clsRemoved_phi': convert_to_list,
                'ele_clsRemoved_energy': convert_to_list
            })
            # some hacks to fix the data
            df.rename(columns={"output_object.csv":"genpart_index"}, inplace=True)
            df = df.iloc[:-1]
            df["ele_passConversionVeto"] = (df.ele_passConversionVeto==True).astype(int)
            # Append the DataFrame to the store
            # Remove the last row
            store.append('df', df, index=False)
    # Close the store
    store.close()
