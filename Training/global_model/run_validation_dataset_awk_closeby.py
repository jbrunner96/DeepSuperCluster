import numpy as np
import awk_data
import awkward as ak
import pandas as pd
import tensorflow as tf
import loader_awk
import argparse 
from collections import defaultdict
import os
import json
from time import time

parser = argparse.ArgumentParser()

parser.add_argument("--model-config", type=str, help="Model configuration", required=True)
parser.add_argument("--model-weights", type=str, help="Model weights", required=True)
parser.add_argument("--conf-overwrite", type=str, help="Validation config overwrite", required=False)
parser.add_argument("-o", "--outputdir", type=str, help="Outputdir", required=True)
parser.add_argument("--diff-model", action="store_true", help="Model with different outputs than baseline model")
parser.add_argument("--flavour", type=str, help='Choose dataset: ele1, gamma1, ele2, gamma2', required=False)
args = parser.parse_args()


config = json.load(open(args.model_config))

features_dict = config["dataset_conf"]["validation"]["columns"]
# The seed features are taken directly from the cluster
features_dict["seed_features"] = [ ]
for cl_f in features_dict["cl_features"]:
    features_dict["seed_features"] = cl_f.replace("cluster","seed")

if args.conf_overwrite != None and args.conf_overwrite!= "" and args.conf_overwrite!="None":
    config_overwrite = json.load(open(args.conf_overwrite))
else:
    config_overwrite = None


    
N_metadata = len(features_dict['window_metadata'])
N_seed_features = len(features_dict['seed_features'])

print("N. metadata: ", N_metadata)
print("N. seed features: ", N_seed_features)
print(">> Load the dataset manually to be able to use all the features")

print(">> Load the dataset and model")
model, dataset, cfg = loader_awk.get_model_and_dataset(args.model_config, args.model_weights,
                                                       training=False,
                                                       awk_dataset=True,
                                                       overwrite=config_overwrite)

include_rechits = cfg.include_rechits
batch_size = cfg.batch_size
print(">> Model successfully loaded")

print(">> Starting to run on events: ")
data = defaultdict(list)
lastT = time()

batches = []
wseed_cseed_list = defaultdict(list) # store data of all windowseeds that are caloseeds
not_wseed_cseed_list = defaultdict(list) # store data of all caloseed clusters that are not the seed of their window
closeby_list = defaultdict(list) # final output

for ib, el in enumerate(dataset):
    if ib % 10 == 0: 
        now = time()
        rate = 10* batch_size / (now-lastT)
        lastT = now
        nsecond = (cfg.maxevents - batch_size*ib) / rate
        print("Events: {} ({:.1f}Hz). Eta: {:.0f}:{:.0f}".format(ib*batch_size, rate, nsecond//60, nsecond%60))

    (X,y_true, weight), df = el
        
    y_out = model(X, training=False)

    if include_rechits:
        if args.diff_model:
            # hard code model inputs for model different than baseline model
            cl_X_initial, wind_X, cl_hits, is_seed, mask_cls, mask_rechits = X
        else:
            cl_X_initial, wind_X, cl_hits, is_seed, mask_cls, mask_rechits = X
    else:
        if args.diff_model:
            # hard code model inputs for model different than baseline model
            cl_X_initial, wind_X,  is_seed, mask_cls = X 
        else:
            cl_X_initial, wind_X,  is_seed, mask_cls = X

    if args.diff_model:
        # hard code model outputs for model different than baseline model
        (dense_clclass,dense_windclass, en_regr_factor, is_seed_calo_seed),  mask_cls  = y_out
        y_clclass, y_windclass, cl_X, wind_X, y_metadata, y_is_seed_calo_seed = y_true
    else:
        # baseline model outputs
        (dense_clclass,dense_windclass, en_regr_factor),  mask_cls  = y_out
        y_clclass, y_windclass, cl_X, wind_X, y_metadata = y_true
    

    # we build up our two lists from which we later extract the closeby events
    mask = (df.window_metadata.is_seed_calo_seed == True) # only keep windows where seed is caloseed
    df_caloseed = df[mask]

    wseed_cseed_list["et"].extend(ak.to_numpy(df_caloseed.meta_seed_features.et_seed))
    wseed_cseed_list["eta"].extend(ak.to_numpy(df_caloseed.meta_seed_features.seed_eta))
    wseed_cseed_list["phi"].extend(ak.to_numpy(df_caloseed.meta_seed_features.seed_phi))
    wseed_cseed_list["en"].extend(ak.to_numpy(df_caloseed.meta_seed_features.en_seed))
    wseed_cseed_list["ncls"].extend(ak.to_numpy(df_caloseed.meta_seed_features.seed_nxtals))

    wseed_cseed_list["wind_id"].extend(ak.to_numpy(df_caloseed.window_metadata.window_index))
    #wseed_cseed_list["event_id"].append(ak.to_numpy(df_caloseed.window_metadata.event_id))
    #wseed_cseed_list["lumi_id"].append(ak.to_numpy(df_caloseed.window_metadata.lumi_id))
    #wseed_cseed_list["run_id"].append(ak.to_numpy(df_caloseed.window_metadata.run_id))

    flat_cl_indices = ak.flatten(df_caloseed.cl_metadata.cl_index, axis=None)
    flat_seeds = ak.flatten(df_caloseed.cl_labels.is_seed, axis=None)
    seed_cl_indices = flat_cl_indices[flat_seeds]
    #wseed_cseed_list["cl_index"].append(ak.to_numpy(seed_cl_indices))

    # (event_id, lumi_id, run_id, cl_index)
    unique_id = ak.zip([df_caloseed.window_metadata.event_id, 
                df_caloseed.window_metadata.lumi_id,
                df_caloseed.window_metadata.run_id,
                seed_cl_indices])
    unique_id = ak.to_numpy(unique_id)
    unique_id= [tuple(map(int, row)) for row in ak.to_list(unique_id)]

    wseed_cseed_list["unique_id"].extend(unique_id)
    wseed_cseed_list["pred"].extend(tf.reshape(tf.nn.sigmoid(is_seed_calo_seed), [-1]).numpy()[mask])





    # now we get all caloseed_cl!=windowseed in windows with windowseed=caloseed

    cl_mask = ak.flatten(df_caloseed.cl_labels.is_calo_seed & (~df_caloseed.cl_labels.is_seed)) # mask of all those clusters

    not_wseed_cseed_list["et"].extend(ak.to_numpy(ak.flatten(df_caloseed.cl_features.et_cluster)[cl_mask]))
    not_wseed_cseed_list["eta"].extend(ak.to_numpy(ak.flatten(df_caloseed.cl_features.cluster_eta)[cl_mask]))
    not_wseed_cseed_list["phi"].extend(ak.to_numpy(ak.flatten(df_caloseed.cl_features.cluster_phi)[cl_mask]))
    not_wseed_cseed_list["en"].extend(ak.to_numpy(ak.flatten(df_caloseed.cl_features.en_cluster)[cl_mask]))
    not_wseed_cseed_list["ncls"].extend(ak.to_numpy(ak.flatten(df_caloseed.cl_features.cl_nxtals)[cl_mask]))

    _ , window_ids_broadcasted = ak.broadcast_arrays(df_caloseed.cl_features.et_cluster, df_caloseed.window_metadata.window_index)
    window_ids_broadcasted = ak.flatten(window_ids_broadcasted)
    window_ids_broadcasted = [''.join(chars) for chars in ak.to_list(window_ids_broadcasted)]
    window_ids_broadcasted = [wid for wid, keep_id in zip(window_ids_broadcasted, cl_mask) if keep_id]
    not_wseed_cseed_list["wind_id"].extend(window_ids_broadcasted)

    #not_wseed_cseed_list["wind_id"].extend(ak.to_numpy(window_ids_broadcasted[cl_mask]))

    _ , lumi_ids_broadcasted = ak.broadcast_arrays(df_caloseed.cl_features.et_cluster, df_caloseed.window_metadata.lumi_id)
    _ , event_ids_broadcasted = ak.broadcast_arrays(df_caloseed.cl_features.et_cluster, df_caloseed.window_metadata.event_id)
    _ , run_ids_broadcasted = ak.broadcast_arrays(df_caloseed.cl_features.et_cluster, df_caloseed.window_metadata.run_id)
    lumi_ids_broadcasted = ak.flatten(lumi_ids_broadcasted)
    event_ids_broadcasted = ak.flatten(event_ids_broadcasted)
    run_ids_broadcasted = ak.flatten(run_ids_broadcasted)

    cl_indices = ak.flatten(df_caloseed.cl_metadata.cl_index, axis=None)[cl_mask]

    # (event_id, lumi_id, run_id, cl_index)
    unique_id = ak.zip([event_ids_broadcasted[cl_mask], 
                lumi_ids_broadcasted[cl_mask],
                run_ids_broadcasted[cl_mask],
                cl_indices])

    unique_id = ak.to_numpy(unique_id)
    unique_id = [tuple(map(int, row)) for row in ak.to_list(unique_id)] # convert to tuple

    not_wseed_cseed_list["unique_id"].extend(unique_id)





# now we build the closeby events dataset

wseed_ids = set(wseed_cseed_list["unique_id"])  # for fast lookup
not_wseed_ids = not_wseed_cseed_list["unique_id"] # ids of potential closeby candidates

mask = [uid in wseed_ids for uid in not_wseed_ids] # true if caloseed cluster in a window is the seed of another window

for i, i_mask in enumerate(mask):
    if i_mask:
        wind_id = not_wseed_cseed_list["wind_id"][i]
        index_1 = wseed_cseed_list["wind_id"].index(wind_id) # find out in which window the cluster appears as window seed
        index_2 = wseed_cseed_list["unique_id"].index(not_wseed_cseed_list["unique_id"][i])

        # cluster 1 is the seed cluster that has the same (event, lumi, run, cl_index) as the non seed cluster
        closeby_list["et_1"].append(wseed_cseed_list["et"][index_1])
        closeby_list["eta_1"].append(wseed_cseed_list["eta"][index_1])
        closeby_list["phi_1"].append(wseed_cseed_list["phi"][index_1])
        closeby_list["en_1"].append(wseed_cseed_list["en"][index_1])
        closeby_list["ncls_1"].append(wseed_cseed_list["ncls"][index_1])
        closeby_list["pred_1"].append(wseed_cseed_list["pred"][index_1])

        # cluster 2 is the windowseed of the window where the non-caloseed cluster was in
        closeby_list["et_2"].append(wseed_cseed_list["et"][index_2])
        closeby_list["eta_2"].append(wseed_cseed_list["eta"][index_2])
        closeby_list["phi_2"].append(wseed_cseed_list["phi"][index_2])
        closeby_list["en_2"].append(wseed_cseed_list["en"][index_2])
        closeby_list["ncls_2"].append(wseed_cseed_list["ncls"][index_2])
        closeby_list["pred_2"].append(wseed_cseed_list["pred"][index_2])


df_closeby = pd.DataFrame(closeby_list)
os.makedirs(args.outputdir, exist_ok=True)

print("Saving on disk")

if args.flavour == "ele2":
    df_closeby.to_csv(args.outputdir +"/closeby_double_ele.csv", sep=";",index=False)
if args.flavour == "gamma2":
    df_closeby.to_csv(args.outputdir +"/closeby_double_gamma.csv", sep=";",index=False)


print("DONE!")
