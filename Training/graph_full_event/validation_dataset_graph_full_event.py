import torch
import numpy as np
import argparse
import os
import json
from dataset_loader import ECALGraphDataset_Validation
from collections import defaultdict
from torch_geometric.loader import DataLoader
from models_archive_graph_full_event.GCN import GCN
from models_archive_graph_full_event.GAT import GAT
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument("--config", type=str, help="Config", required=True)
parser.add_argument("--model", type=str, help="Model .py", required=True)
parser.add_argument("--model-weights", type=str, help="Model weights", required=True)
#parser.add_argument("--name", type=str, help="Model version name", required=True) use name from input
parser.add_argument("--outputdir", type=str,help="Output folder", required=True)


args = parser.parse_args()



# load config (should be stored in the models archive)
config = json.load(open(args.config))

os.makedirs(args.outputdir, exist_ok=True)


# TO DO: GET THIS FROM THE MODELS ARCHIVE ON EOS
model = GAT(config["nnodes"])

model.load_state_dict(torch.load(args.model_weights, weights_only=True))
model.eval()


# get all the file paths of the current flavour (1ele, 2ele, 1gamma, 2gamma)
# note: it is set to training as all files are found in the same folder for the full graph but i didnt change the paths for validation/test folders
flavour_list = config["dataset_conf"]["training"]["input_folders"]

# keys for awk_array
# cl_features and cl_labels are according to the windows creator
cl_features_dict = ["en", "et", "en_calib", "et_calib", "eta", "phi", "ieta", "iphi", "iz", "f5_r9", "f5_r9_sigmaIetaIeta", "f5_r9_sigmaIetaIphi", "f5_r9_sigmaIphiIphi", "f5_swissCross", "r9", "sigmaIetaIeta", "sigmaIetaIphi", "sigmaIphiIphi", "etaWidth", "phiWidth", "nxtals"]

# we dont want to store all the features. note indices of features we want to keep
cl_features_mask = [0,1,2,3,4,5,18,19,20]

cl_labels_dict = ["is_calo_matched", "is_calo_seed", "calo_index", "calo_score", "en_true_sim", "et_true_sim", "en_true_gen", "et_true_gen"]
cl_predictions_dict = ["is_calo_seed_prediction"]

parquet_names = ["validation_data_single_electron.parquet", "validation_data_double_electron.parquet", "validation_data_single_gamma.parquet", "validation_data_double_gamma.parquet"]
csv_names = ["closeby_single_electron.csv", "closeby_double_electron.csv", "closeby_single_gamma.csv", "closeby_double_gamma.csv"]


for flavour_id, flavour in enumerate(flavour_list):

        # get all validation datasets of current flavour
        folder_path = os.path.join(flavour, "processed", "validation")
        validation_folder = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        event_list = [] # store data by event
        closeby_list = defaultdict(list) # store the closeby caloseed cluster pairs

        for validation_path in validation_folder:
                
                # data.x will contain the cl_features of all graphs
                # indices contains information where a graph starts and where it ends
                data, indices = torch.load(validation_path)

                #data.x[0:169,0] #energies of the clusters of the first graph
                #data.x[:, 0] #energies of all clusters in all graphs

                x = data.x # get the cluster features: en, ieta, iphi, ...
                y = data.y # get the cluster labels, calomatched, caloseed, ...
                edges = data.edge_index #get the edge indices of the graph

                # y.shape has shape [#clusters, 8=cluster_labels]

                graph_node_boundaries = indices["x"].tolist() # a list of graph start/endpoints for the nodes
                graph_edge_boundaries = indices["edge_index"].tolist() # a list of the graph start/endpoints for the edges


                print("Load Dataset")
                # efficiently get all the model predictions for the first file
                full_dataset_loader = ECALGraphDataset_Validation(validation_path)
                

                # load dataset as a single batch
                full_dataset_graph =  DataLoader(
                        full_dataset_loader,
                        batch_size=len(graph_node_boundaries)-1,
                        num_workers=1,
                        pin_memory=True
                )

                # get the batch containing all the graphs
                batch = next(iter(full_dataset_graph))


                # get predictions of the full dataset
                with torch.no_grad():
                        out = model(batch.x, batch.edge_index)
                        prediction = torch.sigmoid(out).squeeze()

                # finally: iterate graph by graph over the full loaded file and store the individual graphs
                for i in range(len(graph_node_boundaries)-1):
                        # create classifier validation dataset
                        start_node, end_node = graph_node_boundaries[i], graph_node_boundaries[i+1] # get graph start and end indices of the nodes
                        start_edge, end_edge = graph_edge_boundaries[i], graph_edge_boundaries[i+1] # get graph start and end indices of the edges
                        

                        event_dict = {
                                "cl_features": {cl_features_dict[j]: np.array(x[start_node:end_node, j]) for j in cl_features_mask},
                                "cl_labels": {cl_labels_dict[j]: np.array(y[start_node:end_node, j]) for j in range(len(cl_labels_dict))},
                                "cl_predictions": {cl_predictions_dict[j]: np.array(prediction[start_node:end_node]) for j in range(len(cl_predictions_dict))}
                        }

                        # add the event to the list
                        event_list.append(event_dict)

                        
                        
                        if i % 10000 == 0:
                                print(i, "event out of ", len(graph_node_boundaries)-1, " done")
                
                # create the close-by dataset, this can be done for the full file at once

                caloseed_source = (batch.y[:,1][batch.edge_index[0]] == 1) # mask for all source clusters that are caloseeds
                caloseed_target = (batch.y[:,1][batch.edge_index[1]] == 1) # mask for all target clusters that are caloseeds

                connected_mask = caloseed_source & caloseed_target # mask that is True if both source and target of an edge are caloseeds, else False
                closeby_edge_ids = torch.nonzero(connected_mask, as_tuple=True)[0] # indices of edge tensor where we have a close-by pair

                closeby_ids = batch.edge_index[:,closeby_edge_ids] # source and target indices of all closeby events

                source_id = closeby_ids[0].numpy() # cluster ids of all sources that are caloseed and connected to caloseed
                target_id = closeby_ids[1].numpy() # cluster ids of all targets = windowseeds that are caloseed and connected to caloseed

                
                # we want to avoid double counting. most often two cluster appear as (icl1, icl2) and (icl2, icl1)
                
                source_unique, target_unique = [], []
                unique_pairs = []

                for i in range(len(source_id)):
                        pair = tuple(sorted((source_id[i], target_id[i])))
                        if pair not in unique_pairs:
                                unique_pairs.append(pair)
                                source_unique.append(source_id[i])
                                target_unique.append(target_id[i])

                source_unique = np.array(source_unique)
                target_unique = np.array(target_unique)
                
                pred_source = prediction[source_unique].numpy() # get the classifier predictions for all sources
                pred_target = prediction[target_unique].numpy() # get the classifier predictions for all targets

                # build dataset for all close-by events for the full file
                closeby_list["En_source"].append(batch.x[:,0][source_unique].numpy())
                closeby_list["En_target"].append(batch.x[:,0][target_unique].numpy())

                closeby_list["Et_source"].append(batch.x[:,1][source_unique].numpy())
                closeby_list["Et_target"].append(batch.x[:,1][target_unique].numpy())

                closeby_list["eta_source"].append(batch.x[:,4][source_unique].numpy())
                closeby_list["eta_target"].append(batch.x[:,4][target_unique].numpy())

                closeby_list["phi_source"].append(batch.x[:,5][source_unique].numpy())
                closeby_list["phi_target"].append(batch.x[:,5][target_unique].numpy())

                closeby_list["ncls_source"].append(batch.x[:,20][source_unique].numpy())
                closeby_list["ncls_target"].append(batch.x[:,20][target_unique].numpy())

                closeby_list["pred_source"].append(pred_source)
                closeby_list["pred_target"].append(pred_target)





                


        print("Saving to disk: ", parquet_names[flavour_id])

        # save event as DataFrame. This is much more efficient than saving as awk Array. Due to the structure we can recover the parquet file as awk array anyways
        df = pd.DataFrame(event_list)
        df.to_parquet(os.path.join(args.outputdir, parquet_names[flavour_id]))

        print("Saving to disk: ", csv_names[flavour_id])

        # also save the close-by dataset
        closeby_final = {}
        for k, v in closeby_list.items():
                closeby_final[k] = np.concatenate(v)

        df_closeby_final = pd.DataFrame(closeby_final)
        df_closeby_final.to_csv(os.path.join(args.outputdir, csv_names[flavour_id]) , sep=";",index=False)
