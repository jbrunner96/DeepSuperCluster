import gzip 
import json
import tarfile
import numpy as np
import os
from glob import glob
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset,  Dataset
from torch_geometric.transforms import NormalizeFeatures
from torch.utils.data import IterableDataset
from torch_geometric.data import Data

import torch.nn.functional as F
import sys

def read_file(filename):
    with tarfile.open(filename, "r:gz") as tar:
        # Find the first file inside the archive
        for member in tar.getmembers():
            if member.isfile():  # Ensure it's a file
                extracted_file = tar.extractfile(member)  
                if extracted_file:
                    data_json = [json.loads(line) for line in extracted_file]
    return data_json



def convert_to_tensor(features_dict):
    # Convert dictionary of lists to a single tensor
    features = torch.tensor([features_dict[key] for key in features_dict], dtype=torch.float).T
    return features

def create_data_object(nodes_features, nodes_sim_features, edges_idx, edges_labels):
    # Convert nodes features and labels to tensors
    x = convert_to_tensor(nodes_features)
    y = convert_to_tensor(nodes_sim_features)
    
    # Convert edge indices and labels to tensors
    edge_index = torch.tensor(edges_idx, dtype=torch.long).T
    edge_attr = torch.tensor([edges_labels[key] for key in edges_labels], dtype=torch.float).T
    
    # Create a Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return data


if __name__ == "__main__":
    print(sys.argv)
    input_files = sys.argv[1]
    output_dir = sys.argv[2]
    index_file = int(sys.argv[3])
    graphs_in_file = int(sys.argv[4])

    input_files = input_files.split("#_#")
    for input_file in input_files:
        print("Working on file: ", input_file)
        dataset = read_file(input_file)
        graphs_in_group = [ ]
        ifiles = 0
        for igraph, graph in enumerate(dataset):
            # Create PyG data object
            data = create_data_object(
                nodes_features=graph['nodes_features'],
                nodes_sim_features=graph['nodes_sim_features'],
                edges_idx=graph['edges_idx'],
                edges_labels=graph['edges_labels']
            )

            graphs_in_group.append(data)

            if len(graphs_in_group) >= graphs_in_file or \
               igraph == (len(dataset)-1):
                print(f"Saving file, {igraph}, {len(graphs_in_group)}")
                group_data_list = graphs_in_group
                group_data, slices = torch_geometric.data.InMemoryDataset.collate(group_data_list)

                save_path = os.path.join(
                    output_dir,
                    f'graph_data_group_{index_file}_{ifiles}.pt')
                torch.save((group_data, slices), save_path)
                graphs_in_group = []
                ifiles += 1
