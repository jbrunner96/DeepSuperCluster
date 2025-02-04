import torch
from torch_geometric.data import Data
import glob 
import gzip 
import json
import tarfile
import numpy as np

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
