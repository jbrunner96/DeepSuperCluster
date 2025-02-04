import gzip 
import json
import tarfile
import numpy as np
import math
import os
from glob import glob
import numpy as np
import uproot
import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset,  Dataset
from torch_geometric.transforms import NormalizeFeatures
from torch.utils.data import IterableDataset
from torch_geometric.loader import DataLoader

from torch_geometric.data import Data

#from torch.utils.data import Dataset
import torch.nn.functional as F

from dataset_utils import read_file, convert_to_tensor, create_data_object

class ECALGraphDataset(IterableDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, graphs_in_file=512):
        self.root = root
        self.transform = transform
        self.pre_transform =  pre_transform
        self.pre_filter = pre_filter
        self.raw_dir = self.root + "/raw"
        self.processed_dir = self.root+ "/processed"
        self.graphs_in_file = graphs_in_file
        self._nfiles = len(self.processed_files)
        self._start = 0
        self._end = self._nfiles
        
    @property
    def raw_paths(self):
        """Returns expected raw files (not paths)."""
        return [f  for f in glob.glob(os.path.join(self.root, "raw", "*.tar.gz"))]

    @property
    def processed_files(self):
        """Retrieve the list of processed .pt files."""
        processed_dir = os.path.join(self.root, "processed")
        return sorted(f for f in os.listdir(processed_dir) if f.endswith('.pt'))

    def _preprocess(self, args):
        file, group_idx = args
        dataset = read_file(file)
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
            
            # Apply pre-filter if specified
            if self.pre_filter and not self.pre_filter(data):
                continue
                
            # Apply pre-transform if specified
            if self.pre_transform:
                data = self.pre_transform(data)
                
            graphs_in_group.append(data)

            if len(graphs_in_group) >= self.graphs_in_file or \
                   igraph == (len(dataset)-1):
                print(f"Saving file, {igraph}, {len(graphs_in_group)}")
                group_data_list = graphs_in_group
                group_data, slices = torch_geometric.data.InMemoryDataset.collate(group_data_list)
            
                save_path = os.path.join(
                    self.processed_dir,
                    f'graph_data_group_{group_idx}_{ifiles}.pt')
                torch.save((group_data, slices), save_path)
                graphs_in_group = []
                ifiles += 1
            
    
    def preprocess(self, num_workers=4):
        """Processes raw data into PyTorch Geometric format."""
        
        jobs = []
        for ifile, file in enumerate(self.raw_paths):
            jobs.append((file, ifile))

        #p = Pool(num_workers)
        #p.map(self._preprocess, jobs)
        for job in jobs:
            self._preprocess(job)

    def _load_group(self, file_path):
        """Loads a group of graphs from a file and yields individual graphs."""
        full_path = os.path.join(self.root, "processed", file_path)
        group_data, slices = torch.load(full_path)
        
        num_graphs = len(slices['x']) - 1
        for i in range(num_graphs):
            data = Data()
            for key in group_data.keys():
                if key in slices:
                    start, end = slices[key][i], slices[key][i+1]
                    if key == 'edge_index':
                        data[key] = group_data[key][:, start:end]
                    else:
                        data[key] = group_data[key][start:end]
                else:
                    data[key] = group_data[key]
            if self.transform:
                data = self.transform(data)
            yield data


    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self._start
            iter_end = self._end
        else:  # in a worker process
             # split workload
            per_worker = int(math.ceil((self._end - self._start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self._start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self._end)
        for ifile in range(iter_start, iter_end):
            yield from self._load_group(self.processed_files[ifile])

