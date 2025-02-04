import awkward as ak
import numpy as np
import tensorflow as tf
from collections import namedtuple
import correctionlib
import os

tf_minor_version = int(tf.__version__.split(".")[1])

from glob import glob
from itertools import zip_longest, islice
from collections import deque
import multiprocessing as mp
import queue
import gc


default_features_dict = {
     "cl_features" : [ "en_cluster","et_cluster",
                      "en_cluster_log","et_cluster_log",
                      "cluster_eta", "cluster_phi", 
                      "cluster_ieta","cluster_iphi","cluster_iz",
                      "cluster_deta", "cluster_dphi",
                      "cluster_den_seed","cluster_det_seed",
                      "cluster_den_seed_log","cluster_det_seed_log",
                      "en_cluster_calib", "et_cluster_calib",
                      "en_cluster_calib_log", "et_cluster_calib_log",
                      "cl_f5_r9", "cl_f5_sigmaIetaIeta", "cl_f5_sigmaIetaIphi",
                      "cl_f5_sigmaIphiIphi","cl_f5_swissCross",
                      "cl_r9", "cl_sigmaIetaIeta", "cl_sigmaIetaIphi",
                      "cl_sigmaIphiIphi","cl_swissCross",
                      "cl_nxtals", "cl_etaWidth","cl_phiWidth"],


    "cl_metadata": [ "calo_score", "calo_simen_sig", "calo_simen_PU",
                     "cluster_PUfrac","calo_nxtals_PU",
                     "noise_en","noise_en_uncal","noise_en_nofrac","noise_en_uncal_nofrac" ],

    "cl_labels" : ["is_seed","is_calo_matched","is_calo_seed", "in_scluster","in_geom_mustache"],

    
    "seed_features" : ["seed_eta","seed_phi", "seed_ieta","seed_iphi", "seed_iz", 
                       "en_seed", "et_seed","en_seed_calib","et_seed_calib",
                       "en_seed_log", "et_seed_log", "en_seed_calib_log", "et_seed_calib_log", 
                    "seed_f5_r9","seed_f5_sigmaIetaIeta", "seed_f5_sigmaIetaIphi",
                    "seed_f5_sigmaIphiIphi","seed_f5_swissCross",
                    "seed_r9","seed_sigmaIetaIeta", "seed_sigmaIetaIphi",
                    "seed_sigmaIphiIphi","seed_swissCross",
                    "seed_nxtals","seed_etaWidth","seed_phiWidth"
                    ],

    "seed_metadata": [ "seed_score", "seed_simen_sig", "seed_simen_PU", "seed_PUfrac"],

    "window_features" : [ "max_en_cluster","max_et_cluster","max_deta_cluster","max_dphi_cluster","max_den_cluster","max_det_cluster",
                         "min_en_cluster","min_et_cluster","min_deta_cluster","min_dphi_cluster","min_den_cluster","min_det_cluster",
                         "mean_en_cluster","mean_et_cluster","mean_deta_cluster","mean_dphi_cluster","mean_den_cluster","mean_det_cluster" ],

    "window_metadata": ["ncls", "nclusters_insc",
                        "nVtx", "rho", "obsPU", "truePU",
                        "sim_true_eta", "sim_true_phi",  
                        "gen_true_eta","gen_true_phi",
                        "en_true_sim","et_true_sim", "en_true_gen", "et_true_gen",
                        "en_true_sim_good", "et_true_sim_good",
                        "en_mustache_raw", "et_mustache_raw","en_mustache_calib", "et_mustache_calib",
                        "max_en_cluster_insc","max_deta_cluster_insc","max_dphi_cluster_insc",
                        "event_tot_simen_PU","wtot_simen_PU","wtot_simen_sig",
                       "is_seed_calo_matched", "is_seed_calo_seed", "is_seed_mustache_matched"],    

}


##### Configuration namedtuple

from typing import List, Dict
from dataclasses import dataclass, field


@dataclass
class LoaderConfig():
    # list of input files [[fileA1, fileB1], [fileA2, fileB2]]
    # each list will be handled by worker in parallel.
    # Different files in each list will be interleaved and shuffled for each chunk.
    input_files : List[List[str]]  = field(default_factory=list)
    # in alternative a list of directories to be zipped together can be provided
    # The files from each folder will be zipped and samples shuffled together.    
    input_folders : List[str] = field(default_factory=list )
    # Group of records to read from awk files
    file_input_columns: List[str] = field(default_factory=lambda : ["cl_features", "cl_labels",
                                                                "window_features", "window_metadata", "cl_h"])
    # specific fields to read out for each cl, window, labels..
    columns: Dict[str,list] = field(default_factory=lambda: default_features_dict)
    additional_columns: Dict[str,list] = field(default_factory=lambda: {}) 
    padding: bool = True # zero padding or not
    include_rechits: bool = True # include the rechits in the preprocessing
    # if -1 it will be dynami# c for each batch,
    #if >0 it will be a fix number with clippingq
    ncls_padding: int = 45 
    nhits_padding: int = 45 # as ncls_padding
    # dimension of the chunk to read at once from each file,
    # must be a multiple of the batch_size                         
    chunk_size: int = 256*20
    batch_size: int = 256 # final batch size of the arrays
    maxevents: int = 2560  # maximum number of event to be read in total
    offset: int = 0     # Offset for reading records from each file
    # normalization strategy for cl_features and window_features,
    # stdscale, minmax,or None
    norm_type: str = "stdscale"
    norm_factors_file: str = "normalization_factors_v1.json"     #file with normalization factors
    reweighting_file: str = None # File with reweighting correctlib json
    norm_factors: dict = None     #normalization factors array dictionary
    nworkers: int = 2   # number of parallele process to use to read files
    max_batches_in_memory: int = 30 #  number of batches to load at max in memory
    '''
    The output of the pipeline is fed to tensorflow by providing 3 items: X,y,weight.
    Each item can contain a tuple of tensors. The `output_tensors` configuration
    defines the tensors for each item: e.g. [[cl_X, wind_X, is_seed],[cl_y], []].
    The available tensors are defined by the preprocessing function. 
    '''
    output_tensors : List[List[str]] = field(default_factory=lambda : [
        ["cl_X_norm", "wind_X_norm", "cl_hits", "is_seed", "cls_mask", "hits_mask"],["in_scluster", "flavour", "cl_X", "wind_X", "wind_meta", "is_seed_calo_seed"], ["weight"] ])


def get_tensors_spec(config):
    if tf_minor_version >=4:
        spec = {
        "cl_X": tf.TensorSpec(shape=(None,None,len(config.columns["cl_features"])), dtype=tf.float32), # cl_x (batch, ncls, #cl_x_features)
        "cl_X_norm": tf.TensorSpec(shape=(None,None,len(config.columns["cl_features"])), dtype=tf.float32),
        "wind_X" : tf.TensorSpec(shape=(None,len(config.columns["window_features"])), dtype=tf.float32),  #windox_X (batch, #wind_x)
        "wind_X_norm" : tf.TensorSpec(shape=(None,len(config.columns["window_features"])), dtype=tf.float32),  #windox_X (batch, #wind_x)
        "wind_meta": tf.TensorSpec(shape=(None,len(config.columns["window_metadata"])), dtype=tf.float32),  #windox_X (batch, #wind_x)
        "cl_hits" : tf.TensorSpec(shape=(None,None, None, 4), dtype=tf.float32), #hits  (batch, ncls, nhits, 4)
        "is_seed": tf.TensorSpec(shape=(None,None), dtype=tf.float32),  # is seed (batch, ncls,)
        "in_scluster": tf.TensorSpec(shape=(None,None), dtype=tf.int64),  # in_supercluster (batch, ncls,)
        "is_seed_calo_seed": tf.TensorSpec(shape=(None), dtype=tf.float32), #seed_is_caloseed (batch, #wind_x)
        "cl_Y": tf.TensorSpec(shape=(None,None, len(config.columns["cl_labels"])), dtype=tf.bool),  #cl_y (batch, ncls, #cl_labels)
        "flavour" : tf.TensorSpec(shape=(None), dtype=tf.float32),  #windox_X (batch, #wind_x)
        "weight": tf.TensorSpec(shape=(None), dtype=tf.float32), # weights
        "cls_mask": tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        "hits_mask": tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
        
        }
        return tuple([ tuple([spec[label]  for label in conf ])    for conf in config.output_tensors])
    else:
        types = {
            "cl_X": tf.float32,
            "cl_X_norm": tf.float32,
            "wind_X": tf.float32,
            "wind_X_norm": tf.float32,
            "wind_meta": tf.float32,
            "cl_hits": tf.float32,
            "is_seed": tf.float32,
            "in_scluster": tf.int64,
            "cl_y": tf.bool,
            "is_seed_calo_seed": tf.float32,
            "flavour": tf.float32,
            "weight": tf.float32,
            "cls_mask": tf.float32,
            "hits_mask": tf.float32
        }
        return tuple([ tuple([types[label]  for label in conf ])    for conf in config.output_tensors]) 


def get_output_indices(output_tensors):
    '''
    This function translates a list of labels for the tensors outputs to a list of
    indices. In fact the generator internally returns all the tensors in a tuple, for performance
    optiomization. The output formatting is then handled by the main thread.
    The order of the output between this list and the preprocessing function MUST match!
    '''
    indices = [ "cl_X", "cl_X_norm", "cl_Y",
                "is_seed",  "in_scluster", "cl_hits",
                "wind_X","wind_X_norm", "wind_meta",
                "flavour", "hits_mask", "cls_mask", "is_seed_calo_seed", "weight"]    
    return tuple([ tuple([indices.index(label) for label in cfg  ]) for cfg in output_tensors])
        
########################################################
### Utility functions to build the generator chain   ###
########################################################

def load_dataset_chunks(df, config, chunk_size, offset=0, maxevents=None):
    # Filtering the columns to keey only the requested ones
    cols = { key: df[key][v] for key, v in config.columns.items() }
    # Adding additional columns (not used for training)
    cols.update( { "meta_"+key: df[key][v] for key,v in config.additional_columns.items()})
    # Adding the clusters hits 
    cols['cl_h'] = df.cl_h
    filtered_df = ak.zip(cols, depth_limit=1)
    # Now load in large chunks batching
    if maxevents:
        nchunks = maxevents // chunk_size
    else:
        nchunks = ak.num(filtered_df.cl_features, axis=0)//chunk_size 
    for i in range(nchunks):
        # Then materialize it
        #pid = mp.current_process().pid
        #print(f"--> worker {pid}) Loading chunk {i} from {offset} with size {chunk_size}")
        yield chunk_size, ak.materialized(filtered_df[offset + i*chunk_size: offset + (i+1)*chunk_size])
        #yield batch_size, df[i*batch_size: (i+1)*batch_size]


        
def split_batches(gen, batch_size):
    for size, df in gen:
        if size % batch_size == 0:
            for i in range(size//batch_size):
                if isinstance(df, tuple):
                    yield batch_size, tuple(d[i*batch_size : (i+1)*batch_size] for d in df)
                elif isinstance(df, dict):
                    yield batch_size, { k: d[i*batch_size: (i+1)*batch_size] for k, d in df.items() } 
                else:
                    yield batch_size, df[i*batch_size : (i+1)*batch_size]
        else:
            raise Exception("Please specifie a batchsize compatible with the loaded chunks size")
        
def buffer(gen,size):
    ''' This generator buffer a number `size` of elements from an iterator and yields them. 
    When the buffer is empty the quee is filled again'''
    q = deque()
    while True:
        # Caching in the the queue some number of elements
        in_q = 0
        try:
            for _ in range(size):
                q.append(next(gen))
                in_q +=1
        except StopIteration:
            for _ in range(in_q):
                yield q.popleft()
            break
        # Now serve them
        for _ in range(in_q):
            yield q.popleft()
        
def shuffle_fn(size, df):
    try:
        perm_i = np.random.permutation(size)
        return size, df[perm_i]
    except:
        return 0, ak.Array([])
    
    
def shuffle_dataset(gen, n_batches=None):
    #if n_batches==None: 
    # permute the single batch
    for i, (size, df) in enumerate(gen):
        yield shuffle_fn(size, df)
    # else:
    #     for dflist in cache_generator(gen, n_batches):
    #         size = dflist[0][0] 
    #         perm_i = np.random.permutation(size*len(dflist))
    #         totdf = ak.concatenate([df[1] for df in dflist])[perm_i]
    #         for i in range(n_batches):
    #             yield size, totdf[i*size: (i+1)*size]
                
def zip_datasets(*iterables):
    yield from zip_longest(*iterables, fillvalue=(0, ak.Array([])))
    
def concat_fn(dfs):
    return sum([d[0] for d in dfs]), ak.concatenate([d[1] for d in dfs])

def concat_datasets(*iterables):
    for dfs in zip_datasets(*iterables):
        yield concat_fn(dfs)

def build_output_structure(gen, structure):
    for size, df in gen:
        yield size, tuple([
            tuple([df[label] for label in cfg])
            for cfg in structure
        ])
        
def to_flat_numpy(X, axis=2, allow_missing=True):
    return np.stack([ak.to_numpy(X[f], allow_missing=allow_missing) for f in X.fields], axis=axis)

def convert_to_tf(df):
    return [tf.convert_to_tensor(d) for d in df ]

##############################################################################################
# Multiprocessor generator running a separate process for each group of
# input files. The result of each process is put in a queue and consumed by the main thread.

def multiprocessor_generator_from_files(files, internal_generator, output_queue_size=40, nworkers=4, maxevents=None):
    '''
    Generator with multiprocessing working on a list of input files.
    All the input files are put in a Queue that is consumed by a Pool of workers. 
    Each worker passes the file to the `internal_generator` and consumes it. 
    The output is put in an output Queue which is consumed by the main thread.
    Doing so the processing is in parallel. 
    '''
    def process(input_q, output_q, mess_q):
        # Change the random seed for each processor
        #output_q.cancel_join_thread()
        pid = mp.current_process().pid
        np.random.seed()
        working = True
        while working:
            # Getting a file
            file = input_q.get()
            #print(f"--> worker {pid}) Processing file: ", file)
            if file is None:
                output_q.put(None)
                break
            # We give the file to the generator and then yield from it
            for out in internal_generator(file):
                if not mess_q.empty():
                    #print(f"--> worker {pid}) Received message: ", mess_q.get())
                    working = False
                    output_q.put(None, block=True)#, timeout=5)
                    break
                
                #print(f"--> worker {pid}) wants to yield > events: ", out[0])
                try:
                    output_q.put(out, block=True) #,  block=True, timeout=5)
                    #print(f"--> worker {pid}) Yielded > events: ", out[0])
                except queue.Full:
                    pass
                    #print(f"--> worker {pid}) Queue is full")

            if not mess_q.empty() and working:
                output_q.put(None, block=True)#, timeout=5)
                #print(f"--> worker {pid}) Received message: ", mess_q.get())
                break

            #print("I'm exited from the internal_generator loop")
        output_q.close()
        #print(f"--> worker {pid}) Closing worker")

    
    input_q = mp.SimpleQueue()
    # Load all the files in the input file
    for file in files: 
        input_q.put(file)
    #Once generator is consumed, send end-signal
    for i in range(nworkers):
        input_q.put(None)
    
    output_q = mp.Queue(maxsize=output_queue_size)
    #output_q.cancel_join_thread()
    mess_q = mp.SimpleQueue()
    #output_q = mp.SimpleQueue()
    # Here we need 2 groups of worker :
    # * One that do the main processing. It will be `pool`.
    # * One that read the results and yield it back, to keep it as a generator. The main thread will do it.
    pool = mp.Pool(nworkers, initializer=process, initargs=(input_q, output_q, mess_q))
    
    try : 
        finished_workers = 0
        tot_events = 0
        while True:
            it = output_q.get(block=True)
            if it is None:
                finished_workers += 1
                if finished_workers == nworkers:
                    break
            else:
                size, df = it
                tot_events += size
                if maxevents and tot_events > maxevents:
                    #print("Max events reached")
                    # refilling the input files
                    # for file in files: 
                    #     input_q.put(file)
                    break

                else:
                    #pid = mp.current_process().pid
                    #print(f"external generator yielding > tot events: ", tot_events)
                    yield it
                    # explicit delete of the data
                    for d in df:
                        del d
                    del df
                    del it

    # except GeneratorExit:
    #     # Put None in input file in the case that workers are still waiting for inptu
    #     for i in range(nworkers):
    #         input_q.put(None)
    
    finally:
        # This is called at GeneratorExit
        #print("Multiprocessing generator closing...")
        pool.close()
        # We need to grafeully close the pool
        # by sending a message to the queue to terminate and stop
        # pushing data
        for i in range(nworkers):
            mess_q.put("terminate")
        # There may be stuff left in the queue
        # We need a better way
        # Cleaning the queue
        # doing that, also if the workers are blocked waiting to push data
        # they are released and they read the terminate message and exit gracefully

        n_pool_close = 0
        while n_pool_close < nworkers:
            #print("Flushing the data queue...")
            out = output_q.get(block=True)
            if out is None:
                #print("Worker closed")
                n_pool_close += 1
            else:
                size, df = out
                for d in df:
                    del d
                del df            
        #print("Data queue flushed")

        #print("Closing the queues...")
        output_q.close()
        output_q.join_thread()
        input_q.close()
        mess_q.close()
        
        #print("Waiting for the pool worker to join...")
        pool.join()
        pool.terminate()
        del input_q
        del mess_q
        del output_q
        del pool
        gc.collect()
        #print("Multiprocessing generator closed") 
            

###############################################################

 

def load_batches_from_files_generator(config, preprocessing_fn, shuffle=True, return_original=False):
    '''
    Generator reading full batches from a list of files.
    The process is the following:
    - a chunk is read from each file in the list
    - chunks get concatenated
    - samples are shuffled
    - a preprocessing function is applied on the shuffled samples
    - the chunk is split in batches and returned as a generator.
    
    A config file is needed to specify which columns are read from the files,
    padding, and the size of chunks and batched.

    N.B.: the chunk size must be a multiple of the batch size. 
    '''
    # Prepare the preprocessing function with the config
    _preprocess_fn = preprocessing_fn(config)
    
    def _fn(files): 
        # Parquet files
        dfs_raw = [ ak.from_parquet(file, lazy=True, use_threads=True, columns=config.file_input_columns) for file in files if file!=None]
        # Loading chunks from the files
        initial_dfs = [ load_dataset_chunks(df, config, chunk_size=config.chunk_size, offset=config.offset) for df in dfs_raw]
        # Contatenate the chunks from the list of files
        df = concat_datasets(*initial_dfs)
        # Shuffle the axis=0
        if shuffle:
            df = shuffle_dataset(df)
        # Processing the data to extract X,Y, etc
        def preproc(df):
            for data in df: #consuming the generator
                processed = _preprocess_fn(data)
                if return_original:
                    # Create a "fake" generato to add the chunk
                    yield processed[0], (*processed[1], data[1]) # data[1] if the original chunk in awk
                else:
                    yield processed
        # Split in batches
        try:
            yield from split_batches(preproc(df), config.batch_size)
        except GeneratorExit:
            print("Internal generator closed")
            for d in dfs_raw:
                del d
            for d in initial_dfs:
                del d
            del df
        except Exception as e:
            print("Error in the internal generator: ", e)
            for d in dfs_raw:
                del d
            for d in initial_dfs:
                del d
            del df
        finally:
            #print("THE END of the internal generator")
            return
            
    return _fn


###########################################################################################################
# Preprocessing function to prepare numpy data for training
def preprocessing(config):
    '''
    Preprocessing function preparing the data to be in the format needed for training.
     Several zero-padded numpy arrays are returned:
     - Cluster features (batchsize, Nclusters, Nfeatures)
     - Cluster labels (batchsize, Nclusters, Nlabels)
     - is_seed mask (batchsize, Ncluster)
     - Rechits (batchsize, Nclusters, Nrechits, Nrechits_features)
     - Window features (batchsize, Nwind_features)
     - Window metadata (batchsize, Nwind_meatadata)
     - flavour (ele/gamma/jets) (batchsize,)
     - Rechits padding mask  (batchsize, Ncluster, Nrechits)
     - Clusters padding mask (batchsize, Ncluster)
     - is_seed_calo_seed mask (batchsize, Ncluster)
     - Reweighting weight

    The config for the function contains all the info and have the format
     The zero-padding can be fixed side (specified in the config dizionary),
     or computed dinamically for each chunk.

    The order of the output MUST BE SYNCHRONIZED with the get_output_indices function
    '''
    if config.reweighting_file != None:
        cset = correctionlib.CorrectionSet.from_file(config.reweighting_file)
        corr = cset.compound["total_reweighting"]
    else:
        corr = None
    
    def process_fn(data): 
        size, df = data
        # Extraction of the ntuples and zero padding

        #padding
        if config.padding:
            if config.ncls_padding == -1:
                # dynamic padding
                max_ncls = ak.max(ak.num(df.cl_features, axis=1))
            else:
                max_ncls = config.ncls_padding
            if config.nhits_padding == -1 and config.include_rechits:
                max_nhits = ak.max(ak.num(df.cl_h, axis=2))
            else:
                max_nhits = config.nhits_padding

            # Computing the weight
            flavour = np.asarray(df.window_metadata.flavour)
            weight = np.ones((size,), dtype=float)
    
            if corr!= None:
                seed_df = df.cl_features[df.cl_labels.is_seed==1][["cluster_eta","et_cluster"]]
                seed_eta = ak.to_numpy(abs(ak.flatten(seed_df.cluster_eta)))
                seed_et = ak.to_numpy(ak.flatten(seed_df.et_cluster))
                ncls_tot = ak.to_numpy(df.window_metadata.ncls)
                # Different weight for electrons and photons
                index = np.indices([size]).flatten()
                mask_ele = flavour == 11
                mask_pho = flavour == 22
                weight[index[mask_ele]] = corr.evaluate(11, seed_eta[mask_ele], seed_et[mask_ele], ncls_tot[mask_ele])
                weight[index[mask_pho]] = corr.evaluate(22, seed_eta[mask_pho], seed_et[mask_pho], ncls_tot[mask_pho])
                
            # Padding 
            cls_X_pad = ak.pad_none(df.cl_features, max_ncls, clip=True)
            cls_Y_pad = ak.pad_none(df.cl_labels, max_ncls, clip=True)
            # Fill padding with empty record
            cls_X_pad = ak.fill_none(cls_X_pad, {k:0. for k in df.cl_features.fields})
            cls_Y_pad = ak.fill_none(cls_Y_pad, {k:0 for k in df.cl_labels.fields})
            
            wind_X = df.window_features
            wind_meta = df.window_metadata
            is_seed_pad = ak.fill_none(ak.pad_none(df.cl_labels["is_seed"], max_ncls, clip=True),0)
            in_scluster_pad = ak.fill_none(ak.pad_none(df.cl_labels["in_scluster"], max_ncls, clip=True),0)

            # Converting to numpy after padding
            cls_X_pad_np = to_flat_numpy(cls_X_pad, axis=2, allow_missing=False)
            cls_Y_pad_np = to_flat_numpy(cls_Y_pad, axis=2, allow_missing=False)
            is_seed_pad_np = ak.to_numpy(is_seed_pad, allow_missing=False)
            in_scluster_pad_np = ak.to_numpy(in_scluster_pad, allow_missing=False)
            wind_X_np = to_flat_numpy(wind_X, axis=1)
            wind_meta_np = to_flat_numpy(wind_meta, axis=1)

            # hits padding
            if config.include_rechits:
                cl_hits_padrec = ak.pad_none(df.cl_h, max_nhits, axis=2, clip=True) # --> pad rechits dim
                cl_hits_padded = ak.pad_none(cl_hits_padrec, max_ncls, axis=1, clip=True) # --> pad ncls dimension
                # fill none with array of correct dimension
                cl_hits_padded = ak.fill_none(cl_hits_padded, np.zeros(4), axis=2) # --> rechit level
                cl_hits_padded = ak.fill_none(cl_hits_padded, np.zeros((max_nhits, 4)), axis=1)
                # Only hits have truly padded None to be converted to masked numpy arrays
                cl_hits_pad_np = ak.to_numpy(cl_hits_padded, allow_missing=False)
                hits_mask = np.array(np.sum(cl_hits_padded, axis=-1) != 0, dtype=float)
            else:
                cl_hits_pad_np = np.zeros((size))
                hits_mask = np.zeros((size))
            
            # Masks for padding
            cls_mask = ak.to_numpy(cls_X_pad.en_cluster != 0).astype(float)
            # cls_mask = np.any(hits_mask, axis=-1).astype(int)
            #not adding the last dim for broadcasting to give the user more flexibility

            # Normalization
            norm_fact = config.norm_factors
            if config.norm_type == "stdscale":
                # With remasking
                cls_X_pad_norm = ((cls_X_pad_np - norm_fact["cluster"]["mean"])/ norm_fact["cluster"]["std"] ) * cls_mask[:,:,None]
                wind_X_norm =  ((wind_X_np - norm_fact["window"]["mean"])/ norm_fact["window"]["std"] )
                #Window features are always scaled with min max
                # wind_X_norm =  ((wind_X_np - norm_fact["window"]["min"])/ (norm_fact["window"]["max"]-norm_fact["window"]["min"]) ) 
            elif config.norm_type == "minmax":
                cls_X_pad_norm = ((cls_X_pad_np - norm_fact["cluster"]["min"])/ (norm_fact["cluster"]["max"]-norm_fact["cluster"]["min"])) \
                    * cls_mask[:,:,:None]
                wind_X_norm =  ((wind_X_np - norm_fact["window"]["min"])/ (norm_fact["window"]["max"]-norm_fact["window"]["min"]) ) 
            else:
                cls_X_pad_norm = cls_X_pad_np
                windo_X_norm = wind_X_norm

            is_seed_calo_seed = df.window_metadata["is_seed_calo_seed"]

            return size, (cls_X_pad_np, cls_X_pad_norm, cls_Y_pad_np, is_seed_pad_np,
                          in_scluster_pad_np, cl_hits_pad_np, wind_X_np, wind_X_norm,
                          wind_meta_np, flavour, hits_mask, cls_mask, is_seed_calo_seed, weight)
        else:
            # No padding --> never used
            raise Exception("Not implemented!")
            cls_X = df.cl_features, max_ncls
            cls_Y = df.cl_labels["in_scluster"], max_ncls
            is_seed = df.cl_labels["is_seed"], max_ncls
            cl_hits = df.cl_h
            return size, (cls_X, cls_Y, is_seed, cl_hits, flavour)
            
    return process_fn


###########################################################################################################
# Function reading from file the normalization factors 

def get_norm_factors(norm_file, cl_features, wind_features, numpy=True):
    # Loading the factors from file
    norm_factors = ak.from_json(norm_file)
    if numpy:
        return {
            "cluster" : {
                "mean": to_flat_numpy(norm_factors["cluster"]["mean"][cl_features], axis=0),
                "std": to_flat_numpy(norm_factors["cluster"]["std"][cl_features], axis=0),
                "min": to_flat_numpy(norm_factors["cluster"]["min"][cl_features], axis=0),
                "max": to_flat_numpy(norm_factors["cluster"]["max"][cl_features], axis=0)
            },
            "window":{
                "mean": to_flat_numpy(norm_factors["window"]["mean"][wind_features], axis=0),
                "std": to_flat_numpy(norm_factors["window"]["std"][wind_features], axis=0),
                "min": to_flat_numpy(norm_factors["window"]["min"][wind_features], axis=0),
                "max": to_flat_numpy(norm_factors["window"]["max"][wind_features], axis=0)
            }
        }
    else:
        return {
            "cluster" : {
                "mean": norm_factors["cluster"]["mean"][cl_features],
                "std": norm_factors["cluster"]["std"][cl_features],
                "min": norm_factors["cluster"]["min"][cl_features],
                "max": norm_factors["cluster"]["max"][cl_features],
            },
            "window":{
                "mean": norm_factors["window"]["mean"][wind_features],
                "std": norm_factors["window"]["std"][wind_features], 
                "min": norm_factors["window"]["min"][wind_features], 
                "max": norm_factors["window"]["max"][wind_features], 
            }
        }


#####################################################################################################
### Tensorflow tensors conversion

def numpy_generator(config):
    '''
    The function converts the multipurpose generator chain to a
    Tensorflow format chain.
    The list of output_tensors is read from the general output and put in the
    requested order.
    '''
    # Getting the output configuration from the config
    out_index = get_output_indices(config.output_tensors)
    # Generator function
    def _gen():
        file_loader_generator = load_batches_from_files_generator(config, preprocessing)
        multidataset = multiprocessor_generator_from_files(config.input_files, 
                                                           file_loader_generator, 
                                                           output_queue_size=config.max_batches_in_memory, 
                                                           nworkers=config.nworkers, 
                                                           maxevents=config.maxevents)

        for size, df in multidataset:
            # Now the output is formatted with the order requested in the config
            yield tuple([ tuple([df[i] for i in o]) for o in out_index])
    return _gen


def tf_generator(config):
    '''
    The function converts the multipurpose generator chain to a
    Tensorflow format chain.
    The list of output_tensors is read from the general output and put in the
    requested order.
    '''
    # Getting the output configuration from the config
    out_index = get_output_indices(config.output_tensors)
    # Generator function
    def _gen():
        #print("!!! STARTING NEW GENERATOR")
        file_loader_generator = load_batches_from_files_generator(config, preprocessing)
        multidataset = multiprocessor_generator_from_files(config.input_files, 
                                                           file_loader_generator, 
                                                           output_queue_size=config.max_batches_in_memory, 
                                                           nworkers=config.nworkers, 
                                                           maxevents=config.maxevents)
        try:
            for size, df in multidataset:
                #df_tf = convert_to_tf(df)
                # Now the output is formatted with the order requested in the config
                #out = tuple([ tuple([df_tf[i] for i in o]) for o in out_index])

                out = tuple([ tuple([df[i] for i in o]) for o in out_index])
                yield out
                # deleting all the stuff
                for d in out:
                    for j in d:
                        del j
                    del d
                del out
                # for d in df_tf:
                #     del d
                # del df_tf
                for d in df:
                    del d
                del df
                
                   
        except GeneratorExit:
            pass
            #print("GeneratorExit of the TF generator")
        finally:
            #print("End of the TF generator")
            gc.collect()
            #print("GC of TF generator internal")
    return _gen




################################
# User API to get a dataset general

def load_dataset (config: LoaderConfig, output_type="tf"):
    '''
    Function exposing to the end user the tensorflow dataset loading through the awkward chain. 
    '''
    # Check if folders instead of files have been provided
    if config.input_folders:
        for folder in config.input_folders:
            if not os.path.exists(folder):
                raise Exception(f"Folder {folder} does not exists! Check your configuration file")
        config.input_files = list(zip_longest(*[glob(folder+"/*.parquet") for folder in config.input_folders]))
    if not config.input_folders and not config.input_files:
        raise Exception("No input folders or files provided! Please provide some input!")
    # Load the normalization factors
    if config.norm_factors == None and config.norm_factors_file:
        config.norm_factors = get_norm_factors(config.norm_factors_file, config.columns["cl_features"], config.columns["window_features"])

    if output_type == "tf":
        if tf_minor_version >=4:    
            df = tf.data.Dataset.from_generator(tf_generator(config), 
                                                output_signature= get_tensors_spec(config))

        else:
            df = tf.data.Dataset.from_generator(tf_generator(config), 
                                                output_types= get_tensors_spec(config))
        return df

    elif output_type == "numpy":
        return numpy_generator(config)

    
# def load_dataset2 (config: LoaderConfig, output_type="tf"):
#     '''
#     Function exposing to the end user the tensorflow dataset loading through the awkward chain. 
#     '''
#     # Check if folders instead of files have been provided
#     if config.input_folders:
#         for folder in config.input_folders:
#             if not os.path.exists(folder):
#                 raise Exception(f"Folder {folder} does not exists! Check your configuration file")
#         config.input_files = list(zip_longest(*[glob(folder+"/*.parquet") for folder in config.input_folders]))
#     if not config.input_folders and not config.input_files:
#         raise Exception("No input folders or files provided! Please provide some input!")
#     # Load the normalization factors
#     if config.norm_factors == None and config.norm_factors_file:
#         config.norm_factors = get_norm_factors(config.norm_factors_file, config.columns["cl_features"], config.columns["window_features"])


#     iterator = iter(tf_generator(config)())
#     def get():
#         try:
#             while True:
#                 df = next(iterator)
#                 print("yielding from fake generator")
#                 yield df
#         except GeneratorExit:
#             print("Fake Generator closed")
         
#     df = tf.data.Dataset.from_generator(get,
#                                         output_signature= get_tensors_spec(config))
#     return df


#####################################
## Functions for performance evaluation

def load_tfdataset_and_original(config:LoaderConfig):
    ''' This function gets an awkward array input and format the data
    in the tf output format defined by the configuration.
    The function does not handle shuffling or batching.
    It returns the data in the tf format as required in the config.
    '''
    if config.input_folders:
        for folder in config.input_folders:
            if not os.path.exists(folder):
                raise Exception(f"Folder {folder} does not exists! Check your configuration file")
        config.input_files = list(zip_longest(*[glob(folder+"/*.parquet") for folder in config.input_folders]))
    if not config.input_folders and not config.input_files:
        raise Exception("No input folders or files provided! Please provide some input!")
    # Load the normalization factors
    if config.norm_factors == None and config.norm_factors_file:
        config.norm_factors = get_norm_factors(config.norm_factors_file, config.columns["cl_features"], config.columns["window_features"])

    tot_samples = 0
    file_loader_generator = load_batches_from_files_generator(config, preprocessing,
                                                              shuffle=False, return_original=True)
    out_index = get_output_indices(config.output_tensors)
    for files in config.input_files:
        for size, df in file_loader_generator(files):
            tot_samples += size
            if tot_samples >= config.maxevents:
                break
            original = df[-1]
            df_tf = convert_to_tf(df[:-1]) #exclude the last entry which is the original
            yield tuple([ tuple([df_tf[i] for i in o]) for o in out_index]), original
        

#Utils for debugging tf dataset

def get(dataset):
    el = next(iter(dataset.take(1)))
    return el
