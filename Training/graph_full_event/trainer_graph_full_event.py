import comet_ml
from comet_ml.integration.pytorch import watch
import argparse
import json
import os
import importlib.util
from dataset_loader import ECALGraphDataset, RoundRobinDataset
from torch_geometric.loader import DataLoader
import numpy as np
#from models_archive_graph_full_event.GCN import GCN
#from loss_functions_graph_full_event.loss_functions import FocalLoss #Currently not needed, I think better to put loss in model file
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch


parser = argparse.ArgumentParser()

parser.add_argument("--config", type=str, help="Config", required=True)
parser.add_argument("--model", type=str, help="Model .py", required=True)
parser.add_argument("--name", type=str, help="Model version name", required=True)
parser.add_argument("--apikey", type=str, help="comet API key", required=True)
parser.add_argument("--output", type=str,help="Override output folder", required=False)
#parser.add_argument("--debug", action="store_true", help="Debug and run TF eagerly")
#parser.add_argument("-v","--verbose", action="store_true", help="Verbose")
args = parser.parse_args()


def get_et_weights(batch, et_bins, noise_bin_weight, seed_bin_weight, matched_not_seed_bin_weight):
    "get et bin weights from config file and assign weight to each cluster based on its et and class: caloseed, calomatched not seed, not calomatched"
    
    et_batch = batch.x[:,1] # all et of full batch
    is_calo_matched = batch.y[:,0] # 0 if not matched, 1 if matched
    is_calo_seed = batch.y[:,1] # 0 if not seed, 1 if seed

    et_weights = torch.ones(len(et_batch)) # tensor to store weights
    bin_indices = np.digitize(et_batch, et_bins) - 1 # for each cluster we determine in which energy bin it is. bins from reweighting file

    # calculate the weights for caloseed clusters
    indices = torch.nonzero(is_calo_seed, as_tuple=True)[0] # get indices
    et_weights[indices] = torch.tensor(seed_bin_weight[bin_indices[indices]], dtype=et_weights.dtype) # set new weights for calo seed clusters

    # calculate the weights of calomatched AND not seed clusters
    indices = torch.nonzero((is_calo_matched == 1) & (is_calo_seed == 0), as_tuple=True)[0] # get indices
    et_weights[indices] = torch.tensor(matched_not_seed_bin_weight[bin_indices[indices]], dtype=et_weights.dtype)

    # calculate the weights of noise clusters
    indices = torch.nonzero(is_calo_matched == 0, as_tuple=True)[0] # get indices
    et_weights[indices] = torch.tensor(noise_bin_weight[bin_indices[indices]], dtype=et_weights.dtype)

    return et_weights



config = json.load(open(args.config)) # load the config file

# create output directory for model weights and config
# from global_training/trainer_awk.py
def get_unique_run():
    previous_runs = list(filter( lambda k: k.startswith("run"), os.listdir(config["models_path"])))
    run_number = 1
    if len(previous_runs) > 0:
        run_number = max([int(s.split('_')[1]) for s in previous_runs if s.startswith("run")]) + 1 
    return run_number


if args.output != None:
    config["models_path"] = args.output
if not os.path.isdir(config["models_path"]):
    os.makedirs(config["models_path"])

name =  f'run_{get_unique_run():02}'
if args.name != None:
    name += f"_{args.name}"

outdir = config["models_path"] + "/"+ name
config["model_name"] = name 

if os.path.isdir(outdir):
    print("Output directory exists: {}".format(outdir), file=sys.stderr)
else:
    os.makedirs(outdir)

print("Model output folder: ", outdir)

# Save the output folder in the config
config["models_path"] = outdir
config["model_definition_path"] = os.path.join(outdir, os.path.split(args.model)[-1])
#Copying the config file and model file in the output dir:
os.system("cp {} {}".format(args.model, outdir))
json.dump(config, open(os.path.join(outdir, "training_config.json"),"w"),
          indent=2)


# load data
print(">>> Load Dataset")


ds_train_single_ele = ECALGraphDataset(config["dataset_conf"]["training"]["input_folders"][0], "training")
ds_train_double_ele = ECALGraphDataset(config["dataset_conf"]["training"]["input_folders"][1], "training")
ds_train_single_gamma = ECALGraphDataset(config["dataset_conf"]["training"]["input_folders"][2], "training")
ds_train_double_gamma = ECALGraphDataset(config["dataset_conf"]["training"]["input_folders"][3], "training")

ds_test_single_ele = ECALGraphDataset(config["dataset_conf"]["training"]["input_folders"][0], "test")
ds_test_double_ele = ECALGraphDataset(config["dataset_conf"]["training"]["input_folders"][1], "test")
ds_test_single_gamma = ECALGraphDataset(config["dataset_conf"]["training"]["input_folders"][2], "test")
ds_test_double_gamma = ECALGraphDataset(config["dataset_conf"]["training"]["input_folders"][3], "test")

ds_train = RoundRobinDataset([ds_train_single_ele, ds_train_double_ele, ds_train_single_gamma, ds_train_double_gamma])
ds_test = RoundRobinDataset([ds_test_single_ele, ds_test_double_ele, ds_test_single_gamma, ds_test_double_gamma])

train_loader = DataLoader(
            ds_train,
            batch_size=config["dataset_conf"]["training"]["batch_size"], 
            num_workers=config["dataset_conf"]["training"]["nworkers"],
            pin_memory=True
)


# create and initialize model
print(">>> Create Model")

# dynamically load the wanted model, more flexibility
spec = importlib.util.spec_from_file_location("model", args.model)
model_lib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_lib)

for batch in train_loader:

    model = model_lib.GAT(config["nnodes"])
    out = model(batch.x, batch.edge_index) #initialize the model

    print(model)
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {param_name} | Parameters: {param.numel()}")
    
    break


# logging to comet

experiment = comet_ml.start(api_key=args.apikey,
                            project_name=config["comet"]["project_name"],
                            workspace=config["comet"]["workspace_name"])
experiment.set_name(name) #later get this from input
watch(model)

# optimizer selection

if config["opt"] == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
elif config["opt"] == "adamW":
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])


# loss selection
#criterion = model_lib.Weighted_BCE(weight=config["Weighted_BCE"])
#criterion = model_lib.FocalLoss(alpha=config["FocalLoss"]["alpha"], gamma=config["FocalLoss"]["gamma"])  # adjust alpha for class imbalance, gamma to penalize far off predictions
#criterion = model_lib.FocalLoss(alpha=0.5, gamma=2.0)
if config["loss"] == "FocalLoss":
    criterion = model_lib.FocalLoss(alpha=config["FocalLoss"]["alpha"], gamma=config["FocalLoss"]["gamma"])

if config["loss"] == "BCE_et_weighted":
    # load weights
    et_weights = json.load(open(config["loss weights"]))

    # store weights
    et_bins = np.array(et_weights["et_bins"])
    seed_bin_weight = np.array(et_weights["seed_bin_weight"])
    matched_not_seed_bin_weight = np.array(et_weights["matched_not_seed_bin_weight"])
    noise_bin_weight = np.array(et_weights["noise_bin_weight"])

    # initialize loss
    criterion = model_lib.BCE_et_weighted()

# start training

batch_step = 0 # for logging the batch metrics on comet
epoch_step = 0 # for logging the epoch metrics on comet
nbatches = 0 # for averaging in epoch loss calculation

for epoch in range(config["nepochs"]):
    
    model.train()  # set model to training mode
    
    # set up data stream for an epoch
    train_loader = DataLoader(
        ds_train,
        batch_size=config["dataset_conf"]["training"]["batch_size"],
        num_workers=config["dataset_conf"]["training"]["nworkers"],
        pin_memory=True  # faster data transfer to GPU
    )
    
    test_loader = DataLoader(
        ds_test,
        batch_size=config["dataset_conf"]["test"]["batch_size"],
        num_workers=config["dataset_conf"]["test"]["nworkers"],
        pin_memory=True  # faster data transfer to GPU
    )

    # for epoch metrics calculations
    nbatches = 0
    aggregated_loss = 0
    aggregated_tp = 0
    aggregated_tn = 0
    aggregated_fp = 0
    aggregated_fn = 0
    
    print("Epoch: ", epoch)
    # Loop over batches
    for batch_id, batch in enumerate(train_loader):

        nbatches += 1

        # Forward pass
        optimizer.zero_grad()  # Zero the gradients before the forward pass
        out = model(batch.x, batch.edge_index)  # Model predictions
        
        # Compute the loss
        #loss = criterion(out.squeeze(), batch.y[:,1])  # Squeeze to match shape
        #loss = criterion(out.squeeze(), batch.y[:,1], batch.y[:,0]) #Weighted_BCE
        if config["loss"] == "FocalLoss":
            loss = criterion(out.squeeze(), batch.y[:,1])  # Squeeze to match shape

        if config["loss"] == "BCE_et_weighted":
            # weights for every cluster in the batch
            cluster_weights = get_et_weights(batch, et_bins, noise_bin_weight, seed_bin_weight, matched_not_seed_bin_weight)
            loss = criterion(out.squeeze(), batch.y[:,1], cluster_weights)
        
        # Backward pass (compute gradients)
        loss.backward()

        # Update weights
        optimizer.step()
    
        # comet batch callbacks: loss and metrics.
        with torch.no_grad():

            # loss
            out = model(batch.x, batch.edge_index)
            #loss = criterion(out.squeeze(), batch.y[:,1]).numpy() # for FocalLoss
            #loss = criterion(out.squeeze(), batch.y[:,1], batch.y[:,0]).numpy() # Weighted BCE
            if config["loss"] == "FocalLoss":
                loss = criterion(out.squeeze(), batch.y[:,1]).numpy()
            
            if config["loss"] == "BCE_et_weighted":
                # weights for every cluster in the batch
                cluster_weights = get_et_weights(batch, et_bins, noise_bin_weight, seed_bin_weight, matched_not_seed_bin_weight)
                loss = criterion(out.squeeze(), batch.y[:,1], cluster_weights).numpy()

            aggregated_loss += loss

            # metrics
            labels = batch.y[:,1]
            predictions = (torch.sigmoid(out.squeeze()) > 0.5).float()

            aggregated_tp += ((labels == 1) & (predictions == 1)).sum().item()
            aggregated_tn += ((labels == 0) & (predictions == 0)).sum().item()
            aggregated_fp += ((labels == 0) & (predictions == 1)).sum().item()
            aggregated_fn += ((labels == 1) & (predictions == 0)).sum().item()
            
            # correctly labeled/all
            #accuracy = (predictions == labels).sum().item()/len(labels)
            accuracy = accuracy_score(labels, predictions, normalize=True)
            experiment.log_metric("batch accuracy", accuracy, step=batch_step)

            # correctly classified caloseed/all classified as caloseed = prob. that something labeled as caloseed is a caloseed
            # TP/TP + FP
            precision = precision_score(labels, predictions)
            experiment.log_metric("batch precision", precision, step=batch_step)

            # correctly classified caloseed/all actual caloseed = prob. to correctly label a caloseed as caloseed
            # TP/TP + FN
            recall = recall_score(labels, predictions)
            experiment.log_metric("batch recall", recall, step=batch_step)

            # blend of precision and recall
            f1 = f1_score(labels, predictions)
            experiment.log_metric("batch f1", f1, step=batch_step)
            

            batch_step += 1

            #print("batch", batch_id, "batch accuracy: ", round(accuracy, 2))
            #print("batch", batch_id, "batch precision: ", round(precision, 2))
            #print("batch", batch_id, "batch recall: ", round(recall, 2))
            #print("batch", batch_id, "batch f1: ", round(f1, 2))
            if nbatches % 10000 == 0:
                print("batch", batch_id, print(loss))
                print("batch", batch_id, "batch accuracy: ", round(accuracy, 2))
                print("batch", batch_id, "batch precision: ", round(precision, 2))
                print("batch", batch_id, "batch recall: ", round(recall, 2))
                print("batch", batch_id, "batch f1: ", round(f1, 2))


    # epoch callbacks
    # epoch loss, epoch f1, accuracy, recall, precision for train dataset
    epoch_loss = aggregated_loss / nbatches
    experiment.log_metric("epoch loss", epoch_loss, step=epoch_step)


    epoch_accuracy = (aggregated_tn + aggregated_tp)/(aggregated_tp + aggregated_tn + aggregated_fp + aggregated_fn)
    experiment.log_metric("epoch accuracy", epoch_accuracy, step=epoch_step)

    epoch_recall = aggregated_tp/(aggregated_tp + aggregated_fn + 1e-8)
    experiment.log_metric("epoch recall", epoch_recall, step=epoch_step)

    epoch_precision = aggregated_tp/(aggregated_tp + aggregated_fp + 1e-8)
    experiment.log_metric("epoch precision", epoch_precision, step=epoch_step)

    epoch_f1 = 2*(epoch_precision*epoch_recall)/(epoch_precision + epoch_recall + 1e-8)
    experiment.log_metric("epoch f1", epoch_f1, step=epoch_step)

    # epoch loss, f1, accuracy, recall, precision for test dataset
    model.eval()

    nbatches = 0
    aggregated_loss = 0
    aggregated_tp = 0
    aggregated_tn = 0
    aggregated_fp = 0
    aggregated_fn = 0

    with torch.no_grad():
        for batch in test_loader:
            nbatches += 1
            out = model(batch.x, batch.edge_index)
            #loss = criterion(out.squeeze(), batch.y[:,1]).numpy()
            #loss = criterion(out.squeeze(), batch.y[:,1], batch.y[:,0]).numpy() # weighted BCE
            if config["loss"] == "FocalLoss":
                loss = criterion(out.squeeze(), batch.y[:,1]).numpy()
                
            if config["loss"] == "BCE_et_weighted":
                # weights for every cluster in the batch
                cluster_weights = get_et_weights(batch, et_bins, noise_bin_weight, seed_bin_weight, matched_not_seed_bin_weight)
                loss = criterion(out.squeeze(), batch.y[:,1], cluster_weights).numpy()
            
            aggregated_loss += loss

            # metrics
            labels = batch.y[:,1]
            predictions = (torch.sigmoid(out.squeeze()) > 0.5).float()

            aggregated_tp += ((labels == 1) & (predictions == 1)).sum().item()
            aggregated_tn += ((labels == 0) & (predictions == 0)).sum().item()
            aggregated_fp += ((labels == 0) & (predictions == 1)).sum().item()
            aggregated_fn += ((labels == 1) & (predictions == 0)).sum().item()
    
    epoch_test_loss = aggregated_loss / nbatches
    experiment.log_metric("epoch test loss", epoch_test_loss, step=epoch_step)

    epoch_test_accuracy = (aggregated_tn + aggregated_tp)/(aggregated_tp + aggregated_tn + aggregated_fp + aggregated_fn)
    experiment.log_metric("epoch test accuracy", epoch_test_accuracy, step=epoch_step)

    epoch_test_recall = aggregated_tp/(aggregated_tp + aggregated_fn + 1e-8)
    experiment.log_metric("epoch test recall", epoch_test_recall, step=epoch_step)

    epoch_test_precision = aggregated_tp/(aggregated_tp + aggregated_fp + 1e-8)
    experiment.log_metric("epoch test precision", epoch_test_precision, step=epoch_step)

    epoch_test_f1 = 2*(epoch_test_precision*epoch_test_recall)/(epoch_test_precision + epoch_test_recall + 1e-8)
    experiment.log_metric("epoch test f1", epoch_test_f1, step=epoch_step)


    epoch_step += 1

    # save weights after each epoch
    weight_path = f"{outdir}/weights.{epoch:02d}-{loss:.6f}.pth"
    torch.save(model.state_dict(), weight_path)

