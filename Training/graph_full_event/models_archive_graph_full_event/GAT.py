import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv



class GAT(torch.nn.Module):
    def __init__(self, nnodes):
        super().__init__()
        torch.manual_seed(1000) #if we want reproducibility -> same weight initialization
        self.nnodes = nnodes
        self.conv1 = GATConv(21, nnodes, heads=4, concat=False)  
        self.conv2 = GATConv(nnodes, 1, heads=2, concat=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        out = self.conv2(x, edge_index)

        return out

class GAT_dropout(torch.nn.Module):
    def __init__(self, nnodes, dropout=0.3):
        super().__init__()
        torch.manual_seed(1000) #if we want reproducibility -> same weight initialization
        self.nnodes = nnodes
        self.dropout = dropout
        self.conv1 = GATConv(21, nnodes, heads=4, concat=False)  
        self.conv2 = GATConv(nnodes, 1, heads=2, concat=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training) 
        out = self.conv2(x, edge_index)

        return out
    
    
    
#FocalLoss if no sigmoid as last model layer
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, truth_labels):
        # inputs: raw logits from the model output (shape: #clusters)
        # truth_labels: ground truth labels (shape: #clusters)
        
        # inputs are logits, applying sigmoid results in probabilities
        predicted_probs = torch.sigmoid(inputs)
        
        # calculate binary cross-entropy loss
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, truth_labels, reduction='none')
        
        # compute focal loss
        
        # p_t = probability of correct class
        # if truth label is 1 p_t = predicted_probs, if truth label is 0, p_t = (1 - predicted_probs)
        # p_t close to 1 if we predicted well, close to 0 if we predicted badly
        p_t = predicted_probs * truth_labels + (1 - predicted_probs) * (1 - truth_labels)
        
        # weighting of minority class, if truth_label = 1 we get alpha/ if truth_label = 0 we get 1 - alpha
        # if imbalanced with truth_label = 1 minority, set alpha between (0.5, 1). if opposite is the case (0, 0.5)
        alpha_t = self.alpha * truth_labels + (1 - self.alpha) * (1 - truth_labels)
        
        # (1 - p_t)**gamma is small if we predicted well and larger if we predicted badly
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * BCE_loss
        
        return focal_loss.mean()  # return the average loss



class BCE_et_weighted(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, inputs, truth_labels, cluster_weights):
        # inputs: raw logits from the model output (shape: #clusters)
        # truth_labels: ground truth labels (shape: #clusters)
        # cluster_weights: weight of each cluster in the loss
        
        # apply sigmoid and calculate ordinary binary cross-entropy loss
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, truth_labels, reduction='none')
        
        # weight the loss
        weighted_loss = BCE_loss * cluster_weights
        
        return weighted_loss.mean()  # return the average loss


