import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, nnodes):
        self.nnodes = nnodes
        super().__init__()
        torch.manual_seed(1000) #if we want reproducibility -> same weight initialization
        self.conv1 = GCNConv(21, self.nnodes) #21 = node features, 16 = random number for hidden layers
        self.conv2 = GCNConv(self.nnodes, 1)
        #self.conv3 = GCNConv(4, 1) #1 since we want a single number for out classifier

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        out = self.conv2(x, edge_index)
        #x = F.relu(x)
        #out = self.conv3(x, edge_index)
        #out = torch.sigmoid(x) #dont use sigmoid, we will use BCEWith_logits as loss. this applies sigmoid later to get probabilities

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


class Weighted_BCE(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
    
    def forward(self, inputs, truth_labels, calomatched_mask):
        # inputs: raw logits from the model output (shape: #clusters)
        # truth_labels: ground truth labels (shape: #clusters)
        # calomatched_mask: mask with 1 if calomatched and 0 if not
        
        # apply sigmoid and calculate ordinary binary cross-entropy loss
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, truth_labels, reduction='none')
        
        # loss_weight: give fewer importance to noncalomatched samples
        loss_weight = torch.where(calomatched_mask == 0, self.weight, 1)
        
        # weight the loss
        weighted_loss = BCE_loss * loss_weight
        
        return weighted_loss.mean()  # return the average loss
