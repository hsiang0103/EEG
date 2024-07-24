import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, pos_weight):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.pos_weight = pos_weight
    
    def forward(self, pred, target):
        batch_num = pred.shape[0]
        total_pixel = pred.shape[1]  #* pred.shape[2] * pred.shape[3]
        total_loss = 0.0
        pred_clamp = torch.clamp(pred, min=1e-7, max=1-1e-7)
        for i in range(batch_num):
            loss = -(1/total_pixel) * torch.sum(
                self.pos_weight * target[i] * torch.log(pred_clamp[i]) + 
                (1 - target[i]) * torch.log((1 - pred_clamp[i]))
            )
            total_loss += loss
        return total_loss / batch_num

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2, reduction:str="mean"):
        super(FocalLoss, self).__init__()
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        elif isinstance(alpha, torch.Tensor):
            self.alpha = alpha
        
        self.gamma      = gamma
        self.reduction  = reduction
    
    def forward(self, pred, target):
        # pred_size = [B, 2]
        # target_size = [B]

        batch_num = pred.shape[0]
        total_loss = 0.0
        pred_clamp = F.softmax(torch.clamp(pred, min=1e-7, max=1 - 1e-7), dim=1)

        # one hot vector
        target_one_hot = torch.zeros_like(pred)
        for i in range(batch_num):
            target_one_hot[i, target[i]] = 1

        for i in range(batch_num):
            loss = torch.sum(
                -1 * self.alpha * target_one_hot[i, :] * torch.pow((1 - pred_clamp[i, :]), self.gamma) * torch.log(pred_clamp[i, :])
            )
            total_loss += loss
        
        if self.reduction.lower() == "mean":
            total_loss /= batch_num
        else:
            total_loss = total_loss
        
        return total_loss
        

