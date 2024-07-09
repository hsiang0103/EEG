import torch
import torch.nn as nn

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