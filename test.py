import torchvision.models as models
import torch.nn as nn
encoder_layer =nn.TransformerEncoderLayer(
    d_model=128,
    nhead=16
)
print(nn.TransformerEncoder(encoder_layer, num_layers=6))