from torchinfo import summary
from ecg_models import *
from torchviz import make_dot
import torch

print("\nECGSMARTNET:")
model = ECGSMARTNET()
summary(model, input_size=(1, 1, 200, 12))

print("\nECGSMARTNET_Attention(attention='se'):")
model = ECGSMARTNET_Attention(attention='se')
summary(model, input_size=(1, 1, 200, 12))