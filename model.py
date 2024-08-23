import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution

class Detector_mlp(nn.Module):
    def __init__(self, n_h):
        super(Detector_mlp, self).__init__()

        self.n_h = n_h
        self.fc1 = nn.Linear(n_h, n_h)
        self.fc2 = nn.Linear(n_h, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        s = self.fc2(x)

        return s
    
    def new(self):
        model_new = Detector_mlp(self.n_h)
        for x, y in zip(model_new.parameters(), self.parameters()):
            x.data.copy_(y.data)
        
        return model_new

class Adaptor(nn.Module):
    def __init__(self, n_h):
        super(Adaptor, self).__init__()

        self.fc1 = nn.Linear(n_h, n_h//4)
        self.fc2 = nn.Linear(n_h//4, n_h)

    def forward(self, h):
        h = F.relu(self.fc1(h))
        h = self.fc2(h)

        return h

