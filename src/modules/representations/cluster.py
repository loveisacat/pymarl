import torch.nn as nn
import torch.nn.functional as F
import torch

class cluster(nn.Module):
    def __init__(self, input_shape):
        super(cluster, self).__init__()
        self.fc1 = nn.Linear(input_shape, 32)


    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        return x

if __name__ == '__main__':
    pass
