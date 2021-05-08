import torch.nn as nn
import torch.nn.functional as F
import torch

class cluster(nn.Module):
    def __init__(self, input_shape):
        super(cluster, self).__init__()
        #self.fc0 = nn.Linear(input_shape, 128)
        self.fc0 = nn.Linear(input_shape, 128)


    def forward(self, inputs):
        x = F.relu(self.fc0(inputs))
        return x

if __name__ == '__main__':
    pass
