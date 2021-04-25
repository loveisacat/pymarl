import torch.nn as nn
import torch.nn.functional as F
import torch

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, inputs):
        
        x = self.conv(inputs.unsqueeze(1))
        x = F.relu(x)
        x = self.pool(x)
        x = x.squeeze(2)

        return x

if __name__ == '__main__':
    pass
