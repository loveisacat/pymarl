import torch.nn as nn
import torch.nn.functional as F
import torch

class Transformer(nn.Module):
    def __init__(self, input_shape):
        super(Transformer, self).__init__()
        self.q = nn.Linear(in_features=input_shape, out_features=32, bias=False)
        self.k = nn.Linear(in_features=input_shape, out_features=32, bias=False)
        self.v = nn.Linear(in_features=input_shape, out_features=32, bias=False)


    def forward(self, inputs):
        q = self.q(inputs)
        k = self.k(inputs)
        v = self.v(inputs)
        qk = torch.matmul(q, k.permute(1, 0))
        qk = qk / (32 ** 0.5)
        qk = F.softmax(qk, dim=1)
        v = torch.matmul(qk, v)

        return v
