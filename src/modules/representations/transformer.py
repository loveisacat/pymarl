import torch.nn as nn
import torch.nn.functional as F
import torch

class Transformer(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Transformer, self).__init__()
        self.out_dim = output_shape
        self.q = nn.Linear(in_features=input_shape, out_features=output_shape, bias=False)
        self.k = nn.Linear(in_features=input_shape, out_features=output_shape, bias=False)
        self.v = nn.Linear(in_features=input_shape, out_features=output_shape, bias=False)


    def forward(self, inputs):
        q = self.q(inputs)
        k = self.k(inputs)
        v = self.v(inputs)
        qk = torch.matmul(q, k.permute(1, 0))
        qk = qk / (self.out_dim ** 0.5)
        qk = F.softmax(qk, dim=1)
        v = torch.matmul(qk, v)

        return v
