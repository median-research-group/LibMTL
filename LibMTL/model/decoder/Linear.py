import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Module):
    def __init__(self, input_num, output_num):
        super(Linear, self).__init__()
        self.net = nn.Linear(input_num, output_num)

    def forward(self, x):
        return self.net(x)