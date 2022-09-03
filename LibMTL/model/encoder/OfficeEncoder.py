import torch
import torch.nn as nn
import LibMTL.model.encoder.resnet as resnet 

class OfficeEncoder(nn.Module):
    def __init__(self, basenet):
        super(OfficeEncoder, self).__init__()

        hidden_dim = 512
        self.resnet_network = resnet.__dict__[basenet](pretrained=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.hidden_layer_list = [nn.Linear(512, hidden_dim),
                                  nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.5)]
        self.hidden_layer = nn.Sequential(*self.hidden_layer_list)

        # initialization
        self.hidden_layer[0].weight.data.normal_(0, 0.005)
        self.hidden_layer[0].bias.data.fill_(0.1)
        
    def forward(self, inputs):
        out = self.resnet_network(inputs)
        out = torch.flatten(self.avgpool(out), 1)
        out = self.hidden_layer(out)
        return out