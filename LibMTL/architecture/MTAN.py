import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.architecture.abstract_arch import AbsArchitecture

class _transform_resnet_MTAN(nn.Module):
    def __init__(self, resnet_network, task_name, device):
        super(_transform_resnet_MTAN, self).__init__()
        
        self.task_name = task_name
        self.task_num = len(task_name)
        self.device = device
        self.forward_task = None
        
        self.expansion = 4 if resnet_network.feature_dim == 2048 else 1
        ch = np.array([64, 128, 256, 512]) * self.expansion
        self.shared_conv = nn.Sequential(resnet_network.conv1, resnet_network.bn1, 
                                         resnet_network.relu, resnet_network.maxpool)
        self.shared_layer, self.encoder_att, self.encoder_block_att = nn.ModuleDict({}), nn.ModuleDict({}), nn.ModuleList([])
        for i in range(4):
            self.shared_layer[str(i)] = nn.ModuleList([eval('resnet_network.layer'+str(i+1)+'[:-1]'), 
                                                       eval('resnet_network.layer'+str(i+1)+'[-1]')])
            
            if i == 0:
                self.encoder_att[str(i)] = nn.ModuleList([self._att_layer(ch[0], 
                                                                          ch[0]//self.expansion,
                                                                          ch[0]).to(self.device) for _ in range(self.task_num)])
            else:
                self.encoder_att[str(i)] = nn.ModuleList([self._att_layer(2*ch[i], 
                                                                            ch[i]//self.expansion, 
                                                                            ch[i]).to(self.device) for _ in range(self.task_num)])
                
            if i < 3:
                self.encoder_block_att.append(self._conv_layer(ch[i], ch[i+1]//self.expansion).to(self.device))
                
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def _att_layer(self, in_channel, intermediate_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=intermediate_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(intermediate_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=intermediate_channel, out_channels=out_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.Sigmoid())
        
    def _conv_layer(self, in_channel, out_channel):
        from LibMTL.model.resnet import conv1x1
        downsample = nn.Sequential(conv1x1(in_channel, self.expansion * out_channel, stride=1),
                                   nn.BatchNorm2d(self.expansion * out_channel))
        if self.expansion == 4:
            from LibMTL.model.resnet import Bottleneck
            return Bottleneck(in_channel, out_channel, downsample=downsample)
        else:
            from LibMTL.model.resnet import BasicBlock
            return BasicBlock(in_channel, out_channel, downsample=downsample)
        
    def forward(self, inputs):
        s_rep = self.shared_conv(inputs)
        ss_rep = {i: [0]*2 for i in range(4)}
        att_rep = [0]*self.task_num
        for i in range(4):
            for j in range(2):
                if i == 0 and j == 0:
                    sh_rep = s_rep
                elif i != 0 and j == 0:
                    sh_rep = ss_rep[i-1][1]
                else:
                    sh_rep = ss_rep[i][0]
                ss_rep[i][j] = self.shared_layer[str(i)][j](sh_rep)
            
            for tn, task in enumerate(self.task_name):
                if self.forward_task is not None and task != self.forward_task:
                    continue
                if i == 0:
                    att_mask = self.encoder_att[str(i)][tn](ss_rep[i][0])
                else:
                    if ss_rep[i][0].size()[-2:] != att_rep[tn].size()[-2:]:
                        att_rep[tn] = self.down_sampling(att_rep[tn])
                    att_mask = self.encoder_att[str(i)][tn](torch.cat([ss_rep[i][0], att_rep[tn]], dim=1))
                att_rep[tn] = att_mask * ss_rep[i][1]
                if i < 3:
                    att_rep[tn] = self.encoder_block_att[i](att_rep[tn])
                if i == 0:
                    att_rep[tn] = self.down_sampling(att_rep[tn])
        if self.forward_task is None:
            return att_rep
        else:
            return att_rep[self.task_name.index(self.forward_task)]

    
class MTAN(AbsArchitecture):
    r"""Multi-Task Attention Network (MTAN).
    
    This method is proposed in `End-To-End Multi-Task Learning With Attention (CVPR 2019) <https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.pdf>`_ \
    and implemented by modifying from the `official PyTorch implementation <https://github.com/lorenmt/mtan>`_. 

    .. warning::
            :class:`MTAN` is only supported by ResNet-based encoders.

    """
    def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs):
        super(MTAN, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)
        
        self.encoder = self.encoder_class()
        try: 
            callable(eval('self.encoder.layer1'))
            self.encoder = _transform_resnet_MTAN(self.encoder.to(device), task_name, device)
        except:
            self.encoder.resnet_network = _transform_resnet_MTAN(self.encoder.resnet_network.to(device), task_name, device)
            
    def forward(self, inputs, task_name=None):
        out = {}
        if self.multi_input:
            try:
                callable(eval('self.encoder.resnet_network'))
                self.encoder.resnet_network.forward_task = task_name
            except:
                self.encoder.forward_task = task_name
        s_rep = self.encoder(inputs)
        for tn, task in enumerate(self.task_name):
            if task_name is not None and task != task_name:
                continue
            ss_rep = s_rep[tn] if isinstance(s_rep, list) else s_rep
            ss_rep = self._prepare_rep(ss_rep, task, same_rep=False)
            out[task] = self.decoders[task](ss_rep)
        return out
        
    def get_share_params(self):
        try:
            callable(eval('self.encoder.resnet_network'))
            r = self.encoder.resnet_network
        except:
            r = self.encoder
        p = []
        p += r.shared_conv.parameters()
        p += r.shared_layer.parameters()
        if r != self.encoder:
            for n, param in self.encoder.named_parameters():
                if 'resnet_network' not in n:
                    p.append(param)
        return p

    def zero_grad_share_params(self):
        try:
            callable(eval('self.encoder.resnet_network'))
            r = self.encoder.resnet_network
        except:
            r = self.encoder
            for n, m in self.encoder.named_modules():
                if 'resnet_network' not in n:
                    m.zero_grad(set_to_none=False)
        r.shared_conv.zero_grad(set_to_none=False)
        r.shared_layer.zero_grad(set_to_none=False)
