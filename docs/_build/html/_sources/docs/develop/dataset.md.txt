## Apply to a New Dataset

Here we would like to introduce how to apply ``LibMTL`` to a new dataset.

### Define a MTL problem

```eval_rst
Firstly, you need to clear the the type of this MTL problem (i.e. a multi-label problem or multi-input problem, refer to `here <./mtl.html#network-architecture>`_) and the information of each task including the task's name, the instantiation of metric and loss classes, and whether the higher metric score the better performance or not. 

The ``multi_input`` is a command-line argument while all tasks' information need to be defined as a dictionary. ``LibMTL`` provides some common loss functions and metrics, please refer to :class:`LibMTL.loss` and :class:`LibMTL.metrics`, respectively. The example of a three-tasks MTL problem on the Office-31 dataset is as follows.
```

#### Example 1 (The Office-31 Dataset)

```python
from LibMTL.loss import CELoss
from LibMTL.metrics import AccMetric

# define tasks
task_name = ['amazon', 'dslr', 'webcam']
task_dict = {task: {'metrics': ['Acc'],
                    'metrics_fn': AccMetric(),
                    'loss_fn': CELoss(),
                    'weight': [1]} for task in task_name}
```

```eval_rst
Besides, ``LibMTL`` also supports users to customize new loss and metric classes. For example, if we would like to develop the metric classes for the segmentation task on the NYUv2 dataset, we need to inherit :class:`LibMTL.metrics.AbsMetric` and rewrite the corresponding methods like :func:`update_fun`, :func:`score_fun`, and :func:`reinit` here, please see :class:`LibMTL.metrics.AbsMetric` for details. The loss class for segmentation is customized similarly, please see :class:`LibMTL.loss.AbsLoss` for details.
```

#### Example 2 (The NYUv2 Dataset)

```python
from LibMTL.metrics import AbsMetric

# seg
class SegMetric(AbsMetric):
    def __init__(self):
        super(SegMetric, self).__init__()
        
        self.num_classes = 13
        self.record = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)
        
    def update_fun(self, pred, gt):
        self.record = self.record.to(pred.device)
        pred = pred.softmax(1).argmax(1).flatten()
        gt = gt.long().flatten()
        k = (gt >= 0) & (gt < self.num_classes)
        inds = self.num_classes * gt[k].to(torch.int64) + pred[k]
        self.record += torch.bincount(inds, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)
        
    def score_fun(self):
        h = self.record.float()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        acc = torch.diag(h).sum() / h.sum()
        return [torch.mean(iu).item(), acc.item()]
    
    def reinit(self):
        self.record = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)
```

```eval_rst
The customized loss and metric classes of three tasks on the NYUv2 dataset are put in ``examples/nyu/utils.py``. After that, the three-tasks MTL problem on the NYUv2 dataset is defined as follows. 
```

```python
from utils import *

# define tasks
task_dict = {'segmentation': {'metrics':['mIoU', 'pixAcc'], 
                              'metrics_fn': SegMetric(),
                              'loss_fn': SegLoss(),
                              'weight': [1, 1]}, 
             'depth': {'metrics':['abs_err', 'rel_err'], 
                       'metrics_fn': DepthMetric(),
                       'loss_fn': DepthLoss(),
                       'weight': [0, 0]},
             'normal': {'metrics':['mean', 'median', '<11.25', '<22.5', '<30'], 
                        'metrics_fn': NormalMetric(),
                        'loss_fn': NormalLoss(),
                        'weight': [0, 0, 1, 1, 1]}}
```

### Prepare Dataloaders

```eval_rst
Secondly, you need to prepare the dataloaders with the correct format. For multi-input problem like the Office-31 datatset, each task need to have its own dataloader and all dataloaders are put in a dictionary with the task names as the corresponding keys.
```

#### Example 1 (The Office-31 Dataset)

```python
train_dataloaders = {'amazon': amazon_dataloader,
                     'dslr': dslr_dataloader,
                     'webcam': webcam_dataloader}
```

```eval_rst
For multi-label problem like the NYUv2 dataset, all tasks share a common dataloader, which outputs a list in every iteration. The first element of this list is the input data tensor and the second element is a dictionary of the label tensors with the task names as the corresponding keys. An example is shown as follows. 
```

#### Example 2 (The NYUv2 Dataset)

```python
nyuv2_train_loader = xx
# print(iter(nyuv2_train_loader).next())
# [torch.Tensor, {'segmentation': torch.Tensor,
# 		  'depth': torch.Tensor,
# 		  'normal': torch.Tensor}]
```

### Define Encoder and Decoders

```eval_rst
Thirdly, you need to define the shared encoder and task-specific decoders. ``LibMTL`` provides some common networks like ResNet-based network, please see :class:`LibMTL.model` for details. Also, you can customize the encoder and decoders.

Note that the encoder does not be instantiated while the decoders should be instantiated.
```

#### Example 1 (The Office-31 Dataset)

```python
import torch
import torch.nn as nn
from LibMTL.model import resnet18

# define encoder and decoders
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        hidden_dim = 512
        self.resnet_network = resnet18(pretrained=True)
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

decoders = nn.ModuleDict({task: nn.Linear(512, class_num) for task in task_name})
```

```eval_rst
If the customized encoder is a ResNet-based network and you would like to use :class:`LibMTL.architecture.MTAN`, please make sure encoder has an attribute named ``resnet_network`` and corresponded to the ResNet network.
```

#### Example 2 (The NYUv2 Dataset)

```python
from aspp import DeepLabHead
from LibMTL.model import resnet_dilated

# define encoder and decoders
def encoder_class():
	return resnet_dilated('resnet50')
num_out_channels = {'segmentation': 13, 'depth': 1, 'normal': 3}
decoders = nn.ModuleDict({task: DeepLabHead(encoder.feature_dim, 
                                            num_out_channels[task]) for task in list(task_dict.keys())})
```

### Instantiate the Training Framework

```eval_rst
Fourthly, you need to instantiate the training framework, please see :class:`LibMTL.Trainer` for more details.
```

#### Example 1 (The Office-31 Dataset)

```python
from LibMTL import Trainer

officeModel = Trainer(task_dict=task_dict, 
                      weighting=weighting_method.__dict__[params.weighting], 
                      architecture=architecture_method.__dict__[params.arch], 
                      encoder_class=Encoder, 
                      decoders=decoders,
                      rep_grad=params.rep_grad,
                      multi_input=params.multi_input,
                      optim_param=optim_param,
                      scheduler_param=scheduler_param,
                      **kwargs)
```

```eval_rst
Also, you can inherit :class:`LibMTL.Trainer` class and rewrite some functions like :func:`process_preds`.
```

#### Example 2 (The NYUv2 Dataset)

```python
from LibMTL import Trainer

class NYUtrainer(Trainer):
    def __init__(self, task_dict, weighting, architecture, encoder_class, 
                 decoders, rep_grad, multi_input, optim_param, scheduler_param, **kwargs):
        super(NYUtrainer, self).__init__(task_dict=task_dict, 
                                        weighting=weighting_method.__dict__[weighting], 
                                        architecture=architecture_method.__dict__[architecture], 
                                        encoder_class=encoder_class, 
                                        decoders=decoders,
                                        rep_grad=rep_grad,
                                        multi_input=multi_input,
                                        optim_param=optim_param,
                                        scheduler_param=scheduler_param,
                                        **kwargs)

    def process_preds(self, preds):
        img_size = (288, 384)
        for task in self.task_name:
            preds[task] = F.interpolate(preds[task], img_size, mode='bilinear', align_corners=True)
        return preds

NYUmodel = NYUtrainer(task_dict=task_dict, 
                      weighting=params.weighting, 
                      architecture=params.arch, 
                      encoder_class=encoder_class, 
                      decoders=decoders,
                      rep_grad=params.rep_grad,
                      multi_input=params.multi_input,
                      optim_param=optim_param,
                      scheduler_param=scheduler_param,
                      **kwargs)
```

### Run a Model

```eval_rst
Finally, you can training the model using :func:`train` function like this.
```

```python
officeModel.train(train_dataloaders=train_dataloaders, 
                  val_dataloaders=val_dataloaders,
                  test_dataloaders=test_dataloaders, 
                  epochs=100)
```

```eval_rst
When the training process ends, the best results on the test dataset will be printed automatically, please see :func:`LibMTL.Trainer.train` and :func:`LibMTL.utils.count_improvement` for details.
```



