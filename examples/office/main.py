import torch, argparse
import torch.nn as nn
import torch.nn.functional as F

from create_dataset import office_dataloader

from LibMTL import Trainer
from LibMTL.model import resnet18
from LibMTL.utils import set_random_seed, set_device
from LibMTL.config import LibMTL_args, prepare_args
from LibMTL.loss import CELoss
from LibMTL.metrics import AccMetric

def parse_args(parser):
    parser.add_argument('--dataset', default='office-31', type=str, help='office-31, office-home')
    parser.add_argument('--bs', default=64, type=int, help='batch size')
    parser.add_argument('--epochs', default=100, type=int, help='training epochs')
    parser.add_argument('--dataset_path', default='/', type=str, help='dataset path')
    return parser.parse_args()

def main(params):
    kwargs, optim_param, scheduler_param = prepare_args(params)

    if params.dataset == 'office-31':
        task_name = ['amazon', 'dslr', 'webcam']
        class_num = 31
    elif params.dataset == 'office-home':
        task_name = ['Art', 'Clipart', 'Product', 'Real_World']
        class_num = 65
    else:
        raise ValueError('No support dataset {}'.format(params.dataset))
    
    # define tasks
    task_dict = {task: {'metrics': ['Acc'],
                       'metrics_fn': AccMetric(),
                       'loss_fn': CELoss(),
                       'weight': [1]} for task in task_name}
    
    # prepare dataloaders
    data_loader, _ = office_dataloader(dataset=params.dataset, batchsize=params.bs, root_path=params.dataset_path)
    train_dataloaders = {task: data_loader[task]['train'] for task in task_name}
    val_dataloaders = {task: data_loader[task]['val'] for task in task_name}
    test_dataloaders = {task: data_loader[task]['test'] for task in task_name}
    
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

    decoders = nn.ModuleDict({task: nn.Linear(512, class_num) for task in list(task_dict.keys())})
    
    officeModel = Trainer(task_dict=task_dict, 
                          weighting=params.weighting, 
                          architecture=params.arch, 
                          encoder_class=Encoder, 
                          decoders=decoders,
                          rep_grad=params.rep_grad,
                          multi_input=params.multi_input,
                          optim_param=optim_param,
                          scheduler_param=scheduler_param,
                          save_path=params.save_path,
                          load_path=params.load_path,
                          **kwargs)
    if params.mode == 'train':
        officeModel.train(train_dataloaders=train_dataloaders, 
                          val_dataloaders=val_dataloaders,
                          test_dataloaders=test_dataloaders, 
                          epochs=params.epochs)
    elif params.mode == 'test':
        officeModel.test(test_dataloaders)
    else:
        raise ValueError
    
if __name__ == "__main__":
    params = parse_args(LibMTL_args)
    # set device
    set_device(params.gpu_id)
    # set random seed
    set_random_seed(params.seed)
    main(params)
