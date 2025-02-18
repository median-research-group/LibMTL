import torch, argparse, sys
import torch.nn as nn
import torch.nn.functional as F

from create_dataset import CityScapes
sys.path.append('../nyu')
from utils import *
from aspp import DeepLabHead

from LibMTL import Trainer as Trainer
from LibMTL.model import resnet_dilated
from LibMTL.utils import set_random_seed, set_device
from LibMTL.config import LibMTL_args, prepare_args

def parse_args(parser):
    parser.add_argument('--train_mode', default='train', type=str, help='trainval, train')
    parser.add_argument('--train_bs', default=64, type=int, help='batch size for training')
    parser.add_argument('--test_bs', default=64, type=int, help='batch size for test')
    parser.add_argument('--epochs', default=200, type=int, help='training epochs')
    parser.add_argument('--dataset_path', default='/', type=str, help='dataset path')
    return parser.parse_args()
    
def main(params):
    kwargs, optim_param, scheduler_param = prepare_args(params)

    # prepare dataloaders
    city_train_set = CityScapes(root=params.dataset_path, mode=params.train_mode)
    city_test_set = CityScapes(root=params.dataset_path, mode='test')
    
    city_train_loader = torch.utils.data.DataLoader(
        dataset=city_train_set,
        batch_size=params.train_bs,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True)
    
    city_test_loader = torch.utils.data.DataLoader(
        dataset=city_test_set,
        batch_size=params.test_bs,
        shuffle=False,
        num_workers=2,
        pin_memory=True)
    
    # define tasks
    task_dict = {'segmentation': {'metrics':['mIoU', 'pixAcc'], 
                              'metrics_fn': SegMetric(num_classes=7),
                              'loss_fn': SegLoss(),
                              'weight': [1, 1]}, 
                 'depth': {'metrics':['abs_err', 'rel_err'], 
                           'metrics_fn': DepthMetric(),
                           'loss_fn': DepthLoss(),
                           'weight': [0, 0]}}
    
    # define encoder and decoders
    def encoder_class(): 
        return resnet_dilated('resnet50')
    num_out_channels = {'segmentation': 7, 'depth': 1}
    decoders = nn.ModuleDict({task: DeepLabHead(2048, 
                                                num_out_channels[task]) for task in list(task_dict.keys())})
    
    class Citytrainer(Trainer):
        def __init__(self, task_dict, weighting, architecture, encoder_class, 
                     decoders, rep_grad, multi_input, optim_param, 
                     scheduler_param, **kwargs):
            super(Citytrainer, self).__init__(task_dict=task_dict, 
                                            weighting=weighting, 
                                            architecture=architecture, 
                                            encoder_class=encoder_class, 
                                            decoders=decoders,
                                            rep_grad=rep_grad,
                                            multi_input=multi_input,
                                            optim_param=optim_param,
                                            scheduler_param=scheduler_param,
                                            **kwargs)

        def process_preds(self, preds):
            img_size = (128, 256)
            for task in self.task_name:
                preds[task] = F.interpolate(preds[task], img_size, mode='bilinear', align_corners=True)
            return preds
    
    Citymodel = Citytrainer(task_dict=task_dict, 
                          weighting=params.weighting, 
                          architecture=params.arch, 
                          encoder_class=encoder_class, 
                          decoders=decoders,
                          rep_grad=params.rep_grad,
                          multi_input=params.multi_input,
                          optim_param=optim_param,
                          scheduler_param=scheduler_param,
                          save_path=params.save_path,
                          load_path=params.load_path,
                          **kwargs)
    if params.mode == 'train':
        Citymodel.train(city_train_loader, city_test_loader, params.epochs)
    elif params.mode == 'test':
        Citymodel.test(city_test_loader)
    else:
        raise ValueError
    
if __name__ == "__main__":
    params = parse_args(LibMTL_args)
    # set device
    set_device(params.gpu_id)
    # set random seed
    set_random_seed(params.seed)
    main(params)
