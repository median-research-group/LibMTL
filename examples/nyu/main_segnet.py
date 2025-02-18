import torch, argparse
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from segnet_mtan import SegNet_MTAN_encoder, SegNet_MTAN_decoder
from create_dataset import NYUv2

from LibMTL import Trainer
from LibMTL.model import resnet_dilated
from LibMTL.utils import set_random_seed, set_device
from LibMTL.config import LibMTL_args, prepare_args

def parse_args(parser):
    parser.add_argument('--aug', action='store_true', default=False, help='data augmentation')
    parser.add_argument('--train_bs', default=2, type=int, help='batch size for training')
    parser.add_argument('--test_bs', default=2, type=int, help='batch size for test')
    parser.add_argument('--epochs', default=200, type=int, help='training epochs')
    parser.add_argument('--dataset_path', default='/', type=str, help='dataset path')
    return parser.parse_args()
    
def main(params):
    kwargs, optim_param, scheduler_param = prepare_args(params)

    # prepare dataloaders
    nyuv2_train_set = NYUv2(root=params.dataset_path, mode='train', augmentation=params.aug)
    nyuv2_test_set = NYUv2(root=params.dataset_path, mode='test', augmentation=False)
    
    nyuv2_train_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_train_set,
        batch_size=params.train_bs,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True)
    
    nyuv2_test_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_test_set,
        batch_size=params.test_bs,
        shuffle=False,
        num_workers=8,
        pin_memory=True)
    
    # define tasks
    task_dict = {'segmentation': {'metrics':['mIoU', 'pixAcc'], 
                              'metrics_fn': SegMetric(num_classes=13),
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
    
    # define encoder and decoders
    def encoder_class(): 
        return SegNet_MTAN_encoder()
    decoders = nn.ModuleDict({task: SegNet_MTAN_decoder(task) for task in list(task_dict.keys())})
    
    NYUmodel = Trainer(task_dict=task_dict, 
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
        NYUmodel.train(nyuv2_train_loader, nyuv2_test_loader, params.epochs)
    elif params.mode == 'test':
        NYUmodel.test(nyuv2_test_loader)
    else:
        raise ValueError
    
if __name__ == "__main__":
    params = parse_args(LibMTL_args)
    # set device
    set_device(params.gpu_id)
    # set random seed
    set_random_seed(params.seed)
    main(params)
