import torch, argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

from LibMTL import Trainer
from LibMTL.utils import set_random_seed, set_device
from LibMTL.config import LibMTL_args, prepare_args
from LibMTL.loss import MSELoss

from utils import QM9Metric
from torch.nn import GRU, Linear, ReLU, Sequential
import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv
from torch_geometric.nn.aggr import Set2Set
from torch_geometric.utils import remove_self_loops

def parse_args(parser):
    parser.add_argument('--bs', default=128, type=int, help='batch size')
    parser.add_argument('--epochs', default=300, type=int, help='training epochs')
    parser.add_argument('--dataset_path', default='/', type=str, help='dataset path')
    parser.add_argument('--target', default=[0, 1, 2, 3, 5, 6, 12, 13, 14, 15, 11], 
                        type=int, nargs='+', help='target')
    return parser.parse_args()
    
def main(params):
    kwargs, optim_param, scheduler_param = prepare_args(params)

    scheduler_param = {'scheduler': 'reduce',
                   'mode': 'max',
                   'factor': 0.7, 
                   'patience': 5,
                   'min_lr': 0.00001}

    target = params.target

    class Complete(object):
        def __call__(self, data):
            device = data.edge_index.device

            row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
            col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

            row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
            col = col.repeat(data.num_nodes)
            edge_index = torch.stack([row, col], dim=0)

            edge_attr = None
            if data.edge_attr is not None:
                idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
                size = list(data.edge_attr.size())
                size[0] = data.num_nodes * data.num_nodes
                edge_attr = data.edge_attr.new_zeros(size)
                edge_attr[idx] = data.edge_attr

            edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
            data.edge_attr = edge_attr
            data.edge_index = edge_index

            return data

    transform = T.Compose([Complete(), T.Distance(norm=False)])
    dataset = QM9(params.dataset_path, transform=transform)

    # Normalize targets to mean = 0 and std = 1.
    mean = dataset.data.y.mean(dim=0, keepdim=True)
    std = dataset.data.y.std(dim=0, keepdim=True)
    dataset.data.y = (dataset.data.y - mean) / std

    class QM9Dataset(Dataset):
        def __init__(self, dataset, target: list):
            self.dataset = dataset
            self.target = target

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            data = self.dataset.__getitem__(idx)
            label = {}
            for tn in self.target:
                label[str(tn)] = data.y[:, tn]
            return data, label

    # Split datasets.
    split = torch.load('./random_split.t')
    test_dataset = QM9Dataset(dataset[split][:10000], target)
    val_dataset = QM9Dataset(dataset[split][10000:20000], target)
    train_dataset = QM9Dataset(dataset[split][20000:], target)

    test_loader = DataLoader(test_dataset, batch_size=params.bs, shuffle=False, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=params.bs, shuffle=False, num_workers=2, pin_memory=True)
    train_loader = DataLoader(train_dataset, batch_size=params.bs, shuffle=True, num_workers=2, pin_memory=True)

    # define tasks
    task_dict = {}
    for _, t in enumerate(target):
        if t in [2, 3, 6, 12, 13, 14, 15]:
            scale = 1000
        else:
            scale = 1
        task_dict[str(t)] = {'metrics':['MAE'], 
                          'metrics_fn': QM9Metric(std[:,t], scale),
                          'loss_fn': MSELoss(),
                          'weight': [0]}
    
    # define encoder and decoders
    class Net(torch.nn.Module):
        def __init__(self, dim=64):
            super().__init__()
            self.lin0 = torch.nn.Linear(11, dim)

            nn = Sequential(Linear(5, 128), ReLU(), Linear(128, dim * dim))
            self.conv = NNConv(dim, dim, nn, aggr='mean')
            self.gru = GRU(dim, dim)

            self.set2set = Set2Set(dim, processing_steps=3)
            self.lin1 = torch.nn.Linear(2 * dim, dim)
            # self.lin2 = torch.nn.Linear(dim, 1)

        def forward(self, data):
            out = F.relu(self.lin0(data.x))
            h = out.unsqueeze(0)

            for i in range(3):
                m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
                out, h = self.gru(m.unsqueeze(0), h)
                out = out.squeeze(0)

            out = self.set2set(out, data.batch)
            out = F.relu(self.lin1(out))
            # out = self.lin2(out)
            return out#.view(-1)
    def encoder_class(): 
        return Net()
    decoders = nn.ModuleDict({task: nn.Linear(64, 1) for task in list(task_dict.keys())})
    
    class QM9trainer(Trainer):
        def __init__(self, task_dict, weighting, architecture, encoder_class, 
                     decoders, rep_grad, multi_input, optim_param, scheduler_param, **kwargs):
            super(QM9trainer, self).__init__(task_dict=task_dict, 
                                            weighting=weighting, 
                                            architecture=architecture, 
                                            encoder_class=encoder_class, 
                                            decoders=decoders,
                                            rep_grad=rep_grad,
                                            multi_input=multi_input,
                                            optim_param=optim_param,
                                            scheduler_param=scheduler_param,
                                            **kwargs)
    
    QM9model = QM9trainer(task_dict=task_dict, 
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
        QM9model.train(train_loader, test_loader, params.epochs, val_dataloaders=val_loader)
    elif params.mode == 'test':
        QM9model.test(test_loader)
    else:
        raise ValueError
    
if __name__ == "__main__":
    params = parse_args(LibMTL_args)
    # set device
    set_device(params.gpu_id)
    # set random seed
    set_random_seed(params.seed)
    main(params)
