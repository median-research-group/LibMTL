import torch, argparse, os
import torch.nn as nn
import torch.nn.functional as F

from transformers import AdamW, logging, BertModel
logging.set_verbosity_error()
logging.set_verbosity_warning()
from create_dataset import DataloaderSC
from utils import *

from LibMTL import Trainer
from LibMTL.utils import set_random_seed, set_device
from LibMTL.config import LibMTL_args, prepare_args
from LibMTL.metrics import AccMetric

def parse_args(parser):
    parser.add_argument('--dataset', default='pawsx', type=str, help='pawsx')
    parser.add_argument('--bs', default=32, type=int, help='batch size')
    parser.add_argument('--epochs', default=100, type=int, help='training epochs')
    parser.add_argument('--dataset_path', default='/', type=str, help='dataset path')
    return parser.parse_args()

def main(params):
    kwargs, optim_param, scheduler_param = prepare_args(params)
    
    lang_list = ['en', 'zh', 'de', 'es']
    
    data_dir = os.path.join(params.dataset_path, params.dataset)
    dataloader, _, labels = DataloaderSC(lang_list=lang_list,
                                      model_name_or_path='bert-base-multilingual-cased',
                                      model_type='bert',
                                      mode_list=['train', 'dev', 'test'],
                                      data_dir=data_dir,
                                      max_seq_length=128,
                                      batch_size=params.bs)
    train_dataloaders = {task: dataloader[task]['train'] for task in lang_list}
    val_dataloaders = {task: dataloader[task]['dev'] for task in lang_list}
    test_dataloaders = {task: dataloader[task]['test'] for task in lang_list}
    
    # define tasks
    task_dict = {task: {'metrics': ['Acc'],
                       'metrics_fn': AccMetric(),
                       'loss_fn': SCLoss(label_num=len(labels)),
                       'weight': [1]} for task in lang_list}
    
    # define encoder and decoders
    class Encoder(nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            self.bert = BertModel.from_pretrained('bert-base-multilingual-cased', add_pooling_layer=True)
        
        def forward(self, inputs):
            outputs = self.bert(input_ids=inputs['input_ids'],
                           attention_mask=inputs['attention_mask'],
                           token_type_ids=inputs['token_type_ids'])
            return outputs[1]
    decoders = nn.ModuleDict({task: nn.Sequential(nn.Dropout(p=0.1, inplace=False),
                                                  nn.Linear(768, len(labels))) for task in lang_list})
    
    class SCtrainer(Trainer):
        def __init__(self, task_dict, weighting, architecture, encoder_class, 
                     decoders, rep_grad, multi_input, optim_param, scheduler_param, **kwargs):
            super(SCtrainer, self).__init__(task_dict=task_dict, 
                                            weighting=weighting, 
                                            architecture=architecture, 
                                            encoder_class=encoder_class, 
                                            decoders=decoders,
                                            rep_grad=rep_grad,
                                            multi_input=multi_input,
                                            optim_param=optim_param,
                                            scheduler_param=scheduler_param,
                                            **kwargs)
            
        def _process_data(self, loader):
            try:
                data_batch = loader[1].next()
            except:
                loader[1] = iter(loader[0])
                data_batch = loader[1].next()
            data_batch = tuple(t.cuda(non_blocking=True) for t in data_batch if t is not None)
            inputs = {"input_ids": data_batch[0], 
                      "attention_mask": data_batch[1], 
                      "token_type_ids": data_batch[2]}
            label = data_batch[3]
            return inputs, label
        
        def _prepare_optimizer(self, optim_param, scheduler_param):
            self.optimizer = AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)
            self.scheduler = None
            
    SCModel = SCtrainer(task_dict=task_dict, 
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
        SCModel.train(train_dataloaders=train_dataloaders, 
                      val_dataloaders=val_dataloaders,
                      test_dataloaders=test_dataloaders, 
                      epochs=params.epochs)
    elif params.mode == 'test':
        SCModel.test(test_dataloaders)
    else:
        raise ValueError
    
if __name__ == "__main__":
    params = parse_args(LibMTL_args)
    # set device
    set_device(params.gpu_id)
    # set random seed
    set_random_seed(params.seed)
    main(params)