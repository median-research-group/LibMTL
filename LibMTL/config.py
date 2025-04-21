import argparse
import numpy as np
import torch

_parser = argparse.ArgumentParser(description='Configuration for LibMTL')
# general
_parser.add_argument('--mode', type=str, default='train', help='train, test')
_parser.add_argument('--seed', type=int, default=0, help='random seed')
_parser.add_argument('--gpu_id', default='0', type=str, help='gpu_id') 
_parser.add_argument('--weighting', type=str, default='EW',
    help='loss weighing strategies, option: EW, UW, GradNorm, GLS, RLW, \
        MGDA, PCGrad, GradVac, CAGrad, GradDrop, DWA, IMTL')
_parser.add_argument('--arch', type=str, default='HPS',
                    help='architecture for MTL, option: HPS, MTAN')
_parser.add_argument('--rep_grad', action='store_true', default=False, 
                    help='computing gradient for representation or sharing parameters')
_parser.add_argument('--multi_input', action='store_true', default=False, 
                    help='whether each task has its own input data')
_parser.add_argument('--save_path', type=str, default=None, 
                    help='save path')
_parser.add_argument('--load_path', type=str, default=None, 
                    help='load ckpt path')
## optim
_parser.add_argument('--optim', type=str, default='adam',
                    help='optimizer for training, option: adam, sgd, adagrad, rmsprop')
_parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for all types of optim')
_parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd')
_parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay for all types of optim')
## scheduler
_parser.add_argument('--scheduler', type=str, #default='step',
                    help='learning rate scheduler for training, option: step, cos, exp')
_parser.add_argument('--step_size', type=int, default=100, help='step size for StepLR')
_parser.add_argument('--gamma', type=float, default=0.5, help='gamma for StepLR')

# args for weighting
## DWA
_parser.add_argument('--T', type=float, default=2.0, help='T for DWA')
## MGDA
_parser.add_argument('--mgda_gn', default='none', type=str, 
                    help='type of gradient normalization for MGDA, option: l2, none, loss, loss+')
## GradVac
_parser.add_argument('--GradVac_beta', type=float, default=0.5, help='beta for GradVac')
_parser.add_argument('--GradVac_group_type', type=int, default=0, 
                    help='parameter granularity for GradVac (0: whole_model; 1: all_layer; 2: all_matrix)')
## GradNorm
_parser.add_argument('--alpha', type=float, default=1.5, help='alpha for GradNorm')
## GradDrop
_parser.add_argument('--leak', type=float, default=0.0, help='leak for GradDrop')
## CAGrad
_parser.add_argument('--calpha', type=float, default=0.5, help='calpha for CAGrad')
_parser.add_argument('--rescale', type=int, default=1, help='rescale for CAGrad')
## Nash_MTL
_parser.add_argument('--update_weights_every', type=int, default=1, help='update_weights_every for Nash_MTL')
_parser.add_argument('--optim_niter', type=int, default=20, help='optim_niter for Nash_MTL')
_parser.add_argument('--max_norm', type=float, default=1.0, help='max_norm for Nash_MTL')
## MoCo
_parser.add_argument('--MoCo_beta', type=float, default=0.5, help='MoCo_beta for MoCo')
_parser.add_argument('--MoCo_beta_sigma', type=float, default=0.5, help='MoCo_beta_sigma for MoCo')
_parser.add_argument('--MoCo_gamma', type=float, default=0.1, help='gamma for MoCo')
_parser.add_argument('--MoCo_gamma_sigma', type=float, default=0.5, help='MoCo_gamma_sigma for MoCo')
_parser.add_argument('--MoCo_rho', type=float, default=0, help='MoCo_rho for MoCo')
## DB_MTL
_parser.add_argument('--DB_beta', type=float, default=0.9, help=' ')
_parser.add_argument('--DB_beta_sigma', type=float, default=0, help=' ')
## STCH
_parser.add_argument('--STCH_mu', type=float, default=1.0, help=' ')
_parser.add_argument('--STCH_warmup_epoch', type=int, default=4, help=' ')
## ExcessMTL
_parser.add_argument('--robust_step_size', default=1e-2, type=float, help='step size')
## FairGrad
_parser.add_argument('--FairGrad_alpha', type=float, default=1.0, help=' ')
## FAMO
_parser.add_argument('--FAMO_w_lr', type=float, default=0.025, help=' ')
_parser.add_argument('--FAMO_w_gamma', type=float, default=1e-3, help=' ')
## MoDo
_parser.add_argument('--MoDo_gamma', type=float, default=1e-3, help=' ')
_parser.add_argument('--MoDo_rho', type=float, default=0.1, help=' ')
## SDMGrad
_parser.add_argument('--SDMGrad_lamda', type=float, default=0.3, help=' ')
_parser.add_argument('--SDMGrad_niter', type=int, default=20, help=' ')
## UPGrad
_parser.add_argument('--UPGrad_norm_eps', type=float, default=0.0001,
                     help='A small value to avoid division by zero when normalizing.')
_parser.add_argument('--UPGrad_reg_eps', type=float, default=0.0001,
                     help='A small value to add to the diagonal of the gramian to make it positive definite.')

#### bilevel methods
_parser.add_argument('--outer_lr', type=float, default=1e-3, help='outer lr')
_parser.add_argument('--inner_lr', type=float, default=0.1, help='inner lr')
_parser.add_argument('--inner_step', type=int, default=5, help=' ')
## FORUM
_parser.add_argument('--FORUM_phi', type=float, default=0.1, help=' ') # FORUM

# args for architecture
## CGC
_parser.add_argument('--img_size', nargs='+', help='image size for CGC')
_parser.add_argument('--num_experts', nargs='+', help='the number of experts for sharing and task-specific')
## DSelect_k
_parser.add_argument('--num_nonzeros', type=int, default=2, help='num_nonzeros for DSelect-k')
_parser.add_argument('--kgamma', type=float, default=1.0, help='gamma for DSelect-k')

LibMTL_args = _parser


def prepare_args(params):
    r"""Return the configuration of hyperparameters, optimizier, and learning rate scheduler.

    Args:
        params (argparse.Namespace): The command-line arguments.
    """
    kwargs = {'weight_args': {}, 'arch_args': {}}
    if params.weighting in ['EW', 'UW', 'GradNorm', 'GLS', 'RLW', 'MGDA', 'IMTL',
                            'PCGrad', 'GradVac', 'CAGrad', 'GradDrop', 'DWA', 
                            'Nash_MTL', 'MoCo', 'Aligned_MTL', 'DB_MTL', 'STCH', 
                            'ExcessMTL', 'FairGrad', 'FAMO', 'MoDo', 'SDMGrad', 'UPGrad']:
        if params.weighting in ['DWA']:
            if params.T is not None:
                kwargs['weight_args']['T'] = params.T
            else:
                raise ValueError('DWA needs keywaord T')
        elif params.weighting in ['GradNorm']:
            if params.alpha is not None:
                kwargs['weight_args']['alpha'] = params.alpha
            else:
                raise ValueError('GradNorm needs keywaord alpha')
        elif params.weighting in ['MGDA']:
            if params.mgda_gn is not None:
                if params.mgda_gn in ['none', 'l2', 'loss', 'loss+']:
                    kwargs['weight_args']['mgda_gn'] = params.mgda_gn
                else:
                    raise ValueError('No support mgda_gn {} for MGDA'.format(params.mgda_gn)) 
            else:
                raise ValueError('MGDA needs keywaord mgda_gn')
        elif params.weighting in ['GradVac']:
            if params.GradVac_beta is not None:
                kwargs['weight_args']['GradVac_beta'] = params.GradVac_beta
                kwargs['weight_args']['GradVac_group_type'] = params.GradVac_group_type
            else:
                raise ValueError('GradVac needs keywaord beta')
        elif params.weighting in ['GradDrop']:
            if params.leak is not None:
                kwargs['weight_args']['leak'] = params.leak
            else:
                raise ValueError('GradDrop needs keywaord leak')
        elif params.weighting in ['CAGrad']:
            if params.calpha is not None and params.rescale is not None:
                kwargs['weight_args']['calpha'] = params.calpha
                kwargs['weight_args']['rescale'] = params.rescale
            else:
                raise ValueError('CAGrad needs keywaord calpha and rescale')
        elif params.weighting in ['Nash_MTL']:
            if params.update_weights_every is not None and params.optim_niter is not None and params.max_norm is not None:
                kwargs['weight_args']['update_weights_every'] = params.update_weights_every
                kwargs['weight_args']['optim_niter'] = params.optim_niter
                kwargs['weight_args']['max_norm'] = params.max_norm
            else:
                raise ValueError('Nash_MTL needs update_weights_every, optim_niter, and max_norm')
        elif params.weighting in ['MoCo']:
            kwargs['weight_args']['MoCo_beta'] = params.MoCo_beta
            kwargs['weight_args']['MoCo_beta_sigma'] = params.MoCo_beta_sigma
            kwargs['weight_args']['MoCo_gamma'] = params.MoCo_gamma
            kwargs['weight_args']['MoCo_gamma_sigma'] = params.MoCo_gamma_sigma
            kwargs['weight_args']['MoCo_rho'] = params.MoCo_rho
        elif params.weighting in ['DB_MTL']:
            kwargs['weight_args']['DB_beta'] = params.DB_beta
            kwargs['weight_args']['DB_beta_sigma'] = params.DB_beta_sigma
        elif params.weighting in ['STCH']:
            kwargs['weight_args']['STCH_mu'] = params.STCH_mu
            kwargs['weight_args']['STCH_warmup_epoch'] = params.STCH_warmup_epoch
        elif params.weighting in ['ExcessMTL']:
            kwargs['weight_args']['robust_step_size'] = params.robust_step_size
        elif params.weighting in ['FairGrad']:
            kwargs['weight_args']['FairGrad_alpha'] = params.FairGrad_alpha
        elif params.weighting in ['FAMO']:
            kwargs['weight_args']['FAMO_w_lr'] = params.FAMO_w_lr
            kwargs['weight_args']['FAMO_w_gamma'] = params.FAMO_w_gamma
        elif params.weighting in ['MoDo']:
            kwargs['weight_args']['MoDo_gamma'] = params.MoDo_gamma
            kwargs['weight_args']['MoDo_rho'] = params.MoDo_rho
        elif params.weighting in ['SDMGrad']:
            kwargs['weight_args']['SDMGrad_lamda'] = params.SDMGrad_lamda
            kwargs['weight_args']['SDMGrad_niter'] = params.SDMGrad_niter
        elif params.weighting in ['UPGrad']:
            kwargs['weight_args']['UPGrad_norm_eps'] = params.UPGrad_norm_eps
            kwargs['weight_args']['UPGrad_reg_eps'] = params.UPGrad_reg_eps
    elif params.weighting in ['MOML', 'FORUM', 'AutoLambda']:
        kwargs['weight_args']['outer_lr'] = params.outer_lr
        kwargs['weight_args']['inner_step'] = params.inner_step
        if params.weighting in ['FORUM']:
            kwargs['weight_args']['FORUM_phi'] = params.FORUM_phi
            kwargs['weight_args']['inner_lr'] = params.inner_lr
        elif params.weighting in ['MOML']:
            kwargs['weight_args']['inner_lr'] = params.inner_lr
    else:
        raise ValueError('No support weighting method {}'.format(params.weighting)) 
        
    if params.arch in ['HPS', 'Cross_stitch', 'MTAN', 'CGC', 'PLE', 'MMoE', 'DSelect_k', 'DIY', 'LTB']:
        if params.arch in ['CGC', 'PLE', 'MMoE', 'DSelect_k']:
            kwargs['arch_args']['img_size'] = tuple(params.img_size)#np.array(params.img_size, dtype=int).prod()
            kwargs['arch_args']['num_experts'] = [int(num) for num in params.num_experts]
        if params.arch in ['DSelect_k']:
            kwargs['arch_args']['kgamma'] = params.kgamma
            kwargs['arch_args']['num_nonzeros'] = params.num_nonzeros
    else:
        raise ValueError('No support architecture method {}'.format(params.arch)) 
        
    if params.optim in ['adam', 'sgd', 'adagrad', 'rmsprop']:
        if params.optim == 'adam':
            optim_param = {'optim': 'adam', 'lr': params.lr, 'weight_decay': params.weight_decay}
        elif params.optim == 'sgd':
            optim_param = {'optim': 'sgd', 'lr': params.lr, 
                           'weight_decay': params.weight_decay, 'momentum': params.momentum}
    else:
        raise ValueError('No support optim method {}'.format(params.optim))
        
    if params.scheduler is not None:
        if params.scheduler in ['step', 'cos', 'exp']:
            if params.scheduler == 'step':
                scheduler_param = {'scheduler': 'step', 'step_size': params.step_size, 'gamma': params.gamma}
        else:
            raise ValueError('No support scheduler method {}'.format(params.scheduler))
    else:
        scheduler_param = None
    
    _display(params, kwargs, optim_param, scheduler_param)
    
    return kwargs, optim_param, scheduler_param

def _display(params, kwargs, optim_param, scheduler_param):
    print('='*40)
    print('General Configuration:')
    print('\tMode:', params.mode)
    print('\tWighting:', params.weighting)
    print('\tArchitecture:', params.arch)
    print('\tRep_Grad:', params.rep_grad)
    print('\tMulti_Input:', params.multi_input)
    print('\tSeed:', params.seed)
    print('\tSave Path:', params.save_path)
    print('\tLoad Path:', params.load_path)
    print('\tDevice: {}'.format('cuda:'+params.gpu_id if torch.cuda.is_available() else 'cpu'))
    for wa, p in zip(['weight_args', 'arch_args'], [params.weighting, params.arch]):
        if kwargs[wa] != {}:
            print('{} Configuration:'.format(p))
            for k, v in kwargs[wa].items():
                print('\t'+k+':', v)
    print('Optimizer Configuration:')
    for k, v in optim_param.items():
        print('\t'+k+':', v)
    if scheduler_param is not None:
        print('Scheduler Configuration:')
        for k, v in scheduler_param.items():
            print('\t'+k+':', v)