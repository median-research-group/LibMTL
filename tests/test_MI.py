from LibMTL.core import Trainer
import shutil, os, pytest

def template(yaml_path='./MI.yaml', w_dict=None, arch_dict=None, raises=False):
    model = Trainer(yaml_path)
    if w_dict is not None:
        model.cfg['weighting'] = w_dict
    if arch_dict is not None:
        model.cfg['architecture'] = arch_dict
    if not raises:
        model._prepare_model()
        model.train()
    else:
        with pytest.raises(ValueError):
            model._prepare_model()
            model.train()   
    shutil.rmtree(os.path.join(model.saver.save_dir, model.saver.exp_name))

# weighting
def test_EW():
    template(w_dict={'name': 'EW', 'rep_grad': False})

def test_MGDA():
    for mgda_gn in ['none', 'l2', 'loss', 'loss+']:
        for rep_grad in [True, False]:
            template(w_dict={'name': 'MGDA', 'mgda_gn': mgda_gn, 'rep_grad': rep_grad})

def test_CAGrad():
    template(w_dict={'name': 'CAGrad', 'alpha': 0.5, 'rescale': 1, 'rep_grad': False})
    template(w_dict={'name': 'CAGrad', 'alpha': 0.5, 'rescale': 1, 'rep_grad': True}, raises=True)

def test_DWA():
    template(w_dict={'name': 'DWA', 'T': 2, 'rep_grad': False})

def test_GLS():
    template(w_dict={'name': 'GLS', 'rep_grad': False})

def test_GradDrop():
    template(w_dict={'name': 'GradDrop', 'leak': 0, 'rep_grad': True})
    template(w_dict={'name': 'GradDrop', 'leak': 0, 'rep_grad': False}, raises=True)

def test_GradNorm():
    template(w_dict={'name': 'GradNorm', 'alpha': 1.5, 'rep_grad': True})
    template(w_dict={'name': 'GradNorm', 'alpha': 1.5, 'rep_grad': False})

def test_GradVac():
    template(w_dict={'name': 'GradVac', 'beta': 0.5, 'rep_grad': False})
    template(w_dict={'name': 'GradVac', 'beta': 0.5, 'rep_grad': True}, raises=True)

def test_IMTL():
    template(w_dict={'name': 'IMTL', 'rep_grad': True})
    template(w_dict={'name': 'IMTL', 'rep_grad': False})

def test_PCGrad():
    template(w_dict={'name': 'PCGrad', 'rep_grad': False})
    template(w_dict={'name': 'PCGrad', 'rep_grad': True}, raises=True)

def test_RLW():
    template(w_dict={'name': 'RLW', 'rep_grad': False})

def test_UW():
    template(w_dict={'name': 'UW', 'rep_grad': False})

# arch
def test_CGC():
    template(arch_dict={'name': 'CGC', 'num_experts': [2, 2, 2, 2], 'img_size': [3, 224, 224]})

def test_MMoE():
    template(arch_dict={'name': 'MMoE', 'num_experts': [2], 'img_size': [3, 224, 224]})

def test_Cross_stitch():
    template(arch_dict={'name': 'Cross_stitch'}, raises=True)

def test_DSelect_k():
    template(arch_dict={'name': 'DSelect_k', 'num_experts': [2], 
                                              'img_size': [3, 224, 224], 'num_nonzeros': 1, 'kgamma': 1.0})

def test_MTAN():
    template(arch_dict={'name': 'MTAN'})

# def test_PLE():
#     template(arch_dict={'name': 'PLE', 'num_experts': [1, 1, 1, 1], 'img_size': [3, 224, 224]})
