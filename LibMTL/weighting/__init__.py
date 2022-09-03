from .abstract_weighting import AbsWeighting
from .EW import EW
from .GradNorm import GradNorm
from .MGDA import MGDA
from .UW import UW 
from .DWA import DWA
from .GLS import GLS
from .GradDrop import GradDrop
from .PCGrad import PCGrad
from .GradVac import GradVac
from .IMTL import IMTL
from .CAGrad import CAGrad
from .Nash_MTL import Nash_MTL
from .RLW import RLW

__all__ = ['AbsWeighting',
           'EW', 
           'GradNorm', 
           'MGDA',
           'UW',
           'DWA',
           'GLS',
           'GradDrop',
           'PCGrad',
           'GradVac',
           'IMTL',
           'CAGrad',
           'Nash_MTL',
           'RLW']