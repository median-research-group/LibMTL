from LibMTL.weighting.abstract_weighting import AbsWeighting
from LibMTL.weighting.EW import EW
from LibMTL.weighting.GradNorm import GradNorm
from LibMTL.weighting.MGDA import MGDA
from LibMTL.weighting.UW import UW 
from LibMTL.weighting.DWA import DWA
from LibMTL.weighting.GLS import GLS
from LibMTL.weighting.GradDrop import GradDrop
from LibMTL.weighting.PCGrad import PCGrad
from LibMTL.weighting.GradVac import GradVac
from LibMTL.weighting.IMTL import IMTL
from LibMTL.weighting.CAGrad import CAGrad
from LibMTL.weighting.Nash_MTL import Nash_MTL
from LibMTL.weighting.RLW import RLW
from LibMTL.weighting.MoCo import MoCo
from LibMTL.weighting.Aligned_MTL import Aligned_MTL

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
           'RLW',
           'MoCo',
           'Aligned_MTL']