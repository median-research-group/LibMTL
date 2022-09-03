from .abstract_loss import AbsLoss
from .CELoss import CELoss
from .DepthLoss import DepthLoss
from .KLDivLoss import KLDivLoss
from .L1Loss import L1Loss
from .MSELoss import MSELoss
from .NormalLoss import NormalLoss
from .SegLoss import SegLoss

__all__ = ['AbsLoss',
           'CELoss',
           'DepthLoss',
           'KLDivLoss',
           'L1Loss',
           'MSELoss',
           'NormalLoss',
           'SegLoss']