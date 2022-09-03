from LibMTL.utils.saver import ExpSaver
from LibMTL.utils.config import load_cfg
from LibMTL.utils.logger import get_root_logger
from LibMTL.utils.env import collect_env
from LibMTL.utils import set_random_seed, set_device
from pprint import pformat

def common_init(cfg_file):
    cfg = load_cfg(cfg_file)
    device = set_device(cfg['general']['gpu_id'])
    set_random_seed(cfg['general']['seed'])
    saver = ExpSaver(cfg)
    logger = get_root_logger()
    # logger.info('Config info:\n' + pformat(cfg) + '\n' + '='*60)
    # logger.info('System info:\n' + pformat(collect_env()) + '\n' + '='*60)
    return cfg, saver, device