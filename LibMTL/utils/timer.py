import torch
from time import time

class TimeRecorder(object):
    """docstring for TimeRecorder"""
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self.elapse = 0
        self.begin_time = None
        self.end_time = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.use_gpu:
            self.end_time.record()
            torch.cuda.synchronize()
            self.elapse = self.begin_time.elapsed_time(self.end_time) / 1000
        else:
            self.elapse = time() - self.begin_time
        self.elapse = round(self.elapse, 3)

    def start(self):
        if self.use_gpu:
            torch.cuda.synchronize()
            self.begin_time = torch.cuda.Event(enable_timing=True)
            self.end_time = torch.cuda.Event(enable_timing=True)
            self.begin_time.record()
        else:
            self.begin_time = time()
