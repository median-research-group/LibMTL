import torch, os
from typing import Optional
from torch.utils.data.dataset import Dataset

class AbsDataset(Dataset):
    def __init__(self, 
                 path: str, 
                 augmentation: bool,
                 task_name: Optional[list] = None,
                 current_task: Optional[str] = None,
                 mode: Optional[str] = None):
        super(AbsDataset, self).__init__()

        self.path = os.path.expanduser(path)
        self.task_name = task_name
        self.current_task = current_task
        self.mode = mode
        self.augmentation = augmentation

        self.index_list = self._prepare_list()

    def _prepare_list(self) -> list:
        return []

    def _get_data_labels(self, idx):
        pass

    def _aug_data_labels(self, data, labels):
        pass

    def __getitem__(self, idx: int):
        data, labels = self._get_data_labels(idx)
        if self.augmentation:
            data, labels = self._aug_data_labels(data, labels)
        return data, labels

    def __len__(self):
        return len(self.index_list)