import torch, os
import torchvision.transforms as transforms
from PIL import Image

from LibMTL.dataset import AbsDataset

class OfficeDataset(AbsDataset):
    def __init__(self, 
                 path: str, 
                 split_path: str, 
                 current_task: str,
                 mode: str, 
                 augmentation: bool = False):
        self.split_path = split_path
        super(OfficeDataset, self).__init__(path=path, 
                                            augmentation=augmentation,
                                            current_task=current_task,
                                            mode=mode)

        self.transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                        ])

    def _prepare_list(self):
        with open(os.path.join(self.split_path, '{}_{}.txt'.format(self.current_task, self.mode)), 'r') as f:
            index_list = f.readlines()
        return index_list

    def _get_data_labels(self, idx):
        img_path = self.index_list[idx][:-1].split(' ')[0]
        label = int(self.index_list[idx][:-1].split(' ')[1])
        img = Image.open(os.path.join(self.path, img_path)).convert('RGB')
        return self.transform(img), label