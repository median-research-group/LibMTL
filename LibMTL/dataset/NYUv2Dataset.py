import os, torch, fnmatch, random
import numpy as np
import torch.nn.functional as F

from LibMTL.dataset import AbsDataset

class NYUv2Dataset(AbsDataset):
    def __init__(self, 
                 path: str, 
                 task_name: list = ['segmentation', 'depth', 'normal'], 
                 augmentation: bool = False):
        super(NYUv2Dataset, self).__init__(path=path, 
                                        task_name=task_name, 
                                        augmentation=augmentation)

    def _prepare_list(self):
        data_len = len(fnmatch.filter(os.listdir(os.path.join(self.path, 'image')), '*.npy'))
        return list(range(data_len))

    def _get_data_labels(self, idx):
        index = self.index_list[idx]
        image = np.moveaxis(np.load(os.path.join(self.path, 'image', '{:d}.npy'.format(index))), -1, 0)
        image = torch.from_numpy(image).float()
        labels = {}
        for task in self.task_name:
            if task == 'segmentation':
                label = np.load(os.path.join(self.path, 'label', '{:d}.npy'.format(index)))
            else:
                label = np.moveaxis(np.load(os.path.join(self.path, task, '{:d}.npy'.format(index))), -1, 0)
            labels[task] = torch.from_numpy(label).float()
        return image, labels

    def _aug_data_labels(self, img, labels):
        height, width = img.shape[-2:]
        scale = [1.0, 1.2, 1.5]
        sc = scale[random.randint(0, len(scale) - 1)]
        h, w = int(height / sc), int(width / sc)
        i = random.randint(0, height - h)
        j = random.randint(0, width - w)

        filp = torch.rand(1) < 0.5

        img = F.interpolate(img[None, :, i:i + h, j:j + w], 
                        size=(height, width), mode='bilinear', align_corners=True).squeeze(0)
        img = torch.flip(img, dims=[2]) if filp else img
        if 'segmentation' in self.task_name:
            label_ = F.interpolate(labels['segmentation'][None, None, i:i + h, j:j + w], 
                                size=(height, width), mode='nearest').squeeze(0).squeeze(0)
            label_ = torch.flip(label_, dims=[1]) if filp else label_
            labels['segmentation'] = label_.float()
        if 'depth' in self.task_name:
            depth_ = F.interpolate(labels['depth'][None, :, i:i + h, j:j + w], 
                                size=(height, width), mode='nearest').squeeze(0)
            depth_ = torch.flip(depth_, dims=[2]) if filp else depth_
            labels['depth'] = (depth_ / sc).float()
        if 'normal' in self.task_name:
            normal_ = F.interpolate(labels['normal'][None, :, i:i + h, j:j + w], 
                                size=(height, width), mode='bilinear', align_corners=True).squeeze(0)
            if filp:
                normal_ = torch.flip(normal_, dims=[2])
                normal_[0, :, :] = - normal_[0, :, :]
            labels['normal'] = normal_.float()
        return img.float(), labels
