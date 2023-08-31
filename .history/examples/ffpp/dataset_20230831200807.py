
import os
import pickle
from glob import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import json
from PIL import Image


def collate_fn_test(batch):
    # img_f,img_r, mask=zip(*batch)
    image, label, vid_id=zip(*batch)
    data={}
    data['img']=torch.tensor(image).float()
    # data['mask']=torch.tensor(mask).float()
    data['label']=torch.tensor(label).long()
    data['vid_id']=torch.tensor(vid_id).long()
    return data['img'], data['label'], data['vid_id']

class FFpp(Dataset):
    def __init__(self, input_size, compression='c23', subset=None):
        self.transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                        ])
        super().__init__()
        self.data_root = '/data5/tjh/standard_crop/FF++'
        self.compression = compression
        # self.transform = create_transforms(input_size, aug='base', split='test')
        self.input_size = input_size
        self.subset = subset
        self._make_dataset()
        self.all_images = self.all_images[::10]
        self.all_labels = self.all_labels[::10]
        self.all_vids = self.all_vid_id[::10]

    def __getitem__(self, index):
        image = self.all_images[index]
        image = self._load_image(image)
        #TODO: define transformer
        # image = self.transform(image=image)['image']
        image=cv2.resize(image,(self.input_size,self.input_size),interpolation=cv2.INTER_LINEAR).astype('float32')/255
        image = image.transpose((2,0,1))
        label = self.all_labels[index]
        vid_id = self.all_vid_id[index]
        return image, label, vid_id


    def __len__(self):
        return len(self.all_images)
    
    def _load_image(self, path):
        # return np.asarray(Image.open(path).convert('RGB'), dtype=np.uint8)
        return cv2.imread(path)[:,:,::-1]
    
    def _make_dataset(self):
        self.all_images = []
        self.all_labels = []
        self.all_vid_id = []

        meta_data_file = '/data3/chenpeng/data/FaceForensics++/splits/train.json'
        with open(meta_data_file) as f:
            meta_data = json.load(f)
        from functools import reduce
        meta_data = reduce(lambda x,y:x+y, meta_data)

        if not self.subset:
            types = ['original', 'Deepfakes', 'Face2Face', 'NeuralTextures', 'FaceSwap']
        else:
            assert isinstance(self.subset, str)
            types = ['original'] + [i for i in self.subset.split('_')]
        vid_id = 0
        for label, t in enumerate(types):
            t_dir = os.path.join(self.data_root, self.compression, t)
            for vid in os.listdir(t_dir):

                if vid[:3] not in meta_data:
                    continue

                all_frames = glob(os.path.join(self.data_root, self.compression, t, vid)+'/*.png')
                if label==0:
                    self.all_vid_id.extend([vid_id]*(len(types)-1)*len(list(all_frames)))
                    self.all_images.extend((len(types)-1)*list(all_frames))
                    self.all_labels.extend((len(types)-1)*len(list(all_frames))*[label])
                else:
                    self.all_vid_id.extend([vid_id]*len(list(all_frames)))
                    self.all_images.extend(list(all_frames))
                    self.all_labels.extend(len(list(all_frames))*[label])
                vid_id+=1

class FaceShifter(Dataset):
    def __init__(self, input_size, compression='c23'):
        super().__init__()
        self.data_root = '/data1/jiahe.tian/standard_crop/FF++'
        self.compression = compression
        # self.transform = create_transforms(input_size,aug='base',split='test')
        self.input_size = input_size
        self._make_dataset()
    
    def __getitem__(self, index):
        image = self.all_images[index]
        image = self._load_image(image)
        # image = self.transform(image=image)['image']
        image=cv2.resize(image,(self.input_size,self.input_size),interpolation=cv2.INTER_LINEAR).astype('float32')/255
        image = image.transpose((2,0,1))
        label = self.all_labels[index]
        vid_id = self.all_vid_id[index]
        return image, label, vid_id

    def __len__(self):
        return len(self.all_images)
    def _load_image(self, path):
        # print(path)
        # return np.asarray(Image.open(path).convert('RGB'), dtype=np.uint8)
        return cv2.imread(path)[:,:,::-1]
    def _make_dataset(self):
        self.all_images = []
        self.all_labels = []
        self.all_vid_id = []

        meta_data_file = self.data_root + '/splits/test.json'
        with open(meta_data_file) as f:
            meta_data = json.load(f)
        from functools import reduce
        meta_data = reduce(lambda x,y:x+y, meta_data)

        types = ['original', 'FaceShifter']

        vid_id = 0
        for label, t in enumerate(types):
            t_dir = os.path.join(self.data_root, t,'c23/videos')
            for vid in os.listdir(t_dir):
                if vid[:3] not in meta_data:
                    continue
                vid_pth = os.path.join(self.data_root,  t,'c23/videos',vid)
                all_frames = glob(vid_pth+'/*.png')
                self.all_images.extend(all_frames)
                self.all_labels.extend([label]*len(all_frames))
                self.all_vid_id.extend([vid_id]*len(all_frames))
                vid_id += 1

class Deeper(Dataset):
    def __init__(self, input_size, compression='c23'):
        super().__init__()
        self.data_root = '/data1/jiahe.tian/standard_crop/FF++'
        self.compression = compression
        self.transform = create_transforms(input_size,aug='base', split='test')
        self.input_size = input_size
        self._make_dataset()
    
    def __getitem__(self, index):
        image = self.all_images[index]
        image = self._load_image(image)
        image = self.transform(image=image)['image']
        
        label = self.all_labels[index]
        vid_id = self.all_vid_id[index]
        return image, label, vid_id

    def __len__(self):
        return len(self.all_images)
    def _load_image(self, path):
        return np.asarray(Image.open(path).convert('RGB'), dtype=np.uint8)
    def _make_dataset(self):
        self.all_images = []
        self.all_labels = []
        self.all_vid_id = []

        meta_data_file = self.data_root + '/splits/test.json'
        with open(meta_data_file) as f:
            meta_data = json.load(f)
        from functools import reduce
        meta_data = reduce(lambda x,y:x+y, meta_data)

        types = ['original', 'FaceShifter']

        vid_id = 0
        for label, t in enumerate(types):
            t_dir = os.path.join(self.data_root,t)
            for vid in os.listdir(t_dir):
                if vid[:3] not in meta_data:
                    continue
                vid_pth = os.path.join(self.data_root, t, vid)
                all_frames = glob(vid_pth+'/*.png')
                self.all_images.extend(all_frames)
                self.all_labels.extend([label]*len(all_frames))
                self.all_vid_id.extend([vid_id]*len(all_frames))
                vid_id += 1



class CDFv2(Dataset):
    def __init__(self, input_size):
        super().__init__()
        self.data_root = '/data5/tjh/standard_crop/Celeb-DFv2/'
        # self.transform = create_transforms(input_size,aug='base', split='test')
        self.input_size = input_size
        self._make_dataset()

    
    def __getitem__(self, index):
        image = self.all_images[index]
        image = self._load_image(image)
        # image = self.transform(image=image)['image']
        image=cv2.resize(image,(self.input_size,self.input_size),interpolation=cv2.INTER_LINEAR).astype('float32')/255
        image = image.transpose((2,0,1))
        label = self.all_labels[index]
        vid_id = self.all_vid_id[index]
        return image, label, vid_id

    def __len__(self):
        return len(self.all_images)
    def _load_image(self, path):
        # return np.asarray(Image.open(path).convert('RGB'), dtype=np.uint8)
        return cv2.imread(path)[:,:,::-1]
    def _make_dataset(self):
        self.all_images = []
        self.all_labels = []
        self.all_vid_id = []

        meta_file = '/data1/chenpeng/data/Celeb-DFv2/test.txt'
        vids = []
        with open(meta_file, 'r') as f:
            all_info= f.readlines()
        for i in all_info:
            l, vid = i.split(' ')
            vid = vid.split('/')[-1].replace('.mp4', '').replace('\n', '')
            vids.append(vid)

        vid_id = 0

        t_dir = os.path.join(self.data_root)
        for vid in os.listdir(t_dir):
            if vid not in vids:
                continue
            if vid.count('id')==2:
                label = 1
            elif vid.count('id')==1:
                label = 0
            else:
                continue

            vid_pth = os.path.join(self.data_root, vid)
            all_frames = glob(vid_pth+'/*.png')
            self.all_images.extend(all_frames)
            self.all_labels.extend([label]*len(all_frames))
            self.all_vid_id.extend([vid_id]*len(all_frames))
            vid_id += 1



class DFDC(Dataset):
    def __init__(self, input_size):
        super().__init__()
        self.data_root = '/data5/tjh/standard_crop/select_dfdc_test/videos'
        self.input_size = input_size
        self._make_dataset()
        self.all_images = self.all_images[::3]
        self.all_labels = self.all_labels[::3]
        self.all_vid_id = self.all_vid_id[::3]
    
    def __getitem__(self, index):
        image = self.all_images[index]
        image = self._load_image(image)
        image=cv2.resize(image,(self.input_size,self.input_size),interpolation=cv2.INTER_LINEAR).astype('float32')/255
        image = image.transpose((2,0,1))
        label = self.all_labels[index]
        vid_id = self.all_vid_id[index]
        return image, label, vid_id


    def __len__(self):
        return len(self.all_images)
    def _load_image(self, path):
        # return np.asarray(Image.open(path).convert('RGB'), dtype=np.uint8)
        return np.asarray(Image.open(path))
    def _make_dataset(self):
        self.all_images = []
        self.all_labels = []
        self.all_vid_id = []

        meta_data_file = '/data5/tjh/standard_crop/dfdc_test.txt'
        with open(meta_data_file) as f:
            meta = f.readlines()
        meta_data = {}
        for i in meta:
            k, v = i.strip().split(' ')
            meta_data[k] = int(v)

        vid_id = 0

        for vid in os.listdir(self.data_root):
            label = meta_data[vid]
            vid_pth = os.path.join(self.data_root, vid)
            all_frames = glob(vid_pth+'/*.png')
            self.all_images.extend(all_frames)
            self.all_labels.extend([label]*len(all_frames))
            self.all_vid_id.extend([vid_id]*len(all_frames))
            vid_id += 1
        print(len(self.all_images))
        print(len(self.all_labels))




if __name__ == "__main__":
    # dataset = FFpp_train(224, )
    # from torch.utils.data import DataLoader
    # data_loader = DataLoader(dataset=dataset, batch_size=2, num_workers=1)
    # for a in data_loader:
    #     print(a[0].shape, a[1].shape)
    #     break
        
    dataset = FFpp_train(224, )
    from torch.utils.data import DataLoader
    data_loader = DataLoader(dataset=dataset, batch_size=2, num_workers=1)
    for a in data_loader:
        print(a[0].shape, a[1].shape)
        break
    dataset = CDFv2(224, )
    from torch.utils.data import DataLoader
    data_loader = DataLoader(dataset=dataset, batch_size=2, num_workers=1)
    for a in data_loader:
        print(a[0].shape, a[1].shape)
        break


    dataset = DFDC(224, )
    from torch.utils.data import DataLoader
    data_loader = DataLoader(dataset=dataset, batch_size=2, num_workers=1)
    for a in data_loader:
        print(a[0].shape, a[1].shape)
        break


    # dataset = Deeper(224, )
    # from torch.utils.data import DataLoader
    # data_loader = DataLoader(dataset=dataset, batch_size=2, num_workers=1)
    # for a in data_loader:
    #     print(a[0].shape, a[1].shape)
    #     break
    # pass
