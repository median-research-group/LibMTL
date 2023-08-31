
from .dataset import *
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
class RandomMixup(torch.nn.Module):
    def __init__(self, num_classes: int = 2, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
 
        if alpha <= 0:
            raise ValueError("Alpha param can't be zero.")
 
        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace
 
    def forward(self, batch, target):
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")
 
        if not self.inplace:
            batch = batch.clone()
            target = target.clone()
 
        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)
 
        if torch.rand(1).item() >= self.p:
            return batch, target
 
        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)
 
        # Implemented as on mixup paper, page 3.
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        batch_rolled.mul_(1.0 - lambda_param)
        batch.mul_(lambda_param).add_(batch_rolled)
 
        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)
 
        return batch, target
 

def get_test_loader(dataset_name, batch_size, num_workers, input_size, compression, subset=None, args=None):
    if dataset_name=='cdf2':
        dataset = CDFv2(input_size)
    if dataset_name=='dfdc':
        dataset = DFDC(input_size)
    if dataset_name=='faceshifter':
        dataset = FaceShifter(input_size, compression=compression)
    if dataset_name=='ffpp':
        dataset = FFpp_test(input_size, compression=compression, subset=subset)
    if dataset_name=='deeperforensics':
        dataset = Deeper(input_size)

    data_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=False,
                            collate_fn=collate_fn_test,
                                    drop_last=True)
    return data_loader

def get_train_loader(batch_size, num_workers, input_size, mode, dist=False, args=None):
    dataset = FFpp_train(input_size, aug=args.aug)
    if dist:
        sampler = DistributedSampler(
                dataset, num_replicas=args.world_size, rank=args.gpu, shuffle=True)
        data_loader = DataLoader(dataset=dataset,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    sampler=sampler,
                                    pin_memory=True,
                                    drop_last=True)
    else:
        data_loader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=True)

    return data_loader

def get_sbi_loader(batch_size, num_workers, input_size, mode, dist=False, args=None):
    dataset = FFsbi(input_size, None, args.compression)
    if dist:
        sampler = DistributedSampler(
                dataset, num_replicas=args.world_size, rank=args.gpu, shuffle=True)
        data_loader = DataLoader(dataset=dataset,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    sampler=sampler,
                                    pin_memory=True,
                                    drop_last=True,
                                    collate_fn=dataset.collate_fn,
					                worker_init_fn=dataset.worker_init_fn)
    else:
        data_loader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=True,
                                collate_fn=dataset.collate_fn,
					                worker_init_fn=dataset.worker_init_fn)

    return data_loader

def get_vfhq_sbi_loader(batch_size, num_workers, input_size, mode, dist=False, args=None):
    dataset = VFHQ_train(input_size, None)
    if dist:
        sampler = DistributedSampler(
                dataset, num_replicas=args.world_size, rank=args.gpu, shuffle=True)
        data_loader = DataLoader(dataset=dataset,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    sampler=sampler,
                                    pin_memory=True,
                                    drop_last=True,
                                    collate_fn=dataset.collate_fn,
					                worker_init_fn=dataset.worker_init_fn)
    else:
        data_loader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                collate_fn=dataset.collate_fn,
					                worker_init_fn=dataset.worker_init_fn,
                                shuffle=True)

    return data_loader

def get_vfhq_inpaint_loader(batch_size, num_workers, input_size, mode, dist=False, args=None):
    dataset = VFHQInpaint(input_size)
    if dist:
        sampler = DistributedSampler(
                dataset, num_replicas=args.world_size, rank=args.gpu, shuffle=True)
        data_loader = DataLoader(dataset=dataset,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    sampler=sampler,
                                    pin_memory=True,
                                    drop_last=True,
                                    collate_fn=dataset.collate_fn,
					                worker_init_fn=dataset.worker_init_fn)
    else:
        data_loader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                collate_fn=dataset.collate_fn,
                                 worker_init_fn=dataset.worker_init_fn,
                                shuffle=True)
    return data_loader


def get_dfmnist_sbi_loader(batch_size, num_workers, input_size, mode, dist=False, args=None):
    dataset = DFMnistsbi_train(input_size, None)
    if dist:
        sampler = DistributedSampler(
                dataset, num_replicas=args.world_size, rank=args.gpu, shuffle=True)
        data_loader = DataLoader(dataset=dataset,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    sampler=sampler,
                                    pin_memory=True,
                                    drop_last=True,
                                    collate_fn=dataset.collate_fn,
					                worker_init_fn=dataset.worker_init_fn)
    else:
        data_loader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                collate_fn=dataset.collate_fn,
					                worker_init_fn=dataset.worker_init_fn,
                                shuffle=True)

    return data_loader

def get_dfmnist_inpaint_loader(batch_size, num_workers, input_size, mode, dist=False, args=None):
    dataset = DFMnistInpaint(input_size)
    if dist:
        sampler = DistributedSampler(
                dataset, num_replicas=args.world_size, rank=args.gpu, shuffle=True)
        data_loader = DataLoader(dataset=dataset,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    sampler=sampler,
                                    pin_memory=True,
                                    drop_last=True,
                                    collate_fn=dataset.collate_fn,
					                worker_init_fn=dataset.worker_init_fn)
    else:
        data_loader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                collate_fn=dataset.collate_fn,
                                 worker_init_fn=dataset.worker_init_fn,
                                shuffle=True)
    return data_loader

def get_tri_loader(batch_size, num_workers, input_size, mode, dist=False, args=None, phase='train'):
    dataset = FFtri(input_size, phase=phase)
    if dist:
        sampler = DistributedSampler(
                dataset, num_replicas=args.world_size, rank=args.gpu, shuffle=True)
        data_loader = DataLoader(dataset=dataset,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    sampler=sampler,
                                    pin_memory=True,
                                    drop_last=True,
                                    collate_fn=dataset.collate_fn,
					                worker_init_fn=dataset.worker_init_fn)
    else:
        data_loader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                collate_fn=dataset.collate_fn,
                                shuffle=True)

    return data_loader

# def get_ffhq_sbi_loader(batch_size, num_workers, input_size, mode, dist=False, args=None):
#     dataset = FFHQsbi(input_size)
#     if dist:
#         sampler = DistributedSampler(
#                 dataset, num_replicas=args.world_size, rank=args.gpu, shuffle=True)
#         data_loader = DataLoader(dataset=dataset,
#                                     batch_size=batch_size,
#                                     num_workers=0,
#                                     sampler=sampler,
#                                     pin_memory=True,
#                                     drop_last=True,
#                                     collate_fn=dataset.collate_fn,
# 					                worker_init_fn=dataset.worker_init_fn)
#     else:
#         data_loader = DataLoader(dataset=dataset,
#                                 batch_size=batch_size,
#                                 num_workers=num_workers,
#                                 shuffle=True)

#     return data_loader

if __name__=="__main__":
    import time
    import datetime
    loader = get_train_loader(256,8,224,'train')
    data_iter = iter(loader)
    cnt = 0
    start_time = time.time()
    for i in range(100):
        clips, labels = next(data_iter) 
    elapsed = str(datetime.timedelta(seconds=time.time() - start_time))
    print(elapsed)