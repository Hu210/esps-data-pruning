from scipy.stats import beta
import numpy as np
import time
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from multiprocessing import Pool
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import beta
from datasets_tiny import TinyImageNet

########################################################################################################################
#  Load Data
########################################################################################################################

def load_cifar10_sub(args, target_probs=None, score=None, data_mask=None):
    """
    Load CIFAR10 dataset with specified transformations and subset selection.
    """
    print('Loading CIFAR10... ', end='')
    
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    train_data = dset.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)

    if args.sample == 'random':
        subset_mask = np.random.choice(50000, int(args.subset_rate * 50000), replace=False)
    elif args.sample == 'stratified':
        mis_num = int(args.mis_ratio * 50000)
        easy_index = data_mask[mis_num:]
        selected = stratified_sampling(torch.tensor(score[easy_index]), int(args.subset_rate * 50000))
        subset_mask = easy_index[selected]

    elif args.sample == 'beta':
        subset_mask = beta_sampling(1-args.subset_rate, args.c_d, target_probs, data_mask, score)
    
    elif args.sample == 'balance':
        N = len(data_mask)
        k = int(args.subset_rate * N)

        start = 0
        end = int(k+(1.8-args.subset_rate)*0.1*N)

        end = max(end, k)
        end = min(end, N)

        if args.keep == 'low':
            pool = data_mask[start:end]   
            print(f"[Balance Sampling with low-score] N={N}, k={k}, start={start}, end={end}, pool={len(pool)}")
        else:
            rev = data_mask[::-1] 
            pool = rev[start:end]
            print(f"[Balance Sampling with high-score] N={N}, k={k}, start={start}, end={end}, pool={len(pool)}")

        balance_sampling = BalanceSampling(pool_indices=pool, k=k)

        train_loader = torch.utils.data.DataLoader(
            train_data,                   
            batch_size=args.batch_size,
            sampler=balance_sampling,           
            shuffle=False,                
            num_workers=args.workers,
            pin_memory=True
        )

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True
        )

        return train_loader, test_loader    

    else:
        k = int(args.subset_rate * len(data_mask))
        keep = getattr(args, 'keep', 'high')
        if keep == 'low':
            subset_mask = data_mask[:k]
            print(f"Using low-score samples, k={k}")
        else:
            subset_mask = data_mask[-k:]
            print(f"Using high-score samples, k={k}")
    
    data_set = torch.utils.data.Subset(train_data, subset_mask)
    train_loader = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)
    return train_loader, test_loader



def load_cifar100_sub(args, target_probs=None, score=None, data_mask=None):
    """
    Load CIFAR100 dataset with specified transformations and subset selection.
    """
    print('Loading CIFAR100... ', end='')
    
    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std = [x / 255 for x in [68.2, 65.4, 70.4]]

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    train_data = dset.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
    
    if args.sample == 'random':
        subset_mask = np.random.choice(50000, int(args.subset_rate * 50000), replace=False)
    elif args.sample == 'stratified':
        mis_num = int(args.mis_ratio * 50000)
        easy_index = data_mask[mis_num:]
        selected = stratified_sampling(torch.tensor(score[easy_index]), int(args.subset_rate * 50000))
        subset_mask = easy_index[selected]
    elif args.sample == 'beta':
        subset_mask = beta_sampling(1-args.subset_rate, args.c_d, target_probs, data_mask, score)
    
    elif args.sample == 'balance':
        N = len(data_mask)
        k = int(args.subset_rate * N)

        start = 0
        end = int(k+(1.8-args.subset_rate)*0.1*N)

        end = max(end, k)
        end = min(end, N)

        if args.keep == 'low':
            pool = data_mask[start:end]
            print(f"[Balance Sampling with low-score] N={N}, k={k}, start={start}, end={end}, pool={len(pool)}")
        else:
            rev = data_mask[::-1]
            pool = rev[start:end]
            print(f"[Balance Sampling with high-score] N={N}, k={k}, start={start}, end={end}, pool={len(pool)}")

        balance_sampling = BalanceSampling(pool_indices=pool, k=k)

        train_loader = torch.utils.data.DataLoader(
            train_data,                   
            batch_size=args.batch_size,
            sampler=balance_sampling,           
            shuffle=False,                
            num_workers=args.workers,
            pin_memory=True
        )

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True
        )

        return train_loader, test_loader

    else:
        k = int(args.subset_rate * len(data_mask))
        keep = getattr(args, 'keep', 'high')
        if keep == 'low':
            subset_mask = data_mask[:k]
            print(f"Using low-score samples, k={k}")
        else:
            subset_mask = data_mask[-k:]
            print(f"Using high-score samples, k={k}")

    data_set = torch.utils.data.Subset(train_data, subset_mask)
    train_loader = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)
    return train_loader, test_loader

def load_tiny_sub(args, target_probs=None, score=None, data_mask=None):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(64, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_data = TinyImageNet(
        root=args.data_path,
        split='train',
        transform=train_transform,
        return_index=False
    )

    N = len(train_data)

    if args.sample == 'random':
        subset_mask = np.random.choice(N, int(args.subset_rate * N), replace=False)

    elif args.sample == 'beta':
        subset_mask = beta_sampling(
            1 - args.subset_rate,
            args.c_d,
            target_probs,
            data_mask,
            score
        )

    elif args.sample == 'balance':
        N = len(data_mask)
        k = int(args.subset_rate * N)

        start = 0
        end = int(k+(1.8-args.subset_rate)*0.1*N)

        end = max(end, k)
        end = min(end, N)

        if args.keep == 'low':
            pool = data_mask[start:end]
            print(f"[Balance Sampling with low-score] N={N}, k={k}, start={start}, end={end}, pool={len(pool)}")
        else:
            rev = data_mask[::-1]
            pool = rev[start:end]
            print(f"[Balance Sampling with high-score] N={N}, k={k}, start={start}, end={end}, pool={len(pool)}")

        balance_sampling = BalanceSampling(pool_indices=pool, k=k)


        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.batch_size,
            sampler=balance_sampling,
            shuffle=False,                  
            num_workers=args.workers,
            pin_memory=True
        )

        test_data = TinyImageNet(
            root=args.data_path,
            split='val',
            transform=test_transform,
            return_index=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True
        )

        return train_loader, test_loader
        
    else:
        k = int(args.subset_rate * len(data_mask))
        keep = getattr(args, 'keep', 'high')
        if keep == 'low':
            subset_mask = data_mask[:k]
            print(f"Using low-score samples, k={k}")
        else:
            subset_mask = data_mask[-k:]
            print(f"Using high-score samples, k={k}")

    subset_data = torch.utils.data.Subset(train_data, subset_mask)

    train_loader = torch.utils.data.DataLoader(
        subset_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )

    test_data = TinyImageNet(
        root=args.data_path,
        split='val',
        transform=test_transform,
        return_index=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    return train_loader, test_loader

def stratified_sampling(score, coreset_num):
    stratas = 50
    print('Using stratified sampling...')
    total_num = coreset_num

    min_score = torch.min(score)
    max_score = torch.max(score) * 1.0001
    step = (max_score - min_score) / stratas

    def bin_range(k):
        return min_score + k * step, min_score + (k + 1) * step

    strata_num = []
    ##### calculate number for each strata #####
    for i in range(stratas):
        start, end = bin_range(i)
        num = torch.logical_and(score >= start, score < end).sum()
        strata_num.append(num)

    strata_num = torch.tensor(strata_num)

    def bin_allocate(num, bins):
        sorted_index = torch.argsort(bins)
        sort_bins = bins[sorted_index]

        num_bin = bins.shape[0]

        rest_exp_num = num
        budgets = []
        for i in range(num_bin):
            rest_bins = num_bin - i
            avg = rest_exp_num // rest_bins
            cur_num = min(sort_bins[i].item(), avg)
            budgets.append(cur_num)
            rest_exp_num -= cur_num


        rst = torch.zeros((num_bin,)).type(torch.int)
        rst[sorted_index] = torch.tensor(budgets).type(torch.int)

        return rst

    budgets = bin_allocate(total_num, strata_num)

    selected_index = []
    sample_index = torch.arange(score.shape[0])

    for i in range(stratas):
        start, end = bin_range(i)
        mask = torch.logical_and(score >= start, score < end)
        pool = sample_index[mask]
        rand_index = torch.randperm(pool.shape[0])
        selected_index += [idx.item() for idx in pool[rand_index][:budgets[i]]]

    return selected_index

def beta_sampling(prune_rate, c_d, target_probs, mask, score):
    data_length = target_probs.shape[-1]
    target_probs = torch.tensor(target_probs)[:30] ### Use only first 30 epochs
    pred_mean = target_probs.mean(axis=0)

    subset_n = int((1-prune_rate) * data_length)
    anchor_mean = pred_mean[mask[-10:]].mean()
    y_b = 15 * (1-anchor_mean) * (1 -prune_rate ** c_d)
    y_a = 16 - y_b
    
    pdf_y = beta.pdf(pred_mean, y_a, y_b)
    joint_p = pdf_y * score
    remain_id = np.random.choice(data_length, p=joint_p/joint_p.sum(), size=subset_n, replace=False)
    return remain_id

class BalanceSampling(torch.utils.data.Sampler):

    def __init__(self, pool_indices, k):
        self.pool_indices = np.asarray(pool_indices, dtype=np.int64)
        self.k = int(k)
        if len(self.pool_indices) < self.k:
            raise ValueError(f"Balance Sampling: pool size {len(self.pool_indices)} < k {self.k}")

    def __iter__(self):
        chosen = np.random.choice(self.pool_indices, size=self.k, replace=False)
        return iter(chosen.tolist())

    def __len__(self):
        return self.k

