import os
from PIL import Image
from torch.utils.data import Dataset

class TinyImageNet(Dataset):
    def __init__(self, root, split='train', transform=None, return_index=False):

        self.root = root
        self.split = split
        self.transform = transform
        self.return_index = return_index

        self.samples = []
        self.class_to_idx = {}

        if split == 'train':
            self._load_train()
        elif split == 'val':
            self._load_val()
        else:
            raise ValueError("split must be 'train' or 'val'")

    def _load_train(self):
        train_dir = os.path.join(self.root, 'train')
        wnids = sorted(os.listdir(train_dir))
        self.class_to_idx = {wnid: i for i, wnid in enumerate(wnids)}

        for wnid in wnids:
            img_dir = os.path.join(train_dir, wnid, 'images')
            for fname in os.listdir(img_dir):
                if fname.endswith('.JPEG'):
                    path = os.path.join(img_dir, fname)
                    label = self.class_to_idx[wnid]
                    self.samples.append((path, label))

    def _load_val(self):
        val_dir = os.path.join(self.root, 'val')
        ann_path = os.path.join(val_dir, 'val_annotations.txt')

        wnids = sorted(os.listdir(os.path.join(self.root, 'train')))
        self.class_to_idx = {wnid: i for i, wnid in enumerate(wnids)}

        img_to_wnid = {}
        with open(ann_path, 'r') as f:
            for line in f:
                img, wnid = line.split('\t')[:2]
                img_to_wnid[img] = wnid

        img_dir = os.path.join(val_dir, 'images')
        for img, wnid in img_to_wnid.items():
            path = os.path.join(img_dir, img)
            label = self.class_to_idx[wnid]
            self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        if self.return_index:
            return img, [label, index]
        else:
            return img, label
