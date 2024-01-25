import os
import random
import torch
import torchvision
import numpy as np
from PIL import Image
import torch.utils.data as data_utils
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision import datasets

try:
    from imagecorruptions import corrupt
except ImportError:
    print("Import Error corruption")

SAVE_IMG = False

VALID_SPURIOUS = [
    'TINT',  # apply a fixed class-wise tinting (meant to not affect shape)
]

def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join(
        "~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt"
    )
    cache_path = os.path.expanduser(cache_path)
    return cache_path

def default_loader(path):
    return Image.open(path).convert('RGB')

def default_flist_reader(flist, sep):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split(sep)
            imlist.append((impath, int(imlabel)))

    return imlist

def path_flist_reader(root, flist, sep):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            items = line.strip().split(sep)
            impath, imlabel = items
            imlist.append((os.path.join(root, impath), int(imlabel)))

    return imlist

def pathstyle_flist_reader(root, flist, sep):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            items = line.strip().split(sep)
            impath, imlabel = items
            impath = os.path.splitext(impath)[0] + '.jpg'
            imlist.append((os.path.join(root, impath), int(imlabel)))

    return imlist

def subset_flist_reader(flist, sep, class_list):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split(sep)
            if int(imlabel) in class_list.keys():
                imlist.append((impath, int(imlabel)))

    return imlist

def style_flist_reader(root, flist, sep):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split(sep)

            if impath[0] == '.':
                impath = impath[impath.find('val'):]

            if os.path.exists(os.path.join(root, impath)):
                imlist.append((impath, int(imlabel)))

    return imlist

def folder_reader(data_dir):
    all_img_files = []
    all_labels = []

    class_names = os.walk(data_dir).__next__()[1]
    for index, class_name in enumerate(class_names):
        label = index
        img_dir = os.path.join(data_dir, class_name)
        img_files = os.walk(img_dir).__next__()[2]

        for img_file in img_files:
            img_file = os.path.join(img_dir, img_file)
            img = Image.open(img_file)
            if img is not None:
                all_img_files.append(img_file)
                all_labels.append(int(label))

    return all_img_files, all_labels

def subset_folder_reader(data_dir, flist):
    all_img_files = []
    all_labels = []

    with open(flist, 'r') as rf:
        for line in rf.readlines():
            imfolder, imlabel = line.strip().split(' ')
            class_name = imfolder
            label = imlabel
            img_dir = os.path.join(data_dir, class_name)
            img_files = os.walk(img_dir).__next__()[2]

            for img_file in img_files:
                img_file = os.path.join(img_dir, img_file)
                img = Image.open(img_file)
                if img is not None:
                    all_img_files.append(img_file)
                    all_labels.append(int(label))

    return all_img_files, all_labels

class ImageFilelist(torch.utils.data.Dataset):
    def __init__(self, root, flist=None, folderlist=None, subset_folderlist=None, transform=None, target_transform=None,
                 flist_reader=default_flist_reader, loader=default_loader, sep=' '):
        self.root = root
        self.imlist = []
        if flist:
            self.imlist = flist_reader(flist,sep)

        elif subset_folderlist:
            self.images, self.labels = subset_folder_reader(folderlist, subset_folderlist)
        else:
            self.images, self.labels = folder_reader(folderlist)

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):

        if self.imlist:
            impath, target = self.imlist[index]
            img = self.loader(os.path.join(self.root, impath))
        else:
            img = self.loader(self.images[index])
            target = self.labels[index]

        img_fp = img

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imlist) if self.imlist else len(self.images)

class ImageFilelist_Corrupt(torch.utils.data.Dataset):
    def __init__(self, root, train=True, flist=None, transform=None, target_transform=None,
                 flist_reader=subset_flist_reader, loader=default_loader, sep=' ', class_subset={}, sev=1):
        self.root = root
        self.train = train
        self.imlist = []
        self.imlist = flist_reader(flist, sep, class_subset)
        self.class_subset = class_subset

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.corrupt_list = ['brightness', 'contrast', 'gaussian_noise', 'frost', 'elastic_transform',
                        'gaussian_blur', 'defocus_blur', 'impulse_noise', 'saturate', 'pixelate']
        self.sev = sev

    def __getitem__(self, index):

        impath, target = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        target = self.class_subset[target]

        if self.train:
            img = corrupt(np.asarray(img), corruption_name=self.corrupt_list[target], severity=self.sev)
        else:
            rand_target = random.randint(0, 9)
            img = corrupt(np.asarray(img), corruption_name=self.corrupt_list[rand_target], severity=self.sev)

        img = Image.fromarray(img)
        img_fp = img

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imlist)


def split_dataset(ds, split, folds=10, num_train_folds=2, num_val_folds=0):
    all_data_points = np.arange(len(ds))
    every_other = all_data_points[::folds]
    train_folds = num_train_folds
    val_folds = num_val_folds
    train_points = np.concatenate([every_other + i
                                    for i in range(0, train_folds)])
    if num_val_folds > 0:
        val_points = np.concatenate([every_other + i
                                       for i in range(train_folds,
                                                      train_folds + val_folds)])
    if folds - (train_folds + val_folds) > 0:
        unlabelled_points = np.concatenate([every_other + i
                                            for i in range(train_folds + val_folds,
                                                        folds)])
    if split == 'train_util':
        ds = torch.utils.data.Subset(ds, train_points)
    elif split.startswith('val'):
        if num_val_folds == 0:
            raise ValueError("Can't create a val set with 0 folds")
        ds = torch.utils.data.Subset(ds, val_points)
    else:
        if folds - (train_folds + val_folds) == 0:
            raise ValueError('no room for unlabelled points')
        ds = torch.utils.data.Subset(ds, unlabelled_points)
    return ds

def plot(img, name):
    dir = "/volumes2/feature_prior_project/art/feature_prior/vis"
    out = rf"{dir}/{name}.jpg"
    img.save(out)


class CIFAR10ImbalancedNoisy(datasets.CIFAR10):
    """CIFAR100 dataset, with support for Imbalanced and randomly corrupt labels.

    Params
    ------
    corrupt_prob: float
    Default 0.0. The probability of a label being replaced with
    random label.
    num_classes: int
    Default 10. The number of classes in the dataset.
    """
    def __init__(self, corrupt=0.0, gamma=-1, n_min=25, n_max=500, num_classes=100, perc=1.0, **kwargs):
        super(CIFAR10ImbalancedNoisy, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.perc = perc
        self.gamma = gamma
        self.corrupt = corrupt
        self.n_min = n_min
        self.n_max = n_max
        self.true_labels = deepcopy(self.targets)

        if perc < 1.0:
            print('*' * 30)
            print('Creating a Subset of Dataset')
            self.get_subset()
            (unique, counts) = np.unique(self.targets, return_counts=True)
            frequencies = np.asarray((unique, counts)).T
            print(frequencies)

        if gamma > 0:
            print('*' * 30)
            print('Creating Imbalanced Dataset')
            self.imbalanced_dataset()
            self.true_labels = deepcopy(self.targets)

        if corrupt > 0:
            print('*' * 30)
            print('Applying Label Corruption')
            self.corrupt_labels(corrupt)

    def get_subset(self):
        np.random.seed(12345)

        lst_data = []
        lst_targets = []
        targets = np.array(self.targets)
        for class_idx in range(self.num_classes):
            class_indices = np.where(targets == class_idx)[0]
            num_samples = int(self.perc * len(class_indices))
            sel_class_indices = class_indices[:num_samples]
            lst_data.append(self.data[sel_class_indices])
            lst_targets.append(targets[sel_class_indices])

        self.data = np.concatenate(lst_data)
        self.targets = np.concatenate(lst_targets)

        assert len(self.targets) == len(self.data)


    def imbalanced_dataset(self):
        np.random.seed(12345)
        X = np.array([[1, -self.n_max], [1, -self.n_min]])
        Y = np.array([self.n_max, self.n_min * self.num_classes ** (self.gamma)])

        a, b = np.linalg.solve(X, Y)

        classes = list(range(1, self.num_classes + 1))

        imbal_class_counts = []
        for c in classes:
          num_c = int(np.round(a / (b + (c) ** (self.gamma))))
          print(c, num_c)
          imbal_class_counts.append(num_c)

        print(imbal_class_counts)
        targets = np.array(self.targets)

        # Get class indices
        class_indices = [np.where(targets == i)[0] for i in range(self.num_classes)]

        # Get imbalanced number of instances
        imbal_class_indices = [class_idx[:class_count] for class_idx, class_count in zip(class_indices, imbal_class_counts)]
        imbal_class_indices = np.hstack(imbal_class_indices)

        np.random.shuffle(imbal_class_indices)

        # Set target and data to dataset
        self.targets = targets[imbal_class_indices]
        self.data = self.data[imbal_class_indices]

        assert len(self.targets) == len(self.data)

    def corrupt_labels(self, corrupt_prob):
        labels = np.array(self.targets)
        np.random.seed(12345)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.num_classes, mask.sum())
        labels[mask] = rnd_labels

        # we need to explicitly cast the labels from npy.int64 to
        # builtin int type, otherwise pytorch will fail...
        labels = [int(x) for x in labels]
        self.targets = labels
