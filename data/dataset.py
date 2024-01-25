import os
import torch
import torchvision
from data.utils import *
#from data.utils import CIFAR10ImbalancedNoisy
from utilities import dist_utils

# ======================================================================================
# Helper Functions
# ======================================================================================
def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = "/output/classifier/classification/dml/v2/imagenet/cache_imgnet.pt"
    # os.path.join(
    #     "~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt"
    # )
    cache_path = os.path.expanduser(cache_path)
    return cache_path

class CIFAR10:
    """
    CIFAR-10 dataset
    """
    CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    NUM_CLASSES = 10
    MEAN = [0.4914, 0.4822, 0.4465]
    STD = [0.2023, 0.1994, 0.2010]
    SIZE = 32
    SOBEL_UPSAMPLE_SIZE = 64

    def __init__(self, data_path, perc=1.0, corrupt_prob=0.0):
        self.data_path = data_path
        self.perc = perc
        self.corrupt_prob = corrupt_prob

    def get_dataset(self, split, transform_train, transform_test):
        print('==> Preparing CIFAR10 data..')

        assert split in ['train', 'test']
        # if self.corrupt_prob != '0.0':
        #     ds = CIFAR10ImbalancedNoisy(root=self.data_path, train=True, download=True,transform=transform_train,
        #                                  num_classes=self.NUM_CLASSES, gamma=-1, corrupt=self.corrupt_prob, perc=self.perc)

        if split == 'test':
            ds = torchvision.datasets.CIFAR10(root=self.data_path, train=False, download=True, transform=transform_test)
        else:
            ds = torchvision.datasets.CIFAR10(root=self.data_path, train=True, download=True, transform=transform_train)

        return ds

class CIFAR100:
    """
    CIFAR-100 dataset
    """
    NUM_CLASSES = 100
    MEAN = [0.4914, 0.4822, 0.4465]
    STD = [0.2023, 0.1994, 0.2010]
    SIZE = 32
    SOBEL_UPSAMPLE_SIZE = 64

    def __init__(self, data_path):
        self.data_path = data_path

    def get_dataset(self, split, transform_train, transform_test):
        print('==> Preparing CIFAR100 data..')

        assert split in ['train', 'test']
        if split == 'test':
            ds = torchvision.datasets.CIFAR100(root=self.data_path, train=(split == 'train'), download=True, transform=transform_test)
        else:
            ds = torchvision.datasets.CIFAR100(root=self.data_path, train=True, download=True, transform=transform_train)

        return ds

class SVHN:
    """
    SVHN dataset
    """
    NUM_CLASSES = 10
    MEAN = [0.4914, 0.4822, 0.4465]
    STD = [0.2023, 0.1994, 0.2010]
    SIZE = 32
    SOBEL_UPSAMPLE_SIZE = 64

    def __init__(self, data_path):
        self.data_path = data_path

    def get_dataset(self, split, transform_train, transform_test):
        print('==> Preparing SVHN data..')

        assert split in ['train', 'test']
        if split == 'test':
            ds = torchvision.datasets.SVHN(root=self.data_path, split = 'test', download=True, transform=transform_test)
        else:
            ds = torchvision.datasets.SVHN(root=self.data_path, split = 'train', download=True, transform=transform_train)

        return ds

class STL10():
    NUM_CLASSES = 10
    MEAN = [0.4192, 0.4124, 0.3804]
    STD = [0.2714, 0.2679, 0.2771]
    SIZE = 96
    SOBEL_UPSAMPLE_SIZE = 196

    def __init__(self, data_path, tint=0):
        self.data_path = data_path
        self.tint = tint

    def get_dataset(self, split, transform_train=None, transform_test=None):
        print('==> Preparing STL10 data..')

        if split == 'test':
            ds = torchvision.datasets.STL10(root=self.data_path, split=split, folds=None, download=True, transform=transform_test)
        else:
            ds = torchvision.datasets.STL10(root=self.data_path, split=split, folds=None, download=True, transform=transform_train)

        return ds

class TinyImagenet():
    NUM_CLASSES = 200
    MEAN = [0.4802, 0.4481, 0.3975]
    STD = [0.2302, 0.2265, 0.2262]
    SIZE = 64
    SOBEL_UPSAMPLE_SIZE = 128

    def __init__(self, data_path):
        self.data_path = data_path

    def get_dataset(self, split, transform_train, transform_test):
        print('==> Preparing TinyImageNet data..')

        assert split in ['train', 'test']
        if split == 'test':
            ds = ImageFilelist(root=self.data_path, flist=os.path.join(self.data_path, "val_kv_list.txt"), transform=transform_test)
        else:
            ds = ImageFilelist(root=self.data_path, flist=os.path.join(self.data_path, "train_kv_list.txt"), transform=transform_train)

        return ds

class Imagenet():
    NUM_CLASSES = 1000
    MEAN = [0.4802, 0.4481, 0.3975]
    STD = [0.2302, 0.2265, 0.2262]
    SIZE = 256

    def __init__(self, data_path, cache_dataset):
        self.data_path = data_path
        self.cache_dataset = cache_dataset

    def get_dataset(self, split, transform_train, transform_test, transform_train_fp=None, transform_test_fp=None):
        print('==> Preparing ImageNet data..')
        assert split in ['train_util', 'test']

        cache_path = _get_cache_path(os.path.join(self.data_path, "train_util"))
        if self.cache_dataset and os.path.exists(cache_path):
            # Attention, as the transforms are also cached!
            print("Loading dataset_train from {}".format(cache_path))
            ds, _ = torch.load(cache_path)
        else:
            if split == 'test':
                ds = ImageFilelist(root=self.data_path, folderlist=os.path.join(self.data_path, "val"),
                                   transform=transform_test, transform_fp=transform_test_fp)
            else:
                ds = ImageFilelist(root=self.data_path, folderlist=os.path.join(self.data_path, "train_util"),
                                   transform=transform_train, transform_fp=transform_train_fp)

            if self.cache_dataset:
                print("Saving dataset_train to {}".format(cache_path))
                if not os.path.exists(cache_path):
                    dist_utils.mkdir(os.path.dirname(cache_path))
                dist_utils.save_on_master((ds, os.path.join(self.data_path, "train_util")), cache_path)

        return ds

class Imagenet200():
    NUM_CLASSES = 200
    MEAN = [0.4802, 0.4481, 0.3975]
    STD = [0.2302, 0.2265, 0.2262]
    SIZE = 256

    def __init__(self, data_path, cache_dataset):
        self.data_path = data_path
        self.cache_dataset = cache_dataset

    def get_cache_path(self):
        cache_path = "/output/classifier/classification/dml/final/imagenet200/cache_imgnet.pt"
        cache_path = os.path.expanduser(cache_path)
        return cache_path

    def get_dataset(self, split, transform_train, transform_test):
        assert split in ['train_util', 'test']

        cache_path = self.get_cache_path()
        if self.cache_dataset and os.path.exists(cache_path):
            # Attention, as the transforms are also cached!
            print("Loading dataset_train from {}".format(cache_path))
            ds, _ = torch.load(cache_path)
        else:
            if split == 'test':
                ds = ImageFilelist(root=self.data_path, folderlist=os.path.join(self.data_path, "val"), subset_folderlist=os.path.join(self.data_path, "imgnet200.txt"),
                                   transform=transform_test)
            else:
                ds = ImageFilelist(root=self.data_path, folderlist=os.path.join(self.data_path, "train_util"), subset_folderlist=os.path.join(self.data_path, "imgnet200.txt"),
                                   transform=transform_train)

                if self.cache_dataset:
                    print("Saving dataset_train to {}".format(cache_path))
                    if not os.path.exists(cache_path):
                        dist_utils.mkdir(os.path.dirname(cache_path))
                    dist_utils.save_on_master((ds, os.path.join(self.data_path, "train_util")), cache_path)
        return ds

class Corrupt_TinyImagenet():
    NUM_CLASSES = 10
    MEAN = [0.4802, 0.4481, 0.3975]
    STD = [0.2302, 0.2265, 0.2262]
    SIZE = 64
    SOBEL_UPSAMPLE_SIZE = 128
    # class_subset  = {149:0, 21:1, 37:2, 137:3, 66:4, 75:5, 111:6, 41:7, 13:8 ,5:9}
    class_subset  = {149:0, 21:1, 37:2, 137:3, 66:4, 75:5, 111:6, 41:7, 13:8 ,5:9}

    def __init__(self, data_path, sev):
        self.data_path = data_path
        self.sev = sev

    def get_dataset(self, split, transform_train, transform_test, transform_train_fp=None, transform_test_fp=None):
        assert split in ['train_util', 'test']
        if split == 'test':
            ds = ImageFilelist_Corrupt(root=self.data_path, train=False, flist=os.path.join(self.data_path, "val_kv_list.txt"),
                               transform=transform_test, transform_fp=transform_test_fp, class_subset=self.class_subset, sev=self.sev)
        else:
            ds = ImageFilelist_Corrupt(root=self.data_path, train=True, flist=os.path.join(self.data_path, "train_kv_list.txt"),
                               transform=transform_train, transform_fp=transform_train_fp, class_subset=self.class_subset, sev=self.sev)

        return ds


DATASETS = {
    'cifar10': CIFAR10,
    'cifar100': CIFAR100,
    'svhn':SVHN,
    'tinyimagenet': TinyImagenet,
    'stl10': STL10,
    'stl_fourier' : STL10,
    'imagenet': Imagenet,
    'imagenet200': Imagenet200,
    'cor_tinyimagenet':Corrupt_TinyImagenet,
}
