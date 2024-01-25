import cv2
import numpy as np
import PIL
from PIL import Image
from data.utils import plot
import torchvision.transforms as transforms
# from imagecorruptions import get_corruption_names, corrupt
#
# class ImageCorruptions:
#     def __init__(self, args):
#         self.severity = args.corrupt_severity
#         self.corruption_name = args.corrupt_name
#
#     def __call__(self, image, labels=None):
#
#         image = np.array(image)
#         cor_image = corrupt(image, corruption_name=self.corruption_name,
#                         severity=self.severity)
#
#         return Image.fromarray(cor_image)

def transform_cifar(args, ds_class):

    normalize = transforms.Normalize(mean=ds_class.MEAN,
                                     std=ds_class.STD)
    train_aug =[
            #transforms.Pad(4, padding_mode="reflect"),
            transforms.RandomCrop(ds_class.SIZE, padding = 4),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(2),
            transforms.ToTensor(),
        ]
    if args.norm_std == 'True':
        train_aug.append(normalize)
    transform_train = transforms.Compose(train_aug)

    test_aug =[
            #transforms.Resize(size),
            #transforms.CenterCrop(size),
            transforms.ToTensor(),
            #normalize,
        ]
    if args.norm_std == 'True':
        test_aug.append(normalize)
    transform_test = transforms.Compose(test_aug)

    return transform_train, transform_test

def transform_stl(args, ds_class):

    normalize = transforms.Normalize(mean=ds_class.MEAN,
                                     std=ds_class.STD)
    train_aug = [
            transforms.RandomCrop(ds_class.SIZE, padding = 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #normalize if args.norm_aug else None
        ]
    if args.norm_std == 'True':
        train_aug.append(normalize)
    transform_train = transforms.Compose(train_aug)

    test_aug = [
            #transforms.Resize(ds_class.SIZE),
            #transforms.CenterCrop(size),
            transforms.ToTensor(),
            #normalize,
        ]
    if args.norm_std == 'True':
        test_aug.append(normalize)
    transform_test = transforms.Compose(test_aug)

    return transform_train, transform_test

def transform_imagenet(args, ds_class):

    normalize = transforms.Normalize(mean=ds_class.MEAN,
                                     std=ds_class.STD)
    train_aug = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize,
        ]
    if args.norm_std == 'True':
        train_aug.append(normalize)
    transform_train = transforms.Compose(train_aug)

    test_aug = [
            transforms.Resize((ds_class.SIZE,ds_class.SIZE)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # normalize,
        ]
    if args.norm_std == 'True':
        test_aug.append(normalize)
    transform_test = transforms.Compose(test_aug)
    return transform_train, transform_test

def transform_tinyimagenet(args, ds_class):

    normalize = transforms.Normalize(mean=ds_class.MEAN,
                                     std=ds_class.STD)
    train_aug = [
            transforms.RandomResizedCrop(ds_class.SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #normalize,
        ]
    if args.norm_std == 'True':
        train_aug.append(normalize)
    transform_train = transforms.Compose(train_aug)

    test_aug = [
            transforms.Resize(ds_class.SIZE),
            transforms.CenterCrop(ds_class.SIZE),
            transforms.ToTensor(),
            # normalize,
        ]
    if args.norm_std == 'True':
        test_aug.append(normalize)
    transform_test = transforms.Compose(test_aug)
    return transform_train, transform_test


def build_transforms(args, ds_class):

    if args.dataset == "cifar10" or args.dataset == 'cor_cifar10' or args.dataset == 'cifar100' or args.dataset == 'cifarsmallsub'\
            or args.dataset == 'svhn':
        return transform_cifar(args, ds_class)
    elif args.dataset == "stl10":
        return transform_stl(args, ds_class)
    elif args.dataset == "imagenet" or args.dataset == "imagenet200":
        return transform_imagenet(args, ds_class)
    elif args.dataset == "tinyimagenet" or args.dataset == "imagenet_r" or args.dataset == "imagenet_blurry" or args.dataset == "imagenet_a"  \
            or args.dataset == 'cor_tinyimagenet' or args.dataset == 'style_tiny':
        return transform_tinyimagenet(args, ds_class)