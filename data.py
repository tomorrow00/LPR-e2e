import os
from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from torchvision.transforms import Normalize

from dataset import DatasetFromFolder


def get_training_set():
    root_dir = os.path.join(os.path.expanduser('~'), 'data/plate_recognition/plate_e2e')
    train_dir = join(root_dir, "train")
    crop_size = (72, 272)
    return DatasetFromFolder(train_dir, input_transform=Compose([CenterCrop(crop_size), Resize(crop_size), ToTensor(),Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),]))

    # return DatasetFromFolder(train_dir,
    #                          input_transform=Compose([ToTensor(),Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225]),]))


def get_val_set():
    root_dir =  os.path.join(os.path.expanduser('~'), 'data/plate_recognition/plate_e2e')
    val_dir = join(root_dir, "validation")
    crop_size = (72, 272)
    return DatasetFromFolder(val_dir, input_transform=Compose([CenterCrop(crop_size), Resize(crop_size), ToTensor(),Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),]))

    # return DatasetFromFolder(val_dir,
    #                          input_transform=Compose([ToTensor(),Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225]),]))
                      
