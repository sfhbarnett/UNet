from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy
import torch

class Datastore(Dataset):
    def __init__(self, filelist, masklist, root_dir, transforms=None):
        self.images = filelist
        self.masks = masklist
        self.root_dir = root_dir
        self.trainimagepath = os.path.join(self.root_dir, 'image')
        self.trainmaskpath = os.path.join(self.root_dir, 'label')
        self.transform = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.trainimagepath, self.images[idx])
        mask_name = os.path.join(self.trainmaskpath,self.images[idx])
        print(mask_name)
        if self.transform != None:
            image = self.transform(Image.open(img_name))
            masktransform = transforms.Compose([transforms.ToTensor()])
            mask = Image.open(mask_name)
            mask = masktransform(mask)
            mask = mask[:,5:-5, 5:-5]
            mask = mask*255
            sample = {'image': image, 'mask': mask}
        else:
            image = Image.open(img_name)
            mask = Image.open(mask_name)
            mask[10:-10, 10:-10]
            sample = {'image': image, 'mask': mask}
        return sample

