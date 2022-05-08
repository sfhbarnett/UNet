import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
import os
from scipy.ndimage import map_coordinates
import numpy as np
from scipy import interpolate


def generateField(image, subsamplevector, strength):
    shape = image.shape
    nvectors = shape[1]//subsamplevector
    mu, sigma = 0, 0.1  # mean and standard deviation
    sx = np.random.normal(mu, sigma, (nvectors, nvectors))
    sy = np.random.normal(mu, sigma, (nvectors, nvectors))
    x = np.linspace(1, shape[1], shape[1]//subsamplevector)
    y = np.linspace(1, shape[2], shape[2]//subsamplevector)
    f = interpolate.interp2d(x, y, sx)
    xnew = np.linspace(1, shape[1], shape[1])
    ynew = np.linspace(1, shape[2], shape[2])
    sxnew = f(xnew, ynew)
    f = interpolate.interp2d(x, y, sy)
    synew = f(xnew, ynew)
    #plt.quiver(xnew, ynew, sxnew, synew,scale=10)
    #plt.show()
    return sxnew, synew


def elasticdeform(image, field):
    rows, cols = image.shape[1], image.shape[2]
    sx, sy = field
    src_cols = np.linspace(1, cols, cols)
    src_rows = np.linspace(1, rows, rows)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    out_coord = [src_rows+sx*100, src_cols+sy*100]
    out_img = torch.from_numpy(map_coordinates(image.numpy().squeeze(), out_coord, mode='nearest')).unsqueeze(0)
    return out_img


def transform(image, mask):

    #Elastic deformations
    sx, sy = generateField(image, 50, 1)
    image = elasticdeform(image, (sx, sy))
    mask = elasticdeform(mask, (sx, sy))

    # Random horizontal flipping
    if random.random() > 0.5:
        image = transforms.functional.hflip(image)
        mask = transforms.functional.hflip(mask)

    # Random vertical flipping
    if random.random() > 0.5:
        image = transforms.functional.vflip(image)
        mask = transforms.functional.vflip(mask)
    return image, mask


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
        mask_name = os.path.join(self.trainmaskpath, self.images[idx])
        if self.transform is not None:
            image = self.transform(Image.open(img_name))
            masktransform = transforms.Compose([transforms.ToTensor()])
            mask = Image.open(mask_name)
            mask = masktransform(mask)
            image, mask = transform(image, mask)
            mask = mask[:, 5:-5, 5:-5] #Note sure why this is here
            sample = {'image': image, 'mask': mask}
        else:
            image = Image.open(img_name)
            mask = Image.open(mask_name)
            sample = {'image': image, 'mask': mask}
        return sample

