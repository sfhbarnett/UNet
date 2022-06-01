import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
import os
from scipy.ndimage import map_coordinates
import numpy as np
from scipy import interpolate
from scipy.ndimage import label


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
    # plt.quiver(xnew, ynew, sxnew, synew,scale=10)
    # plt.show()
    return sxnew, synew


def elasticdeform(image, field):
    rows, cols = image.shape[1], image.shape[2]
    sx, sy = field
    src_cols = np.linspace(1, cols, cols)
    src_rows = np.linspace(1, rows, rows)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    out_coord = [src_rows+sx*100, src_cols+sy*100] # scale vectors by a factor of 100
    #convert tensor to numpy and perform elastic deformations
    out_img = torch.from_numpy(map_coordinates(image.numpy().squeeze(), out_coord, mode='nearest')).unsqueeze(0)
    return out_img


def makeMask(radius):
    area = np.zeros((2 * radius + 1, 2 * radius + 1))
    area[0, :] = 1
    area[-1, :] = 1
    area[:, 0] = 1
    area[:, -1] = 1
    return area

def generateWeights(img):
    img = np.asarray(img)

    weights = np.zeros(img.shape)

    structure = np.ones((3, 3), dtype=int)
    padamount = 10

    img = np.pad(img, padamount, mode='constant', constant_values=0)

    labeled, ncomponents = label(img // 255, structure)

    masks = [makeMask(1), makeMask(3), makeMask(5), makeMask(7), makeMask(9)]
    sigma = 2 * 6 ** 2

    for x in range(padamount, img.shape[0] - padamount):
        for y in range(padamount, img.shape[1] - padamount):
            if img[x, y] == 0:
                distance = []
                objectlist = []
                for radius in range(len(masks)):
                    area = masks[radius]
                    radius = (radius + 1) * 2 - 1
                    crop = labeled[x - radius:x + radius + 1, y - radius:y + radius + 1]
                    product = crop * area
                    objectindexes = np.where(product > 0)
                    objectlist += list(crop[objectindexes])
                    objectlist = list(set(objectlist))
                    distance += [*[radius] * len(objectindexes[0])]
                    if len(objectlist) > 1:
                        weights[x - padamount, y - padamount] = np.exp(-(distance[0] + distance[1]) ** 2 / sigma)
                        break
    return weights


def transform(image, mask):

    # Elastic deformations
    sx, sy = generateField(image, 50, 1)
    image = elasticdeform(image, (sx, sy))
    mask = elasticdeform(mask, (sx, sy))
    mask = mask.numpy().squeeze()
    mask[mask <= 0.5] = 0
    mask[mask > 0.5] = 1
    mask = torch.from_numpy(mask).unsqueeze(0)

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
            # Flip image and elastic deform
            image, mask = transform(image, mask)
            sample = {'image': image, 'mask': mask}
        else:
            image = Image.open(img_name)
            mask = Image.open(mask_name)
            sample = {'image': image, 'mask': mask}
        return sample



