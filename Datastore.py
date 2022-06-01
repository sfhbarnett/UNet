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
    nvectors = 3
    mu, sigma = 0, 0.1  # mean and standard deviation
    sx = np.random.normal(mu, sigma, (nvectors, nvectors))
    sy = np.random.normal(mu, sigma, (nvectors, nvectors))
    x = np.linspace(1, shape[1], nvectors)
    y = np.linspace(1, shape[2], nvectors)
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


def generateWeights(img):
    """
    Generates the weights to improve instance segmentation for closely positioned objects of the same class
    :param img: The PIL mask to generate the weights from,
    :return: numpy array of the weights
    """
    img = np.asarray(img)
    weights = np.zeros(img.shape)
    # Make 4-connected structure
    structure = np.ones((3, 3), dtype=int)
    structure[0, 0] = 0
    structure[0, -1] = 0
    structure[-1, 0] = 0
    structure[-1, -1] = 0

    # Padding to deal with boreder region
    padamount = 10
    img = np.pad(img, padamount, mode='constant', constant_values=0)

    # find connected components
    labeled, ncomponents = label(img // 255, structure)

    sigma = 2 * 6 ** 2
    radius = 10

    # Loop over each pixel
    for x in range(padamount, img.shape[0] - padamount):
        for y in range(padamount, img.shape[1] - padamount):
            if img[x, y] == 0:
                distance = []
                crop = labeled[x - radius:x + radius + 1, y - radius:y + radius + 1]  # Crop out region to test
                objects = np.unique(crop)  # find unique objects
                for obj in objects[1:]:  # find closed point in each object to central/test pixel
                    coords = np.where(crop == obj)
                    d = [(coords[0][x] - radius + 1) ** 2 + (coords[1][x] - radius + 1) ** 2 for x in
                         range(len(coords[0]))]
                    distance.append(min(d))
                if len(distance) > 1:
                    distance.sort() # get closest two points and calculate weight
                    weights[x - padamount, y - padamount] = np.exp(
                        -(np.sqrt(distance[0]) + np.sqrt(distance[1])) ** 2 / sigma)
    return weights


def transform(image, mask, weights):

    # Elastic deformations
    sx, sy = generateField(image, 50, 1)
    image = elasticdeform(image, (sx, sy))
    mask = elasticdeform(mask, (sx, sy))
    mask = mask.numpy().squeeze()
    mask[mask <= 0.5] = 0
    mask[mask > 0.5] = 1
    mask = torch.from_numpy(mask).unsqueeze(0)
    weights = elasticdeform(weights, (sx, sy))

    # Random horizontal flipping
    if random.random() > 0.5:
        image = transforms.functional.hflip(image)
        mask = transforms.functional.hflip(mask)
        weights = transforms.functional.hflip(weights)

    # Random vertical flipping
    if random.random() > 0.5:
        image = transforms.functional.vflip(image)
        mask = transforms.functional.vflip(mask)
        weights = transforms.functional.hflip(weights)
    return image, mask, weights


class Datastore(Dataset):
    def __init__(self, filelist, masklist, weightlist, root_dir, transforms=None):
        self.images = filelist
        self.masks = masklist
        self.weights = weightlist
        self.root_dir = root_dir
        self.trainimagepath = os.path.join(self.root_dir, 'image')
        self.trainmaskpath = os.path.join(self.root_dir, 'label')
        self.trainweightpath = os.path.join(self.root_dir, 'weights')
        self.transform = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.trainimagepath, self.images[idx])
        mask_name = os.path.join(self.trainmaskpath, self.images[idx])
        weight_name = os.path.join(self.trainweightpath, self.weights[idx])
        if self.transform is not None:
            image = self.transform(Image.open(img_name))
            masktransform = transforms.Compose([transforms.ToTensor()])
            mask = Image.open(mask_name)
            mask = masktransform(mask)
            weights = Image.open(weight_name)
            weights = masktransform(weights)+1*20
            # Flip image and elastic deform
            image, mask, weights = transform(image, mask, weights)
            sample = {'image': image, 'mask': mask, 'weights': weights}
        else:
            image = Image.open(img_name)
            mask = Image.open(mask_name)
            sample = {'image': image, 'mask': mask}
        return sample



