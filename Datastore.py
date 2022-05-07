from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
import os


def transform(image, mask):
    #randomCrop 50 percent of time
    if random.random() < 0.5:
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(120, 120))
        image = transforms.functional.crop(image, i, j, h, w)
        mask = transforms.functional.crop(mask, i, j, h, w)

        toScale = random.random()
        if toScale < 0.5:
            resize = transforms.Resize(size=(512,512))
            image = resize(image)
            mask = resize(mask)

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

