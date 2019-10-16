from torch.utils.data import Dataset
from PIL import Image
import os

class Datastore(Dataset):
    def __init__(self, filelist,masklist, root_dir, transforms=None):
        self.images = filelist
        self.masks = masklist
        self.root_dir = root_dir
        self.trainimagepath = os.path.join(self.root_dir, 'train')
        self.trainmaskpath = os.path.join(self.root_dir, 'masks')
        self.transform = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.trainimagepath, self.images[idx])
        img_name = os.path.join(self.trainimagepath, self.images[idx])
        mask_name = os.path.join(self.trainmaskpath,self.masks[idx])
        if self.transform != None:
            image = self.transform(Image.open(img_name))
            mask = self.transform(Image.open(mask_name))
            sample = {'image': image, 'mask': mask}
        else:
            image = Image.open(img_name)
            mask = Image.open(mask_name)
            sample = {'image': image, 'mask': mask}
        return sample

