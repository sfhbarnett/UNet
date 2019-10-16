import torch
import os
from torchvision import transforms
import torch.nn.functional as F
import Datastore
from Net import UNet
from torch import optim
import torch.nn as nn

mainpath  = r'C:\Users\MBISFHB\Documents\DL_SEG\prac'
trainpath = os.path.join(mainpath,'train')
trainmasks = os.path.join(mainpath,'masks')
filelist = os.listdir(trainpath)
masklist = os.listdir(trainmasks)
tforms = transforms.Compose([transforms.ToTensor()])#,transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = Datastore.Datastore(filelist,masklist, mainpath, transforms=tforms)

trainloader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=True,num_workers=0)

net = UNet(n_channels=3,n_classes=1)

epochs = 5
lr = 0.1
batch_size = 1
val_percent = 0.05
img_scale = 0.5
optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)
criterion = nn.BCELoss()

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs = data['image']
        inputs = inputs.permute(0,1,2,3)
        print(inputs.shape)
        masks = data['mask']
        optimizer.zero_grad()
        predicted_masks = net(inputs)
        print(masks.shape,predicted_masks.shape)
        loss = criterion(predicted_masks.view(-1),masks.view(-1))
        running_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(running_loss)

