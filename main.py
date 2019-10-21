import torch
import os
from torchvision import transforms
import torch.nn.functional as F
import Datastore
from Net import UNet
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt


def main(mainpath):
    torch.cuda.device(0)
    plt.ion()
    trainpath = os.path.join(mainpath, 'train')
    trainmasks = os.path.join(mainpath, 'masks')
    filelist = os.listdir(trainpath)
    masklist = os.listdir(trainmasks)
    tforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = Datastore.Datastore(filelist, masklist, mainpath, transforms=tforms)
    batch_N = 1
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_N, shuffle=True, num_workers=0)
    N_train = len(dataset)
    gpu = 1
    net = UNet(n_channels=3, n_classes=1)
    if gpu == 1:
        gpu = torch.device("cuda:0")
        print("Connected to device: ", gpu)
        net = net.to(gpu)
    epochs = 1
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
        epoch_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs = data['image']
            inputs = inputs.permute(0, 1, 2, 3)
            masks = data['mask']

            inputs = inputs.to(gpu)
            masks = masks.to(gpu)

            predicted_masks = net(inputs)

            if gpu == 0:
                fig = plt.figure(figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')
                plt.subplot(1, 3, 1)
                im = plt.imshow(inputs[0].permute(1, 2, 0).detach().numpy().squeeze())
                plt.colorbar(im, fraction=0.046, pad=0.04)
                plt.subplot(1, 3, 2)
                im = plt.imshow(masks[0].detach().numpy().squeeze())
                plt.colorbar(im, fraction=0.046, pad=0.04)
                plt.subplot(1, 3, 3)
                im = plt.imshow(predicted_masks[0].detach().numpy().squeeze())
                plt.colorbar(im, fraction=0.046, pad=0.04)
                plt.show();

            loss = criterion(predicted_masks.view(-1), masks.view(-1))
            epoch_loss += loss.item()
            print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_N / N_train, loss.item()))
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
        print('Epoch finished ! Loss: {}'.format(epoch_loss / i))
    torch.save(net.state_dict(), os.path.join(mainpath, 'model.pth'))


if __name__ == "__main__":
    mainpath = r'C:\Users\MBISFHB\Documents\DL_SEG\prac'
    main(mainpath)
