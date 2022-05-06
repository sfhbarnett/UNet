import torch
import os
from torchvision import transforms
import Datastore
from Net import UNet
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt


def main(mainpath,load=False):


    torch.cuda.device(0)
    plt.ion()
    trainpath = os.path.join(mainpath, 'image')
    trainmasks = os.path.join(mainpath, 'label')
    filelist = os.listdir(trainpath)
    masklist = os.listdir(trainmasks)
    rgb = 0
    if rgb:
        tforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        net = UNet(n_channels=3, n_classes=1)
    else:
        tforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
        net = UNet(n_channels=1, n_classes=1)
    dataset = Datastore.Datastore(filelist, masklist, mainpath, transforms=tforms)
    batch_N = 1
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_N, shuffle=True, num_workers=0)
    N_train = len(dataset)
    gpu = 0

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
    fig = plt.figure(figsize=(14, 9), dpi=80, facecolor='w', edgecolor='k')

    if load:
        checkpoint = torch.load(os.path.join(mainpath, 'model.pt'))
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        #model.eval()
        # - or -
        #model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs = data['image']
            inputs = inputs.permute(0, 1, 2, 3)
            masks = data['mask']

            if gpu:
                inputs = inputs.to(gpu)
                masks = masks.to(gpu)

            predicted_masks = net(inputs)

            if gpu == 0:

                plt.subplot(1, 3, 1)
                plt.title("Input")
                im = plt.imshow(inputs[0].permute(1, 2, 0).detach().numpy().squeeze())
                plt.colorbar(im, fraction=0.046, pad=0.04)
                plt.subplot(1, 3, 2)
                plt.title("Mask")
                im = plt.imshow(masks[0].detach().numpy().squeeze())
                plt.colorbar(im, fraction=0.046, pad=0.04)
                plt.subplot(1, 3, 3)
                plt.title("Prediction")
                im = plt.imshow(predicted_masks[0].detach().numpy().squeeze())
                plt.colorbar(im, fraction=0.046, pad=0.04)
                plt.show()
                plt.draw()
                plt.pause(0.0001)

            loss = criterion(predicted_masks.view(-1), masks.view(-1))
            epoch_loss += loss.item()
            print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_N / N_train, loss.item()))
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
        print('Epoch finished ! Loss: {}'.format(epoch_loss / i))

        torch.save(net.state_dict(), os.path.join(mainpath, 'model.pt'))

        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, os.path.join(mainpath, 'model.pt'))


if __name__ == "__main__":
    rootpath = 'membrane/train/'
    main(rootpath, load=True)
