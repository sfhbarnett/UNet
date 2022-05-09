import torch
import os
from torchvision import transforms
import Datastore
from Net import UNet
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter



def main(mainpath, load=False):

    torch.cuda.device(0)
    plt.ion()
    trainpath = os.path.join(mainpath, 'image')
    trainmasks = os.path.join(mainpath, 'label')
    filelist = os.listdir(trainpath)
    masklist = os.listdir(trainmasks)
    rgb = 0
    if rgb:
        tforms = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                     transforms.RandomCrop(100)])
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
    epochs = 10
    lr = 0.001
    batch_size = 1
    val_percent = 0.05
    img_scale = 0.5
    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9)
    criterion = nn.BCELoss()
    fig = plt.figure(figsize=(18, 8), dpi=80, facecolor='w', edgecolor='k')
    fig.tight_layout()

    if load:
        try:
            checkpoint = torch.load('model.pt')
            net.load_state_dict(checkpoint['model_state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
        except FileNotFoundError:
            print(f"No model file found at {mainpath}")

    train(net, optimizer, criterion, trainloader, epochs, gpu, batch_N, N_train, mainpath)


def train(net, optimizer, criterion, trainloader, epochs, gpu, batch_N, N_train, mainpath):
    writer = SummaryWriter()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs = data['image']
            inputs = inputs.permute(0, 1, 2, 3)
            masks = data['mask']
            diff = masks.size()[3]-322
            masks = masks[:,:,diff//2:-diff//2,diff//2:-diff//2]
            optimizer.zero_grad()
            if gpu:
                inputs = inputs.to(gpu)
                masks = masks.to(gpu)

            predicted_masks = net(inputs)

            if gpu == 0:

                plt.subplot(1, 4, 1)
                plt.title("Input")
                im = plt.imshow(inputs[0].permute(1, 2, 0).detach().numpy().squeeze())
                plt.colorbar(im, orientation='horizontal',fraction=0.046, pad=0.04)
                plt.subplot(1, 4, 2)
                plt.title("Mask")
                im = plt.imshow(masks[0].detach().numpy().squeeze())
                plt.colorbar(im, orientation='horizontal',fraction=0.046, pad=0.04)
                plt.subplot(1, 4, 3)
                plt.title("Prediction")
                im = plt.imshow(predicted_masks[0].detach().numpy().squeeze())
                plt.colorbar(im, orientation='horizontal', fraction=0.046, pad=0.04)
                plt.subplot(1, 4, 4)
                plt.title("Prediction scaled")
                im = plt.imshow(predicted_masks[0].detach().numpy().squeeze(), vmin=0, vmax=1)
                plt.colorbar(im, orientation='horizontal',fraction=0.046, pad=0.04)
                plt.show()
                plt.draw()
                plt.pause(0.0001)
            loss = criterion(predicted_masks.view(-1), masks.contiguous().view(-1))
            epoch_loss += loss.item()

            print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_N / N_train, loss.item()))
            #optimizer.zero_grad()

            loss.backward()
            optimizer.step()
        print('Epoch finished ! Loss: {}'.format(epoch_loss / i))
        writer.add_scalar("Loss/train", epoch_loss/i, epoch)

        torch.save(net.state_dict(), os.path.join(mainpath, 'model.pt'))

        modelsavepath = 'model.pt'

        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, modelsavepath)

        print(f'Model saved at {modelsavepath}')


if __name__ == "__main__":
    rootpath = 'membrane/train/'
    main(rootpath, load=True)
