import torch
import os
from torchvision import transforms
import Datastore
from Net import UNet
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np


def main(mainpath, load=False, training=True, weights=False, rgb=0):

    torch.cuda.device(0)
    plt.ion()

    # If data is multi or single channel
    if rgb:
        tforms = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        net = UNet(n_channels=3, n_classes=1)
    else:
        tforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
        net = UNet(n_channels=1, n_classes=1)

    if training:
        trainpath = os.path.join(mainpath, 'image')
        filelist = os.listdir(trainpath)
        trainmasks = os.path.join(mainpath, 'label')
        masklist = os.listdir(trainmasks)

        if weights:
            if os.path.isdir(os.path.join(mainpath, 'weights')) != 1:
                os.mkdir(os.path.join(mainpath, 'weights'))
                print("generating weights")
                for file in masklist:
                    img = Image.open(os.path.join(mainpath, 'label', file))
                    weights = Datastore.generateWeights(img)
                    weights = Image.fromarray(weights)
                    weights.save(os.path.join(mainpath, 'weights', file[:-4]+'.tif'))
                print("generated weights")
                weightspath = os.path.join(mainpath, 'weights')
                weightslist = os.listdir(weightspath)
            else:
                weightspath = os.path.join(mainpath,'weights')
                weightslist = os.listdir(weightspath)

        dataset = Datastore.Datastore(filelist, masklist, weightslist, mainpath, transforms=tforms)
        batch_N = 1
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_N, shuffle=True, num_workers=0)
        N_train = len(dataset)
        gpu = 0
        startepoch = 0

        if gpu == 1:
            gpu = torch.device("cuda:0")
            print("Connected to device: ", gpu)
            net = net.to(gpu)

        epochs = 50
        lr = 0.001
        val_percent = 0.05
        optimizer = optim.SGD(net.parameters(),
                              lr=lr,
                              momentum=0.9)
        criterion = nn.BCEWithLogitsLoss()
        fig = plt.figure(figsize=(18, 5), dpi=80, facecolor='w', edgecolor='k')
        fig.tight_layout()

        # Load in previous model
        if load:
            try:
                checkpoint = torch.load('model2.pt')
                net.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                startepoch = checkpoint['epoch'] + 1
                loss = checkpoint['loss']
            except FileNotFoundError:
                print(f"No model file found at {mainpath}")

        train(net, optimizer, criterion, trainloader, startepoch, epochs, gpu, batch_N, N_train, mainpath)
    else:
        checkpoint = torch.load('model2.pt')
        net.load_state_dict(checkpoint['model_state_dict'])
        predict(net, mainpath)


def train(net, optimizer, criterion, trainloader, startepoch, epochs, gpu, batch_N, N_train, mainpath):
    writer = SummaryWriter()
    for epoch in range(startepoch,epochs):
        epoch_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs = data['image']
            inputs = inputs.permute(0, 1, 2, 3)
            masks = data['mask']
            weights = data['weights']
            optimizer.zero_grad()
            if gpu:
                inputs = inputs.to(gpu)
                masks = masks.to(gpu)

            predicted_masks = net(inputs)

            if gpu == 0:
                plt.subplot(1, 4, 1)
                plt.title("Input")
                im = plt.imshow(inputs[0].permute(1, 2, 0).detach().numpy().squeeze())
                plt.colorbar(im, orientation='horizontal', fraction=0.046, pad=0.04)
                plt.subplot(1, 4, 2)
                plt.title("Mask")
                im = plt.imshow(masks[0].detach().numpy().squeeze())
                plt.colorbar(im, orientation='horizontal', fraction=0.046, pad=0.04)
                plt.subplot(1, 4, 3)
                plt.title("Prediction")
                im = plt.imshow(predicted_masks[0].detach().numpy().squeeze())
                plt.colorbar(im, orientation='horizontal', fraction=0.046, pad=0.04)
                plt.subplot(1, 4, 4)
                plt.title("Prediction scaled")
                im = plt.imshow(predicted_masks[0].detach().numpy().squeeze(), vmin=0, vmax=1)
                plt.colorbar(im, orientation='horizontal', fraction=0.046, pad=0.04)
                plt.show()
                plt.draw()
                plt.pause(0.0001)

            criterion = nn.BCEWithLogitsLoss(weight=weights.view(-1))
            loss = criterion(predicted_masks.view(-1), masks.contiguous().view(-1))
            epoch_loss += loss.item()

            print('Epoch - {0:.1f} --- Progress - {1:.4f} --- loss: {2:.6f}'.format(epoch, i * batch_N / N_train,
                                                                                    loss.item()))
            loss.backward()
            optimizer.step()
        print('Epoch finished ! Mean loss: {}'.format(epoch_loss / i))
        writer.add_scalar("Loss/train", epoch_loss/i, epoch)

        torch.save(net.state_dict(), os.path.join(mainpath, 'model.pt'))

        modelsavepath = 'model2.pt'

        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, modelsavepath)

        print(f'Model saved at {modelsavepath}')


def predict(net, mainpath):
    filelist = os.listdir(mainpath)
    net.eval()
    for filename in filelist[1:]:
        print(filename)
        img = np.array(Image.open(os.path.join(mainpath, filename)))
        tforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
        img = tforms(img)
        img = img.unsqueeze(0)

        with torch.no_grad():
            prediction = net(img)
            # plt.imshow(prediction[0].detach().numpy().squeeze())
            # plt.show()
            predicted = prediction[0].detach().numpy().squeeze()
            predicted[predicted <= 0] = 0
            predicted = Image.fromarray(predicted)
            predicted.save(os.path.join(mainpath, filename[:-4]+'_predicted.tif'))


if __name__ == "__main__":
    rootpath = 'membrane/train/'
    main(rootpath, load=True, training=True, weights=True)
