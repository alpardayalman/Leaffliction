import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms

from model import CustomImageDataset

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 12, 3, stride=2)
        self.conv2 = nn.Conv2d(12, 10, 3, stride=2)
        self.conv3 = nn.Conv2d(10, 10, 3, stride=2)
        self.conv4 = nn.Conv2d(10, 10, 3, stride=2)
        self.fc1 = nn.Linear(10 * 15 * 15, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 8)

        self.debug = True

    def forward(self, x):
        if self.debug: print('Input', x.shape)  # DEBUG
        x = F.relu(self.conv1(x))
        if self.debug: print('Conv1', x.shape)  # DEBUG
        x = F.relu(self.conv2(x))
        if self.debug: print('Conv2', x.shape)  # DEBUG
        x = F.relu(self.conv3(x))
        if self.debug: print('Conv3', x.shape)  # DEBUG
        x = F.relu(self.conv4(x))
        if self.debug: print('Conv4', x.shape)  # DEBUG
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        if self.debug:
            self.debug = False

        return x

def train_net(net, trainLoader, optimizer, criterion,
              device=None, epoch=100):

    from time import time
    from datetime import timedelta

    print('Starting training with', epoch, "number of epoch")

    start = time()
    
    for epoch in range(epoch):
        running_loss = 0.
        next_epoch = epoch + 1
        counter = 0
        for _, data in enumerate(trainLoader, 0):
            inputs, labels = data

            if device:
                inputs = inputs.to(device)
                labels = labels.to(device)

            # Zero parameter gradients
            optimizer.zero_grad()
            # for param in optimizer.parameters():  # Faster?
            #     param.grad = None

            # forward + backward + optimization
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if epoch == next_epoch:
                print(f"[{epoch:3d}] "
                      f"loss: {running_loss / counter:.3f} "
                      f"counter: {counter}")
                running_loss = 0.
                counter = 0

            counter += 1
            running_loss += loss.item()
            # loss_history.append(loss.item())

    end = time()
    
    print("Training is finished.")
    print("Elapsed time :", timedelta(seconds=(end - start)))

def test_net(net, testLoader, device=None):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testLoader:
            images, labels = data

            if device:
                images = images.to(device)
                labels = labels.to(device)
            
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy of the network on {total} "
          f"test images: {100 * correct // total}%")

def main():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])

    device = torch.accelerator.current_accelerator().type \
        if torch.accelerator.is_available() \
           else "cpu"
    print(f"Using {device} device")

    data = CustomImageDataset("/home/timur/workspace/test/images",
                              transform=transform)

    gen = torch.Generator().manual_seed(40)
    train, test = torch.utils.data.random_split(data,
                                                [.8, .2],
                                                generator=gen)

    batch_size = 64
    
    trainLoader = torch.utils.data.DataLoader(train,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=3)
    testLoader = torch.utils.data.DataLoader(test,
                                             batch_size=32,
                                             shuffle=True)

    net = Net().to(device)
    print(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    train_net(net, trainLoader, optimizer, criterion, device=device, epoch=100)

    torch.save(net.state_dict(), "mytestmodel.pth")

    test_net(net, testLoader, device=device)


if __name__ == "__main__":
    main()
