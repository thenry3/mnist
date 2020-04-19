import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils import data
import torch.optim.lr_scheduler as schedule
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        
        self.fc1 = nn.Linear(1600, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(epoch, model, trainloader, optimizer):
    model.train()
    running_loss = 0.0
    for i, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print('epoch: %d | batch: %d loss: %.3f' % (epoch, i + 1, running_loss / 2000))
            running_loss = 0.0

def test(model, testloader):
    model.eval()
    loss, correct = 0, 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.cross_entropy(output, target, reduction='sum').item()
            predictions = output.argmax(dim=1, keepdim=True)
            correct += predictions.eq(target.view_as(predictions)).sum().item()

    loss /= len(testloader.dataset)

    print("\n--TEST--")
    print('Average Loss: %.4f' % (loss))
    print("Accuracy: %d / %d --> %f%%" % (correct, len(testloader.dataset), 100.0 * correct / len(testloader.dataset)))


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataparams = {"batch_size": 4, "shuffle": True, "num_workers": 2}

    trainset = torchvision.datasets.MNIST(root="./data", 
                                        train=True, 
                                        download=True, 
                                        transform=transform)
    trainloader = data.DataLoader(trainset, **dataparams)

    testset = torchvision.datasets.MNIST(root="./data", 
                                        train=False, 
                                        download=True, 
                                        transform=transform)
    testloader = data.DataLoader(testset, **dataparams)

    model = MnistNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.0014)
    scheduler = schedule.StepLR(optimizer, step_size=1)

    # 3 epochs bc idk
    for epoch in range(1, 4):
        train(epoch, model, trainloader, optimizer)
        scheduler.step()

    test(model, testloader)

    torch.save(model.state_dict(), "mnist_cnn.pt")




