

import torch
from torchvision import datasets, transforms

def load_data(batch_size=64):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_set = datasets.MNIST(root='./data', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    test_set = datasets.MNIST(root='./data', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
    
    return trainloader, testloader

if __name__ == "__main__":
    trainloader, testloader = load_data()
    print(f'Training set size: {len(trainloader.dataset)}')
    print(f'Test set size: {len(testloader.dataset)}')





