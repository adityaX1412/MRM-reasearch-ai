# model.py

import torch
from torch import nn, optim
import torch.nn.functional as F
from loader import load_data
import os

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.convolutaional_neural_network_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=24*7*7, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=64, out_features=10)
        )

    def forward(self, x):
        x = self.convolutaional_neural_network_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        x = F.log_softmax(x, dim=1)
        return x

def train_and_save_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader, testloader = load_data()
    
    model = Network()
    model.to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 10
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        test_loss = 0
        accuracy = 0

        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                log_probabilities = model(images)
                test_loss += criterion(log_probabilities, labels).item()
                probabilities = torch.exp(log_probabilities)
                top_prob, top_class = probabilities.topk(1, dim=1)
                predictions = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(predictions.type(torch.FloatTensor))

        train_losses.append(train_loss / len(trainloader))
        test_losses.append(test_loss / len(testloader))
        print(f"Epoch: {epoch+1}/{epochs}  Training loss: {train_loss/len(trainloader):.4f}  Testing loss: {test_loss/len(testloader):.4f}  Test accuracy: {accuracy/len(testloader):.4f}")

    model_path = "mnist_cnn.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_save_model()
