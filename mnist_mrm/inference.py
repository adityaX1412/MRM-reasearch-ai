# inference.py

import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from loader import load_data
from model import Network

# moves your model to train on your gpu if available else it uses your cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainloader, testloader = load_data()

# Display a batch of images
training_data = enumerate(trainloader)
batch_idx, (images, labels) = next(training_data)
print(type(images))
print(images.shape)
print(labels.shape)

fig = plt.figure()
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.tight_layout()
    plt.imshow(images[i][0], cmap='inferno')
    plt.title("Ground Truth Label: {}".format(labels[i]))
    plt.yticks([])
    plt.xticks([])
fig.show()

# Initialize and print model
model = Network()
model.to(device)
print(model)

# Define optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Training and evaluation
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

# Single image inference
img = images[0].to(device).view(-1, 1, 28, 28)

with torch.no_grad():
    logits = model.forward(img)

probabilities = F.softmax(logits, dim=1).detach().cpu().numpy().squeeze()
print(probabilities)

fig, (ax1, ax2) = plt.subplots(figsize=(6, 8), ncols=2)
ax1.imshow(img.view(1, 28, 28).detach().cpu().numpy().squeeze(), cmap='inferno')
ax1.axis('off')
ax2.barh(np.arange(10), probabilities, color='r')
ax2.set_aspect(0.1)
ax2.set_yticks(np.arange(10))
ax2.set_yticklabels(np.arange(10))
ax2.set_title('Class Probability')
ax2.set_xlim(0, 1.1)

plt.tight_layout()
plt.show()
