# inference.py

import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from loader import load_data
from model import Network

# moves your model to train on your gpu if available else it uses your cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = Network()
model_path = "mnist_cnn.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Load the data
_, testloader = load_data()

# Display a batch of images
test_data = enumerate(testloader)
batch_idx, (images, labels) = next(test_data)
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

