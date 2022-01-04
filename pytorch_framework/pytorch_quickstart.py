"""
PyTorch has two primitives to work with data - 
torch.utils.data.Dataset : Dataset stores the samples and their corresponding labels.
torch.utils.data.DataLoader : It wraps an iterable around the Dataset
"""

from typing import ForwardRef
import torch
from torch import nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.linear import Linear
from torch.utils.data import DataLoader
from torchvision import datasets
"""
PyTorch offers domain specific libraries such as TorchText, TorchVision, and TorchAudio
all of the above include datasets.
"""
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

"""
Here we will use torchvision.datasets module which contains Dataset objects for many real-world vision data like
CIFAR, COCO etc. https://pytorch.org/vision/stable/datasets.html
Here, we will use FashionMNIST dataset.
Every TorchVision Dataset object includes two arguments, transform and target_transform to modify the samples and labels respectively.
"""

# Download the training data from open datasets -

training_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor(),
)

# Download the test data from open datasets -

test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor(),
)

"""
We pass the Dataset as an argument to DataLoader. This wraps an iterable over our dataset, and supports automatic batching, 
sampling. shuffling and multi-process data loading. Here we define a batch size of 64, i.e. each element in the dataloader
iterable will return a batch of 64 features and labels
"""
batch_size = 64

# Create data loaders -

train_dataloader = DataLoader(training_data, batch_size = batch_size)
test_dataloader = DataLoader(test_data, batch_size = batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

"""
Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. We intend Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

https://github.com/zalandoresearch/fashion-mnist
"""

"""
Creating Models -
To define a neural network in PyTorch, we create a class that inherits from nn.Module. We define the layers of the network in the __init__ function
and specify how data will pass through the network in the forward function. To accelerate operations in the neural network, we move it to the GPU if available.
"""

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define Model-

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

# Defining the loss function and model parameters -

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)

"""
In a single training loop, the model makes predictions on the training dataset (fed to it in batches),
and backpropagates the prediction error to adjust the model's parameters.
"""

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error -
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation - 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d} / {size:>5d}]")

"""
Check model's performance against the test dataset to ensure the model us learning
"""

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg Loss: {test_loss :>8f} \n")

"""
The training process is conducted over several iterations (epochs). During each epoch, the model learns 
parameters to make better predictions. We print model's accuracy and loss at each epoch.
"""

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1} \n ----------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done !!!")

"""
Saving the model : We serialize the internal state dictionary (containing the model parameters)
"""
torch.save(model.state_dict(), "models/FashionMNIST.pth")
print("Saved PyTorch Model State to FashionMNIST.pth")

"""
Loading the model : Re-creating the model structure and loading the state dictionary
"""
model = NeuralNetwork()
model.load_state_dict(torch.load("models/FashionMNIST.pth"))

"""
Make the predictions -
"""
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')