'''
==================== Datasets & DataLoaders ======================
'''
# Let's load FashionMNIST dataset - 
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor()
)

test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor()
)

# Iterating and visualizing the dataset -

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3

for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size = (1, )).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap = "gray")
plt.show()

# Creating a Custom Dataset for your files -
'''
A custom dataste must implement three functions - 
__init__, __len__, __getitem__

Here, in the below implementation, FashionMNIST images are stored in a directiry called
img_dir, and their labels are stored separately in a CSV file called 
annotations_file

The labels.csv data looks like -
tshirt1.jpg, 0
tshirt2.jpg, 0
......
ankleboot999.jpg, 9
'''

import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform = None, target_transform = None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        '''
        It loads and returns a sample from the dataset at a given index idx.
        Based on the index, it identifies the image's location on the disk,
        converts that to a tensor using read_image, retrieves the corresponding 
        label from the csv data, calls the transform funtions on them and 
        returns the tensor image and corresonding label in a tuple
        '''
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

'''
The Dataset retrieves our dataset's features and labels one sample at a time. 
'''