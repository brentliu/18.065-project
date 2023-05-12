from skimage import io
from skimage.transform import resize
import os
import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn as nn

np.random.seed(0)
torch.manual_seed(0)

img_width = 200
img_height = 200
img_channels = 3
n_components = 100

img_dir = 'Images'

def get_random_breeds(img_dir, num_breeds):
    breed_names = []
    for breed in os.listdir(img_dir):
        breed_names.append(breed)
    return np.random.choice(breed_names, size=num_breeds, replace=False)

num_breeds = 10
breeds = get_random_breeds(img_dir, num_breeds)

orig_data = []
labels = []
print('Loading images')
for idx in range(len(breeds)):
    breed_dir = f'{img_dir}/{breeds[idx]}'
    for img_file in os.listdir(breed_dir):
        img = resize(io.imread(f'{breed_dir}/{img_file}'), (img_width, img_height)).flatten()
        orig_data.append(img)
        labels.append(idx)
print('Finished loading images')
orig_data = np.array(orig_data)
labels = np.array(labels)

print('Performing PCA')
pca = PCA(n_components=n_components)
data = pca.fit_transform(orig_data)
print('Finished PCA')

train_proportion = 0.8
train_mask = np.random.uniform(size=len(data)) < train_proportion

train_data = data[train_mask]
train_labels = labels[train_mask]
test_data = data[~train_mask]
test_labels = labels[~train_mask]

class MyModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MyModel, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, output_dim),
        )
        
    def forward(self, x):
        return self.linear_relu_stack(x)

net = MyModel(train_data.shape[1], len(breeds))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())

train_data = torch.tensor(train_data, dtype=torch.float32)
test_data = torch.tensor(test_data, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_labels = torch.tensor(test_labels, dtype=torch.long)

for epoch in range(100):  # loop over the dataset multiple times

    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(train_data)
    loss = criterion(outputs, train_labels)
    loss.backward()
    optimizer.step()

    # print statistics
    if epoch % 10 == 0:
        print(f'[{epoch} loss: {loss.item():.3f}')

print('Finished Training')

test_outputs = net(test_data)
_, predicted = torch.max(test_outputs, 1)
correct = (predicted == test_labels).sum().item()
total = test_labels.size(0)
print(f'Accuracy: {correct / total}')

largest_count = torch.topk(torch.bincount(test_labels), 1).values[0].item()
print(f'Largest frequency: {largest_count / total}')