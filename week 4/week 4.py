import os
import zipfile
import requests
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder

import torch
from torch import nn
from torch.utils.data import Subset,DataLoader
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
import random
random.seed(42)
torch.manual_seed(42)


class MyCNN(nn.Module):
    def __init__(self, num_classes=101):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3, padding=1),   # (32, 128, 128)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # (32, 64, 64)

            nn.Conv2d(10, 20, kernel_size=3, padding=1),  # (64, 64, 64)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # (64, 32, 32)

            nn.Conv2d(20, 40, kernel_size=3, padding=1), # (128, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # (128, 16, 16)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                     # 128*16*16 = 32768
            nn.Linear(40 * 16 * 16, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
#HYPERPARAMETERS
batch_size = 256
num_epochs = 5
learning_rate = 1e-1

model = MyCNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


#DATA I/O
url = "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1"
data_dir = Path("./caltech101")
zip_path = data_dir / "caltech-101.zip"
extract_dir = data_dir 

data_dir.mkdir(parents=True, exist_ok=True)


# Download if needed
if not zip_path.exists():
    print("📥 Downloading Caltech-101...")
    r = requests.get(url, stream=True)
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

# Extract if needed
if not extract_dir.exists():
    print("📂 Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(data_dir)

import tarfile

# Path to the .tar.gz file inside the extracted zip
inner_tar_path = data_dir / "caltech-101" / "101_ObjectCategories.tar.gz"
target_extract_path = data_dir / "data"

# Extract it only if not already done
if not target_extract_path.exists():
    print("📦 Extracting inner .tar.gz...")
    with tarfile.open(inner_tar_path, "r:gz") as tar:
        tar.extractall(path=target_extract_path)


# Transform and display
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Path to the extracted folder
root_dir = "./caltech101/data/101_ObjectCategories"

# Load dataset
dataset = ImageFolder(root=root_dir, transform=transform)

#Split stratifiedically dataset 
from sklearn.model_selection import StratifiedShuffleSplit
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_indices, test_indices = next(splitter.split(X=torch.zeros(len(dataset.targets)), y=dataset.targets))

train_ds = Subset(dataset, train_indices)
test_ds = Subset(dataset, test_indices)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)


def compute_accuracy(model, dataloader, device=device):
    model.eval()  # set to eval mode (no dropout/batchnorm updates)
    correct = 0
    total = 0

    with torch.no_grad():  # disable gradient tracking
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            #add in background logit = 0
            background_logits = torch.zeros(preds.shape[0], 1, device=device)
            preds = torch.cat((background_logits,preds), dim=1)

            predicted_labels = preds.argmax(dim=1)
            correct += (predicted_labels == y).sum().item()
            total += y.size(0)

    accuracy = correct / total
    return accuracy

def compute_loss(model, dataloader, device=device):
    model.eval()  # set to eval mode (no dropout/batchnorm updates)
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():  # disable gradient tracking
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            #add in background logit = 0
            background_logits = torch.zeros(logits.shape[0], 1, device=device)
            logits = torch.cat((background_logits, logits), dim=1)

            loss = loss_fn(logits, labels)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    return total_loss / total_samples

train_images_seen = 0
train_loss_buffer = []
train_acc_buffer = []
test_loss_buffer = []
test_acc_buffer = []

for t in range(num_epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    # train_loop(train_loader, model, loss_fn, optimizer)
    model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels= labels.to(device)

        #forward
        logits = model(images)
        #add in background logit = 0
        background_logits = torch.zeros(logits.shape[0], 1, device=device)
        #bacground is always 1st class assumed
        logits = torch.cat((background_logits,logits), dim=1)

        loss = loss_fn(logits, labels)  
        #buffer loss
        train_loss_buffer.append([train_images_seen,loss.item()])
        print(loss.item())

        #backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_images_seen += images.shape[0]
   
    #log other metrics
    model.eval()

    test_acc = compute_accuracy(model,test_loader)
    test_acc_buffer.append([train_images_seen, compute_accuracy(model,test_loader)])

    test_loss = compute_loss(model,test_loader)
    test_loss_buffer.append([train_images_seen, test_loss])
    print(f"Test Error: \n Accuracy: {test_acc:>0.1f}%, Avg loss: {test_loss:>8f} \n")

train_acc = compute_accuracy(model,train_loader)
train_acc_buffer.append([train_images_seen, train_acc])

import csv
from pathlib import Path

def save_buffer_csv(buffer, filename):
    filepath = Path(filename)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["images_seen", "value"])
        writer.writerows(buffer)

log_dir = Path("./Baseline")
log_dir.mkdir(exist_ok=True)

save_buffer_csv(train_loss_buffer, log_dir /"train_loss.csv")
save_buffer_csv(train_acc_buffer, log_dir /"train_acc.csv")
save_buffer_csv(test_loss_buffer, log_dir /"test_loss.csv")
save_buffer_csv(test_acc_buffer, log_dir /"test_acc.csv")
