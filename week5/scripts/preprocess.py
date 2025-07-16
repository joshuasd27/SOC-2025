from torchvision import datasets, transforms
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from pathlib import Path

DEFAULT_DATA_DIR = Path("../data/raw/pokemon_images")
IMAGE_SIZE = (256, 256)
# BATCH_SIZE = 32
mean=0
std=(2/27)**0.5

def get_transforms():
    train_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomCrop(size=(256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean]*3, std=[std]*3)
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean]*3, std=[std]*3)
    ])

    return train_transforms, val_transforms

def preprocess(data_dir = DEFAULT_DATA_DIR):
    print(f"Loading data from {data_dir}")

    train_tfms, val_tfms = get_transforms()
    
    # full_dataset = datasets.ImageFolder(root=data_dir, transform=train_tfms)
    train_transform_dataset = datasets.ImageFolder(root=data_dir, transform=train_tfms)
    val_transform_dataset = datasets.ImageFolder(root=data_dir, transform=val_tfms)

    indices = list(range(len(train_transform_dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2)# stratify=train_transform_dataset.targets)

    train_dataset = Subset(train_transform_dataset, train_idx)
    val_dataset = Subset(val_transform_dataset, val_idx)
    
    return train_dataset, val_dataset, train_transform_dataset.classes