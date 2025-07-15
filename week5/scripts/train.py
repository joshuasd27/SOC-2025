import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os
from preprocess import preprocess
import argparse
import matplotlib.pyplot as plt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.pokemon_classifier import PokemonCNN

# Set environment variables for reproducibility
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(device)

ROOT_DIR = Path(__file__).resolve().parent.parent
data_dir = ROOT_DIR / "data" / "raw" / "pokemon_images"

ACTIVATION_MAP = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "Tanh": nn.Tanh,
    "Sigmoid": nn.Sigmoid,
    "ELU": nn.ELU
}

parser = argparse.ArgumentParser(description="Train Pokémon CNN model")
parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
parser.add_argument('--batch_size', type=int, default=64, help="Training batch size")
parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
parser.add_argument('--activation', type=str, default="ReLU", choices=ACTIVATION_MAP.keys(),help="Activation function to use in conv layers")
parser.add_argument('--save', type=bool, default=False, help="Set True to save")
args = parser.parse_args()

train_dataset, val_dataset, classes = preprocess(data_dir)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size,shuffle=False)

activation = ACTIVATION_MAP[args.activation]
model = PokemonCNN(num_classes=len(classes), activation=activation).to(device)
loss_fn=nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr)

def train_model(model=model, 
                train_loader=train_loader, 
                val_loader=val_loader, 
                num_epochs=5, 
                loss_fn=loss_fn, 
                optimizer=optimizer,
                lr=args.lr,
                save=args.save
                ):
    train_losses=[] 
    val_losses=[] 
    val_accs = []
    train_accs=[]
    for epoch in range(num_epochs):
        model.train()
        train_loss=0.0
        train_correct=0
        for i, (image, label) in enumerate(train_loader):
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(image)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * image.size(0)
            _, preds = torch.max(output, 1)
            train_correct+= (preds==label).sum().item()
            if (i+1)%10==0:
                print(f'Epoch {epoch+1}, Step {i+1}, Loss= {loss.item():.4f}')
        train_loss/=len(train_loader.dataset)
        train_losses.append(train_loss)
        train_acc = train_correct/len(train_loader.dataset)
        train_accs.append(train_acc)
        
        model.eval()
        val_correct = 0
        val_loss = 0.0
        with torch.no_grad():
            for i, (image, label) in enumerate(val_loader):
                image = image.to(device)
                label= label.to(device)
                output=model(image)
                loss = loss_fn(output, label)
                val_loss += loss.item()*image.size(0)
                _, preds = torch.max(output, 1)
                val_correct+= (preds==label).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct/len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f'Correct {val_correct} out of {len(val_loader.dataset)}')
        print(f'\nEpoch {epoch+1}\nTrain_Loss= {train_loss:.4f}\nValidation Loss= {val_loss:.4f}\nValidation accuracy= {val_acc:.4f}\nTrain_accuracy= {train_acc:.4f}\n')
    if save:
        torch.save(model.state_dict(), ROOT_DIR/"models"/"trained_model.pth")
        print('Saved the model at', Path(ROOT_DIR/"models"/"trained_model.pth"))
    
    plt.figure(figsize=(10,4))
    plt.suptitle(f'Epochs={num_epochs} lr={lr} activation={args.activation}')
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(train_accs, label= 'Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.legend()
    plt.show()
    
    
train_model(model=model, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            num_epochs=args.epochs, 
            loss_fn=loss_fn, 
            optimizer=optimizer)