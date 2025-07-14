import torch
import sys
from pathlib import Path
from PIL import Image
from torchvision import transforms
import argparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from models.pokemon_classifier import PokemonCNN

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(device)

IMAGE_SIZE = (256, 256)
mean = 0
std = (2/27)**0.5
data_dir = ROOT / "data" / "raw" / "pokemon_images"
classes = [f.name for f in data_dir.iterdir() if f.is_dir()]

def load_img(image_path):
    transform= transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean]*3, std=[std]*3)  
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    return image

def predict(image_path, classes):
    image = load_img(image_path)
    model = PokemonCNN(num_classes=len(classes))
    model.load_state_dict(torch.load(ROOT/"models"/"trained_model.pth", weights_only=True))
    model.eval()
    model.to(device)
    with torch.no_grad():
        output = model(image)
        prob = torch.softmax(output, dim=1)
        conf, pred = torch.max(prob,1)
    return classes[pred.item()], conf.item()

def predict_gradio(image):
    transform= transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean]*3, std=[std]*3)  
    ])
    image = Image.fromarray(image).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    model = PokemonCNN(num_classes=len(classes))
    model.load_state_dict(torch.load(ROOT/"models"/"trained_model.pth", weights_only=True))
    model.eval()
    model.to(device)
    with torch.no_grad():
        output = model(image)
        prob = torch.softmax(output, dim=1)
        conf, pred = torch.sort(prob, dim=1, descending=True)
        conf, pred = conf[0], pred[0]
    return {classes[pred[i].item()]: conf[i].item() for i in range(5)}
    
        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    args = parser.parse_args()
    label, conf = predict(image_path=args.image_path, classes=classes)
    print(f'Predicted PokeMon: {label}, with confidence {conf:.3f}')