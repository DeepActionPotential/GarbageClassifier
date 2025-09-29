# utils.py

import torch
from torchvision import transforms
from PIL import Image

IMG_SIZE = 224  # Or your desired size

class_names = ['battery', 'biological', 'cardboard', 'clothes', 'glass',
               'metal', 'paper', 'platic', 'shoes', 'trash']

# Transformation same as your test transform
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

def load_model(weights_path, device):



    model = torch.load(weights_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    return model

def predict_image(model, image, device):
    image = image.convert("RGB")
    input_tensor = test_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        class_index = predicted.item()
        return class_names[class_index]
