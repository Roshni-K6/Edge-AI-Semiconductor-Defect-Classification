import torch
from torchvision import transforms
from PIL import Image

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
import torchvision.models as models
model = models.mobilenet_v2(pretrained=True)
num_classes = 9
in_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(in_features, num_classes)
model.load_state_dict(torch.load("mobilenet_defect_model.pth", map_location=device))
model = model.to(device)
model.eval()

# Class names
train_dir = "final_dataset/dataset/train"
from torchvision import datasets
class_names = datasets.ImageFolder(train_dir).classes

# Image path
image_path = "path_to_your_image.png"  # UPDATE this to your test image path

# Preprocess
def preprocess_image(path):
    image = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

input_tensor = preprocess_image(image_path)

# Predict
with torch.no_grad():
    output = model(input_tensor)
    predicted_idx = torch.argmax(output, dim=1).item()

predicted_class = class_names[predicted_idx]
print("Predicted Defect Class:", predicted_class)

