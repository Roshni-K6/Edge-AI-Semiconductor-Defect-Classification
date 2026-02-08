#single-image inference
import torch
import torchvision.models as models
from PIL import Image
from utils import get_transforms

classes = ['clean','other','particle','scratch','opens','cracks','cmp','vias','bridges']

model = models.mobilenet_v2()
model.classifier[1] = torch.nn.Linear(model.last_channel, 9)
model.load_state_dict(torch.load("mobilenet_defect_model.pth"))
model.eval()

img = Image.open("test.jpg")
x = get_transforms()(img).unsqueeze(0)

pred = model(x).argmax(1).item()
print("Prediction:", classes[pred])
