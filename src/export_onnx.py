import torch
import torchvision.models as models

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load trained model
model = models.mobilenet_v2(pretrained=True)
num_classes = 9
in_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(in_features, num_classes)
model.load_state_dict(torch.load("mobilenet_defect_model.pth", map_location=device))
model = model.to(device)
model.eval()

# Dummy input
dummy_input = torch.randn(1, 3, 224, 224, device=device)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model/defect_classifier.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=18
)

print("Model exported to ONNX successfully at model/defect_classifier.onnx")

