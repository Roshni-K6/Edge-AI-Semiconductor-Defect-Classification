#PyTorch → ONNX
import torch
import torchvision.models as models

model = models.mobilenet_v2()
model.classifier[1] = torch.nn.Linear(model.last_channel, 9)
model.load_state_dict(torch.load("mobilenet_defect_model.pth"))
model.eval()

dummy = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy,
    "mobilenet_defect_model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=18
)

print("ONNX export successful")
