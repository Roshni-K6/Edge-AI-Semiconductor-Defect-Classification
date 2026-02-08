# metrics + confusion matrix
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from utils import get_datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, val_ds, _ = get_datasets("final_dataset/dataset")
loader = DataLoader(val_ds, batch_size=32)

model = models.mobilenet_v2()
model.classifier[1] = torch.nn.Linear(model.last_channel, 9)
model.load_state_dict(torch.load("mobilenet_defect_model.pth"))
model.to(device)
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for x, y in loader:
        x = x.to(device)
        preds = model(x).argmax(1).cpu()
        y_pred.extend(preds)
        y_true.extend(y)

print(classification_report(y_true, y_pred))
