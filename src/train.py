 # training logic
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from utils import get_datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_ds, val_ds, _ = get_datasets("final_dataset/dataset")

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32)

model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = torch.nn.Linear(model.last_channel, 9)
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    correct = total = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    print(f"Epoch {epoch+1}: Accuracy {(100*correct/total):.2f}%")

torch.save(model.state_dict(), "mobilenet_defect_model.pth")
