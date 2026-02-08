 # transforms, loaders, helpers
from torchvision import datasets, transforms

def get_transforms():
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

def get_datasets(data_dir):
    transform = get_transforms()
    train = datasets.ImageFolder(f"{data_dir}/train", transform=transform)
    val   = datasets.ImageFolder(f"{data_dir}/val", transform=transform)
    test  = datasets.ImageFolder(f"{data_dir}/test", transform=transform)
    return train, val, test
