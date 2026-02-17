import os
import numpy as np
import onnxruntime as ort
from PIL import Image 
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# =========================================================
# CONFIGURATION
# =========================================================

MODEL_PATH = "mobilenet_defect_model (3).onnx"
DATASET_PATH = "hackathon_test_dataset"
LOG_FILE = "prediction_log.txt"

class_names = [
    "bridge",
    "clean",
    "cmp",
    "crack",
    "opens",
    "other",
    "particle",
    "scratch",
    "vias"
]

# =========================================================
# LOAD ONNX MODEL
# =========================================================

print("Loading ONNX model...")
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
print("Model loaded successfully.\n")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================================================
# FOLDER NAME MAPPING
# =========================================================

def map_folder(folder_name):
    name = folder_name.lower()

    if name == "open":
        return "opens"
    elif name == "via":
        return "vias"
    elif name == "ler":
        return "other"
    else:
        return name

# =========================================================
# INFERENCE
# =========================================================

true_labels = []
predictions = []
total_images = 0

print("Starting inference...\n")

for folder_name in sorted(os.listdir(DATASET_PATH)):

    folder_path = os.path.join(DATASET_PATH, folder_name)

    if not os.path.isdir(folder_path):
        continue

    mapped_class = map_folder(folder_name)

    if mapped_class not in class_names:
        print(f"Skipping unknown folder: {folder_name}")
        continue

    true_class_index = class_names.index(mapped_class)

    for img_name in sorted(os.listdir(folder_path)):

        img_path = os.path.join(folder_path, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0).numpy()

            outputs = session.run(None, {input_name: image})
            predicted_index = int(np.argmax(outputs[0], axis=1)[0])

            true_labels.append(true_class_index)
            predictions.append(predicted_index)

            total_images += 1

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

print(f"\nTotal images processed: {total_images}")

# =========================================================
# METRICS
# =========================================================

accuracy = accuracy_score(true_labels, predictions)

report = classification_report(
    true_labels,
    predictions,
    labels=list(range(len(class_names))),
    target_names=class_names,
    digits=4,
    zero_division=0
)

cm = confusion_matrix(true_labels, predictions)

print("\n====================================")
print(" PHASE-2 EVALUATION RESULTS")
print("====================================\n")

print(f"Accuracy: {accuracy * 100:.2f} %\n")
print("Classification Report:\n")
print(report)
print("Confusion Matrix:\n")
print(cm)

# =========================================================
# SAVE LOG FILE
# =========================================================

with open(LOG_FILE, "w") as f:
    f.write("DeepTech Hackathon 2026 - Phase-2 Evaluation\n")
    f.write("=============================================\n\n")
    f.write(f"Total Images Processed: {total_images}\n\n")
    f.write(f"Accuracy: {accuracy * 100:.2f} %\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))

print(f"\nLog file saved as '{LOG_FILE}'")
print("Inference completed successfully.")
