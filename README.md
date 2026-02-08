# Edge-AI-Semiconductor-Defect-Classification
***Overview***

This project implements an edge-optimized image classification system to identify semiconductor wafer defects using a MobileNetV2 deep learning model. The solution is designed with edge deployment constraints in mind (low model size, fast inference) and exported to ONNX for compatibility with NXP eIQ / ONNX Runtime.
The model classifies inspection images into multiple defect categories, along with Clean and Other, enabling automated quality control in semiconductor manufacturing.

***Key Features***

-Lightweight MobileNetV2 architecture

-Transfer learning using PyTorch

-Optimized ONNX model (~8.5 MB)

-High accuracy on test data

-Modular, clean codebase

-Dataset and model shared via Google Drive (GitHub-friendly)

***Model Summary***

| Item               | Value              |
| ------------------ | ------------------ |
| Architecture       | MobileNetV2        |
| Training Approach  | Transfer Learning  |
| Framework          | PyTorch            |
| Export Format      | ONNX               |
| Input Size         | 224 × 224 × 3      |
| Model Size         | ~8.53 MB           |
| Accuracy           | **96.74%**         |
| Precision / Recall | **0.97 / 0.97**    |
| Target             | Edge AI Deployment |

🔗 Model link: see model/README.md

***Dataset Summary***

Total images (current): 1,200+ images

Number of classes: 9 classes (7 defect classes + Clean + Other)

Class list: clean, other, particle/contamination, scratch, opens, cracks, cmp, vias, bridges

Class balance plan: Minimum ~120 images per class to maintain balanced learning and avoid bias toward dominant defect types.

Train / Validation / Test split: 70% / 15% / 15%

Image type: Grayscale images (converted to 3-channel format for CNN compatibility)

Labeling method / source: Manually curated and labeled from publicly available semiconductor defect images sourced from research publications and open references. Data augmentation was applied to increase sample count while preserving defect characteristics.

📂 Dataset link: see dataset/README.md

***How to Run***

This repository does not contain the dataset and trained model because of GitHub size limits.
They must be downloaded separately using the links below.

**1. Clone the Repository**

git clone https://github.com/Roshni-K6/Edge-AI-Semiconductor-Defect-Classification.git

cd Edge-AI-Semiconductor-Defect-Classification

**2. Install Requirements**

Make sure Python 3.8+ is installed.

pip install -r requirements.txt

**3. Download the Dataset**

Download the dataset ZIP from dataset/README.md

👉 Dataset link: https://drive.google.com/file/d/1krg_vpDR0EoZHPNWtp0VPrj425JXW57d/view?usp=sharing

Place the file here: dataset/final_dataset.zip and Extract it

Each folder contains these classes:

clean, other, particle, scratch, opens, cracks, cmp, vias, bridges

**4. Download the Trained ONNX Model**

👉 ONNX model link: https://drive.google.com/file/d/12Pi88YtciSbqGKFJ_QCWeCkzd6X7Go-F/view?usp=drivesdk

Place it here: models/mobilenet_defect_model.onnx

**5. Train the Model (Optional)**

Run this only if you want to train the model yourself: python src/train.py

Output: models/mobilenet_defect_model.pth

**6. Evaluate the Model**

python src/evaluate.py

Outputs: results/confusion_matrix.png
results/metrics.txt

**7. Export Model to ONNX**

python src/export_onnx.py

Output: models/mobilenet_defect.onnx

**8. Run Inference on One Image**

Open src/inference_onnx.py

Set image path: image_path = "path/to/image.png"

Run: python src/inference_onnx.py

Output: Predicted defect class printed in the terminal

***Results***

Accuracy: 96.74%

Precision / Recall: 0.97 / 0.97

Confusion Matrix: see results/confusion_matrix.jpeg

Model Size: ~8.5 MB (edge-ready)

***Team Members***

Roshni K | Reevasri S | Kanimozhi S | Srri Lakshmi M R
