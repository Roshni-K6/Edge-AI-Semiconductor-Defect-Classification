# Edge-AI-Semiconductor-Defect-Classification

**Background**

Semiconductor fabrication involves hundreds of precise steps, where even microscopic defects can reduce yield or cause failures. Modern fabs generate massive volumes of inspection images using optical microscopes, SEM, AFM, and defect review stations. Traditional centralized/manual inspection struggles with high latency, expensive infrastructure, bandwidth bottlenecks, and difficulty scaling to real-time throughput. This project implements an Edge-AI solution to detect and classify wafer defects with high accuracy, low model size, and real-time performance, enabling on-device inspection and portability for Industry 4.0 manufacturing workflows.

**Project Objective**

The goal of this project is to design and develop an Edge-AI capable defect classification system for semiconductor wafer and die inspection. The system aims to automatically detect and classify defects into predefined categories while maintaining high accuracy and a low model size suitable for deployment on edge devices. Additionally, it is designed to support real-time inspection workflows and to be portable across edge deployment frameworks such as NXP eIQ, enabling efficient on-device analysis without relying on cloud infrastructure.

**Dataset Overview**

Source: Semiconductor wafer/die inspection images
Total images: 1000+ (Clean + Defective)
Classes: clean, other, particle/contamination, scratch, opens, cracks, cmp, vias, bridges
Train / Validation / Test split: 70% / 15% / 15%

**Innovation Highlights**

Transfer learning reduces training time and improves accuracy with limited data
Grayscale conversion optional to reduce compute and memory footprint
Edge-ready ONNX model compatible with NXP eIQ and similar frameworks

**Evaluation Metrics**

| Metric           | Value                               |
| ---------------- | ----------------------------------- |
| Accuracy         | 98.91%                              |
| Precision        | 0.99(macro avg)                     |
| Recall           | 0.99(macro avg)                     |
| F1 Score         | 0.99(macro avg)                     |
| Model Size       | 8.2 MB                              |
| Algorithm        | MobileNetV2 (Transfer Learning      |
|Training Platform | PyTorch                             |
| Confusion Matrix | See '/results/confusion_matrix.png' |

**How to Run**

1.Install Dependencies
Make sure Python 3.8+ is installed.
Install all required packages:
**pip install -r requirements.txt**


2.Prepare Dataset
Place your dataset ZIP file as: dataset/dataset.zip
Extract it to maintain this folder structure:
final_dataset/dataset/── train/── val/── test/
Each split contains 9 class folders:
clean, other, particle, scratch, opens, cracks, cmp, vias, bridges


3.Train the Model

Run the training script:
**python src/train.py**
MobileNetV2 is used with transfer learning.
Trains for 10 epochs (adjustable in train.py).
Saves model as: mobilenet_defect_model.pth.


4.Evaluate the Model
Run the evaluation script:
**python src/evaluate.py**
Computes Accuracy, Precision, Recall, F1-score.
Generates confusion matrix saved at: results/confusion_matrix.png.
Prints detailed classification report.


5.Export to ONNX
Run the export script:
**python src/export_onnx.py**
Saves ONNX model at: model/defect_classifier.onnx
Ready for edge deployment (e.g., NXP eIQ).


6.Run Inference on a Single Image
Update image_path in src/inference.py to your image.
Run: **python src/inference.py**
Outputs predicted defect class.


**Team Members**

Roshni K
Reevasri S
Kanimozhi S
Srri Lakshmi M R




