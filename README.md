# 🚧 Road Damage Detection using YOLO11

An AI-powered road damage detection system built using the YOLO object detection framework. This project detects and localizes different types of road damage from images and provides an interactive deployment using Gradio.

---

## 📌 Project Overview

Road surface inspection is essential for maintaining transportation safety. Manual inspection is time-consuming and expensive, while computer vision enables fast and automatic road damage detection.

In this project, three YOLO models were trained, compared, and evaluated to identify the best-performing model for deployment.

---

## 🎯 Objectives

- Train multiple YOLO models for road damage detection.
- Compare the models using standard object detection metrics.
- Perform hyperparameter tuning.
- Select the best-performing model.
- Deploy the final model in an interactive web application.

---

## 📂 Dataset

The project uses the **Road Damage Dataset** containing annotated images of different road damage types.

### Damage Classes

- Alligator Crack
- Block Crack
- Edge Crack
- Longitudinal Crack
- Pothole
- Transverse Crack

---

## 🛠 Technologies

- Python
- YOLOv8
- YOLO11
- YOLO26
- Ultralytics
- OpenCV
- Gradio
- Pandas
- NumPy

---

## ⚙️ Project Workflow

### 1. Data Preparation

- Dataset organization
- Image annotation verification
- Train / Validation / Test split

### 2. Data Augmentation

- Resize
- Mosaic
- Random Flip
- Color Augmentation

### 3. Model Training

The following models were trained:

- YOLOv8
- YOLO11
- YOLO26

### 4. Hyperparameter Tuning

Different values of:

- Learning Rate
- Epochs
- Batch Size
- Confidence Threshold

were tested to improve model performance.

### 5. Model Evaluation

Evaluation metrics include:

- Precision
- Recall
- mAP@50
- mAP@50-95
- Confusion Matrix

---

## 📊 Model Comparison

| Model | Precision | Recall | mAP@50 | mAP@50-95 |
|--------|----------:|--------:|--------:|-----------:|
| YOLOv8 | 58.79% | **39.10%** | 38.46% | 21.63% |
| **YOLO11** | **60.66%** | 37.77% | **38.55%** | 21.68% |
| YOLO26 | 58.17% | 36.98% | 37.83% | **21.78%** |

---

## 🏆 Best Model

YOLO11 achieved the best overall performance by providing the highest Precision and mAP@50 while maintaining strong localization performance.

Therefore, YOLO11 was selected for deployment.

---

## 🚀 Deployment

The trained YOLO11 model was deployed using **Gradio**.

## ▶️ Installation

```bash
pip install ultralytics
pip install gradio
pip install opencv-python
pip install pandas
```
<img width="1536" height="1024" alt="img" src="https://github.com/user-attachments/assets/3e021ba3-d493-4315-95db-a7957b26de3b" />

