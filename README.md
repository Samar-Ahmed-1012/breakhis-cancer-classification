# breakhis-cancer-classification
A deep learning project comparing a custom CNN and a VGG16 transfer learning model for binary classification of breast cancer histopathological images from the BreakHis dataset.

# Breast Cancer Histopathological Image Classification
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-FF6F00?logo=tensorflow)](https://tensorflow.org)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning project implementing and comparing a **Custom Convolutional Neural Network (CNN)** and a **Transfer Learning model with VGG16** for the binary classification of breast cancer histopathological images from the BreakHis dataset.

## 📌 Project Overview

This project was developed as part of an AI/ML internship task. The goal was to build a complete pipeline for a medical image classification problem, from data acquisition and preprocessing to model training, evaluation, and reporting. The models classify images as either **Benign** or **Malignant**.

## 📊 Dataset

The project uses the **BreaKHis (Breast Cancer Histopathological Annotation and Diagnosis)** dataset, available on [Kaggle](https://www.kaggle.com/datasets/ambarish/breakhis).
*   A balanced subset of **500 images** (250 Benign, 250 Malignant) was used.
*   Images were resized to **128x128 pixels** and normalized.

## 🧠 Models & Architectures

1.  **Custom CNN Model:**
    *   3 Convolutional layers with MaxPooling.
    *   Fully Connected layers with Dropout for regularization.
2.  **VGG16 Transfer Learning:**
    *   Pre-trained VGG16 base (weights from ImageNet) with frozen layers.
    *   Custom classifier head with a Dense layer and sigmoid output.

## 🚀 Results

The models were evaluated on a test set of 100 images. The VGG16 model significantly outperformed the custom CNN.

| Model | Test Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Custom CNN** | 50.00% | 25.00% | 50.00% | 33.00% |
| **VGG16 (TL)** | **86.00%** | **87.00%** | **86.00%** | **86.00%** |

**Confusion Matrix (VGG16):**
| | Predicted Benign | Predicted Malignant |
| :--- | :---: | :---: |
| **Actual Benign** | 46 | 4 |
| **Actual Malignant** | 10 | 40 |


## 🛠️ Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/breakhis-cancer-classification.git
    cd breakhis-cancer-classification
    ```

2.  **Install dependencies (if running locally):**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Notebook:**
    *   The primary code is in `breakhis_cancer_detection.ipynb`.
    *   It is recommended to run this on **Google Colab** for easiest setup and access to a GPU.
    *   The notebook will handle the dataset download from Kaggle.

## 📝 Key Learnings

*   The power of **Transfer Learning** for achieving high accuracy with limited data and training time.
*   The importance of model architecture design and the challenges of training CNNs from scratch.
*   The full pipeline of a machine learning project: data preprocessing, model building, training, evaluation, and reporting.

## 👨‍💻 Author

**Samar Ahmed**
*   AI/ML Engineering Intern
*   7th Semester, Bahria University
*   [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin)](https://www.linkedin.com/in/your-profile/)
*   [![GitHub](https://img.shields.io/badge/GitHub-100000?logo=github)](https://github.com/your-username)

## 🔄 Future Work

*   Implement data augmentation to improve model robustness.
*   Debug and improve the custom CNN architecture.
*   Experiment with fine-tuning the VGG16 model.
*   Try other pre-trained architectures like ResNet or EfficientNet.
