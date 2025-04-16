# skin-lesion-image-classifier


# Skin Cancer Prediction Using CNN (HAM10000 Dataset)

This project focuses on predicting different types of skin cancer using Convolutional Neural Networks (CNN) trained on the HAM10000 dataset. The model is deployed using a Flask web interface that allows users to upload skin lesion images and get predictions instantly.

## 🔍 Problem Statement
Skin cancer is one of the most common types of cancer. Early and accurate diagnosis can save lives. This project aims to build a deep learning model to classify skin cancer types from image data.

## 📁 Dataset
- **Source**: [HAM10000 Dataset on Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- Contains over 10,000 dermatoscopic images
- Metadata includes diagnosis labels (`dx`), localization, and image IDs

## 🧠 Model
- CNN (Convolutional Neural Network) using TensorFlow/Keras
- Input shape: 64x64 RGB images
- Output: Multi-class classification based on skin cancer types

## 🧪 Classes Predicted
- Melanocytic nevi (`nv`)
- Melanoma (`mel`)
- Benign keratosis-like lesions (`bkl`)
- Basal cell carcinoma (`bcc`)
- Actinic keratoses (`akiec`)
- Vascular lesions (`vasc`)
- Dermatofibroma (`df`)

## 🛠️ Tech Stack
- **Python** 🐍
- **TensorFlow/Keras** for model building
- **Flask** for web deployment
- **Pandas, NumPy, Matplotlib** for data processing and visualization

## 🚀 How to Run
```bash
git clone https://github.com/yourusername/skin-cancer-prediction.git
cd skin-cancer-prediction
pip install -r requirements.txt
python app.py
