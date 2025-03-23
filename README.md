# Hand Gesture Recognition using Transfer Learning

## 📌 Project Overview
This project focuses on hand gesture recognition using transfer learning with a pre-trained **MobileNetV2** model. The goal is to classify different hand gestures accurately by leveraging deep learning techniques.

## 🔍 Problem Statement
Hand gesture recognition is crucial for applications like sign language interpretation, human-computer interaction, and augmented reality. This project aims to classify hand gestures from images using a deep learning model fine-tuned on a custom dataset.

## 🚀 Approach
1. **Data Collection**: A dataset containing images of different hand gestures is used.
2. **Data Preprocessing**: Images are resized, normalized, and augmented for better generalization.
3. **Model Selection**: MobileNetV2 is chosen as the base model due to its efficiency in feature extraction.
4. **Transfer Learning**: The model is fine-tuned by adding custom fully connected layers.
5. **Training & Evaluation**: The model is trained and evaluated using accuracy and loss metrics.
6. **Prediction**: The trained model is used to classify unseen hand gestures.

## 🏗️ Project Workflow
1. Load and preprocess the dataset.
2. Apply data augmentation for better generalization.
3. Use MobileNetV2 as a feature extractor.
4. Add fully connected layers for classification.
5. Train the model on the dataset.
6. Evaluate model performance.
7. Make predictions on new images.
8. Save and deploy the trained model.

## 📊 Dataset Details
- The dataset contains multiple hand gesture classes.
- Each image is resized to **224x224 pixels** to match MobileNetV2 input size.
- Data is split into **training** and **validation** sets.

## 🔧 Model Architecture
- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Custom Layers**:
  - Global Average Pooling
  - Fully Connected Layers
  - Dropout for regularization
  - Softmax activation for classification

## 🏆 Performance Metrics
- **Accuracy**: Achieved high validation accuracy (~96%)
- **Loss**: Decreased significantly during training
- **Overfitting Handling**: Used dropout and data augmentation

## 📌 How to Use
1. **Install dependencies**:
   - TensorFlow
   - Keras
   - OpenCV
   - NumPy
   - Matplotlib
2. **Train the model** using the provided dataset.
3. **Save the trained model** for future use.
4. **Load the model and make predictions** on new images.

## 📂 Project Structure
```
├── dataset
│   ├── train
│   ├── test
│   ├── validation
├── models
│   ├── saved_model.h5
├── notebooks
│   ├── training.ipynb
├── scripts
│   ├── predict.py
│   ├── train.py
├── README.md
```

## 🎯 Future Improvements
- Increase dataset size for better generalization.
- Experiment with different CNN architectures.
- Deploy the model as a web or mobile application.

## 👨‍💻 Contributors
- **Abhiram** (Project Lead & Developer)

## 📜 License
This project is open-source and available under the MIT License.

