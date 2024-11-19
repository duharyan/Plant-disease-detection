# Plant-disease-detection

This project focuses on detecting plant diseases using Convolutional Neural Networks (CNNs) and Transfer Learning with pretrained models like ResNet and VGG-16. Early and accurate detection of plant diseases is crucial for improving agricultural yields and preventing crop losses.

## Features
- **Custom CNN Model**: A CNN architecture was designed and trained from scratch to classify plant diseases effectively.
- **Transfer Learning**: Leveraged the power of pretrained models, including:
  - **ResNet**: A residual network architecture to capture complex features.
  - **VGG-16**: A deep network with 16 layers known for its simplicity and efficiency in transfer learning scenarios.

---

## Prerequisites
To run this project, you'll need:
- Python (>= 3.7)
- TensorFlow and keras
- Necessary libraries: NumPy, Pandas, Matplotlib, etc.

---

## Dataset
The dataset used in this project is recreated using offline augmentation from the original dataset.This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure. A new directory containing 33 test images is created later for prediction purpose [Watch here](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

---

## Resources for Learning Transfer Learning
- **Introduction to Transfer Learning**: [Watch here](https://www.youtube.com/watch?v=LsdxvjLWkIY&t=260s)
- **Transfer Learning with VGG-16 and ResNet**: [Watch here](https://www.youtube.com/watch?v=zBOavqh3kWU)

---
