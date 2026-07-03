# ♻️ Waste Image Classification using Deep Learning

A Computer Vision project developed to automatically classify waste images into different recycling categories using **Transfer Learning** and **MobileNetV2**.

The objective of this project is to compare the performance of the same deep learning architecture on three different waste image datasets while applying modern techniques such as data augmentation, fine-tuning, class balancing and early stopping.

This project was developed as part of the **Advanced Machine Learning** course of the **Data Engineering and Artificial Intelligence Degree** at the **University of León**.

---

# 📷 Project Overview

The complete machine learning workflow includes:

- Image preprocessing
- Dataset exploration
- Data augmentation
- Transfer Learning using MobileNetV2
- Fine-tuning of pretrained layers
- Class balancing
- Model evaluation
- Confusion matrix generation
- Performance visualization

---

# 🧠 Model Architecture

The classifier is based on **MobileNetV2**, pretrained on ImageNet.

Instead of training a CNN from scratch, Transfer Learning was applied by:

- Loading pretrained ImageNet weights.
- Freezing most convolutional layers.
- Fine-tuning the last 30 layers.
- Adding custom classification layers.

The final architecture consists of:

```
Input Image (224x224)

↓

MobileNetV2 (ImageNet)

↓

Global Average Pooling

↓

Dropout (0.3)

↓

Dense (128, ReLU)

↓

Batch Normalization

↓

Softmax Output Layer
```

---

# 🚀 Training Techniques

Several techniques were used to improve model generalization:

- Transfer Learning
- Fine Tuning
- Data Augmentation
- Early Stopping
- Class Weight Balancing
- Adam Optimizer
- Validation Split
- Confusion Matrix Evaluation

---

# 📊 Data Augmentation

The training images are augmented using:

- Random rotations
- Horizontal flip
- Vertical flip
- Zoom
- Shear transformation
- Width shifting
- Height shifting
- Brightness variation

These transformations significantly reduce overfitting and improve robustness.

---

# 📁 Supported Datasets

The project was designed to work with two different datasets.

## 1. TrashNet

Small academic dataset commonly used for waste classification.

Folder structure:

```
trashnet/
```

Recommended parameters:

| Parameter | Value |
|------------|------:|
| Epochs | 100 |
| EarlyStopping Patience | 8 |

---

## 2. Garbage Classification

The complete dataset is **not included** in this repository because of its size.

It can be downloaded from Kaggle:

https://www.kaggle.com/datasets/mostafaabla/garbage-classification

After downloading:

1. Extract the ZIP file.
2. Place the folder next to `garbage_classification_mobilenetv2.py`.
3. Rename it (or keep it) as:

```
garbage_classification/
```

Recommended parameters:

| Parameter | Value |
|------------|------:|
| Epochs | 20 |
| EarlyStopping Patience | 5 |

---

# ⚙️ Dataset Configuration

The dataset used for training is selected by modifying the following variable inside **train.py**:

```python
base_path = "./garbage_classification/"
```

No other modifications are required.

---

# 📂 Repository Structure

```
waste-image-classification
│
├── screenshots/
│   ├── architecture.png
│   ├── training.png
│   ├── confusion-matrix.png
│   └── prediction.png
│
├── garbage_classification_mobilenetv2.py
├── requirements.txt
├── LICENSE
├── README.md
└── .gitignore
```

---

# ▶️ Installation

Clone the repository

```bash
git clone https://github.com/OscarMartin05/waste-image-classification.git

cd waste-image-classification
```

Install the dependencies

```bash
pip install -r requirements.txt
```

Download the dataset and place it in the project root.

---

# ▶️ Training

Simply execute:

```bash
python garbage_classification_mobilenetv2.py
```

The script automatically:

- Loads the dataset
- Creates the training, validation and testing splits
- Applies data augmentation
- Builds the MobileNetV2 model
- Trains the network
- Evaluates the model
- Displays the learning curves
- Generates the confusion matrix
- Prints the classification report

---

# 📈 Evaluation

The trained model is evaluated using:

- Test Accuracy
- Precision
- Recall
- F1-score
- Classification Report
- Confusion Matrix

---

# 🛠 Technologies

- Python
- TensorFlow
- Keras
- MobileNetV2
- NumPy
- Pandas
- OpenCV
- Scikit-Learn
- Matplotlib
- Seaborn

---

# 📸 Screenshots

## Training Curves

*(Insert screenshot here)*

---

## Confusion Matrix

*(Insert screenshot here)*

---

## Example Predictions

*(Insert screenshot here)*

---

# 📚 Future Improvements

Possible future work includes:

- ResNet50
- EfficientNet
- Vision Transformers (ViT)
- Hyperparameter Optimization
- TensorBoard Integration
- Grad-CAM Explainability
- Model Export for Mobile Devices
- Real-Time Webcam Classification

---

# 👨‍💻 Author

**Óscar Martín García**

B.Sc. Data Engineering & Artificial Intelligence

University of León

---

## ⭐ If you found this project useful, consider giving it a star!