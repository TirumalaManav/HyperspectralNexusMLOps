# 🛰️ Hyperspectral Image Classification Framework

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20Framework-green.svg)](https://flask.palletsprojects.com/)
[![Research](https://img.shields.io/badge/Research-In%20Progress-orange.svg)](https://github.com/TirumalaManav/Hyperspectral-Image-Classification-Framework)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Abstract

The **Hyperspectral Image Classification Framework** is an advanced deep learning platform designed for precision classification of hyperspectral imagery using multiple neural network architectures. This framework implements three distinct approaches: **Convolutional Neural Networks (CNN)**, **Convolutional Autoencoders (CAE)**, and **Generative Adversarial Networks (GAN)** for comprehensive hyperspectral data analysis.

The system incorporates secure user authentication with cryptographic face verification, comprehensive dataset support, and a production-ready web interface for real-time classification and model training.

## 🚀 Key Features

### **🔐 Security & Authentication**
- **Cryptographic Face Verification**: Secure user registration and login
- **Encrypted Data Storage**: .bin files with cryptographic protection
- **Secure Reference System**: Protected storage of biometric templates

### **🧠 Multiple Architectures**
- **HICNN**: Custom CNN with 3/5/7 layer variants
- **HICAE**: Convolutional Autoencoder for feature learning
- **HIC-GAN**: Generative Adversarial Network for data augmentation

### **📊 Comprehensive Dataset Support**
- **Indian Pines**: Agricultural hyperspectral dataset
- **Pavia University/Centre**: Urban hyperspectral imagery
- **Botswana**: Wetland classification dataset
- **Salinas**: Agricultural scene classification
- **KSC**: Kennedy Space Center hyperspectral data

### **🎯 Advanced Optimization**
- **Multiple Optimizers**: Adam, SGD, Adadelta, Adagrad
- **Adaptive Learning**: Dynamic learning rate adjustment
- **Batch Normalization**: Improved training stability
- **Dropout Regularization**: Overfitting prevention


## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    HYPERSPECTRAL CLASSIFICATION FRAMEWORK       │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   WEB INTERFACE │  │  AUTHENTICATION │  │   DATA STORAGE  │  │
│  │                 │  │                 │  │                 │  │
│  │  • Flask App    │  │  • Face Verify  │  │  • Datasets     │  │
│  │  • Templates    │  │  • Cryptography │  │  • Models       │  │
│  │  • Static Files │  │  • Secure Login │  │  • Results      │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                        PROCESSING PIPELINE                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   DATA INPUT    │  │   PREPROCESSING │  │    TRAINING     │  │
│  │                 │  │                 │  │                 │  │
│  │  • Image Upload │  │  • Normalization│  │  • Model Select │  │
│  │  • Dataset Load │  │  • Patch Extract│  │  • Hyperparams  │  │
│  │  • Format Check │  │  • Augmentation │  │  • Optimization │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                      NEURAL ARCHITECTURES                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │     HICNN       │  │     HICAE       │  │     HIC-GAN     │  │
│  │                 │  │                 │  │                 │  │
│  │  • Conv Layers  │  │  • Encoder-Dec  │  │  • Generator    │  │
│  │  • Batch Norm   │  │  • Feature Ext  │  │  • Discriminator│  │
│  │  • Max Pooling  │  │  • Reconstruction│  │  • Classifier   │  │
│  │  • 3/5/7 Depths │  │  • Classification│  │  • Adversarial  │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                        OUTPUT & ANALYSIS                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   PREDICTIONS   │  │   VISUALIZATION │  │     RESULTS     │  │
│  │                 │  │                 │  │                 │  │
│  │  • Class Labels │  │  • Confusion Mat│  │  • Accuracy     │  │
│  │  • Confidence   │  │  • Loss Curves  │  │  • Precision    │  │
│  │  • Probability  │  │  • Feature Maps │  │  • Recall       │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 🔬 Architecture Diagrams

### **1. HICNN (Hyperspectral CNN) Architecture**

```
Input: [Batch, 103, 7, 7] (Hyperspectral Patches)
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                       HICNN ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────┤
│  Input Layer (103 bands, 7×7 patches)                          │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Conv2D(103→64) + BatchNorm + ReLU + MaxPool(2×1)       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Conv2D(64→128) + BatchNorm + ReLU + MaxPool(2×1)       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Conv2D(128→256) + BatchNorm + ReLU + MaxPool(2×1)      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Conv2D(256→512) + BatchNorm + ReLU + MaxPool(2×1)      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Conv2D(512→1024) + BatchNorm + ReLU + MaxPool(2×1)     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Flatten → FC(1024×2×2 → 1024) → ReLU                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ FC(1024 → n_classes) → Softmax                         │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
           Output: [Batch, n_classes] (Classification)
```

### **2. HICAE (Hyperspectral Convolutional Autoencoder) Architecture**

```
Input: [Batch, 103, 7, 7] (Hyperspectral Patches)
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      HICAE ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────┤
│                        ENCODER BRANCH                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Conv2D(103→64) + BatchNorm + ReLU + MaxPool(2×2)       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Conv2D(64→128) + BatchNorm + ReLU + MaxPool(2×2)       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Conv2D(128→256) + BatchNorm + ReLU (Encoded Features)  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│              ┌────────────┴────────────┐                       │
│              ▼                         ▼                       │
│    ┌─────────────────────┐   ┌─────────────────────────────┐   │
│    │   DECODER BRANCH    │   │     CLASSIFIER BRANCH       │   │
│    │                     │   │                             │   │
│    │ ConvTranspose2D     │   │ Conv2D(256→512)             │   │
│    │ (256→128) + BN+ReLU │   │ + BatchNorm + ReLU          │   │
│    │         │           │   │         │                   │   │
│    │         ▼           │   │         ▼                   │   │
│    │ ConvTranspose2D     │   │ Conv2D(512→1024)            │   │
│    │ (128→64) + BN+ReLU  │   │ + BatchNorm + ReLU          │   │
│    │         │           │   │         │                   │   │
│    │         ▼           │   │         ▼                   │   │
│    │ ConvTranspose2D     │   │ AdaptiveAvgPool2d(2×2)      │   │
│    │ (64→103) + Sigmoid  │   │         │                   │   │
│    │         │           │   │         ▼                   │   │
│    │         ▼           │   │ Flatten → FC(4096→1024)     │   │
│    │ Reconstructed       │   │         │                   │   │
│    │ Output              │   │         ▼                   │   │
│    └─────────────────────┘   │ FC(1024→n_classes)          │   │
│                              │         │                   │   │
│                              │         ▼                   │   │
│                              │ Classification Output       │   │
│                              └─────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### **3. HIC-GAN (Hyperspectral GAN) Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│                        HIC-GAN ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Noise Vector (z_dim) + Class Labels (c_dim)                   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    GENERATOR                            │   │
│  │                                                         │   │
│  │  Linear(z_dim+c_dim → h_dim) + SELU                    │   │
│  │                    │                                    │   │
│  │                    ▼                                    │   │
│  │  Linear(h_dim → h_dim) + SELU (×7 layers)              │   │
│  │                    │                                    │   │
│  │                    ▼                                    │   │
│  │  Linear(h_dim → X_dim) + Sigmoid                       │   │
│  │                    │                                    │   │
│  │                    ▼                                    │   │
│  │              Generated Samples                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│           ┌───────────────┼───────────────┐                     │
│           ▼               ▼               ▼                     │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │  DISCRIMINATOR  │ │   CLASSIFIER    │ │  REAL SAMPLES   │   │
│  │                 │ │                 │ │                 │   │
│  │ Linear(X_dim    │ │ Linear(X_dim    │ │ Ground Truth    │   │
│  │ → h_dim) + ELU  │ │ → h_dim) + ELU  │ │ Hyperspectral   │   │
│  │       │         │ │       │         │ │ Data            │   │
│  │       ▼         │ │       ▼         │ │                 │   │
│  │ Linear Layers   │ │ Linear Layers   │ │                 │   │
│  │ (×9) + ELU      │ │ (×9) + ELU      │ │                 │   │
│  │       │         │ │ + Dropout       │ │                 │   │
│  │       ▼         │ │ + BatchNorm     │ │                 │   │
│  │ Linear(h_dim    │ │       │         │ │                 │   │
│  │ → 1) [Real/Fake]│ │       ▼         │ │                 │   │
│  │                 │ │ Linear(h_dim    │ │                 │   │
│  │                 │ │ → c_dim)        │ │                 │   │
│  │                 │ │ [Classification]│ │                 │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                TRAINING OBJECTIVES                      │   │
│  │                                                         │   │
│  │  • Adversarial Loss: min_G max_D V(D,G)                │   │
│  │  • Classification Loss: CrossEntropy(C(x), y)          │   │
│  │  • Feature Matching: ||E[f(x)] - E[f(G(z))]||²        │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```


## 📁 Project Structure

```
Hyperspectral-Image-Classification-Framework/
├── 📁 captured_images/                    # Login face verification images
├── 📁 secure_data/
│   ├── 📁 reference_images/               # Registration face images
│   └── 📁 *.bin                          # Encrypted biometric data
├── 📁 Datasets/
│   ├── 📁 IndianPines/
│   │   ├── Indian_pines_corrected.mat    # Hyperspectral data
│   │   └── Indian_pines_gt.mat           # Ground truth labels
│   └── 📁 PaviaU/
│       ├── PaviaU.mat                    # Pavia University dataset
│       └── PaviaU_gt.mat                 # Pavia ground truth
├── 📁 HIC Images/                        # Training results by architecture
│   ├── 📁 HICNN 3 Adam/                  # CNN 3-layer with Adam optimizer
│   ├── 📁 HICNN 3 SGD/                   # CNN 3-layer with SGD optimizer
│   ├── 📁 HICNN 5 Adadelta/              # CNN 5-layer with Adadelta
│   ├── 📁 HICNN 7 Adagrad/               # CNN 7-layer with Adagrad
│   └── ...                               # All combinations (3×4 optimizers)
├── 📁 results/
│   └── 📁 training_YYYYMMDD_HHMMSS/      # Timestamped training results
│       └── 📁 visualizations/            # Classification reports & graphs
├── 📁 GAN/                               # GAN architecture implementation
│   ├── 📁 Datasets/                      # Multiple hyperspectral datasets
│   │   ├── 📁 Botswana/
│   │   ├── 📁 KSC/
│   │   ├── 📁 Salinas/
│   │   └── ...
│   ├── 📁 GAN Excel/                     # Performance metrics
│   │   ├── Classifier Score.xlsx
│   │   ├── Fake Image Score.xlsx
│   │   └── GAN Scores CRI.xlsx
│   ├── 📁 Gan Results/                   # Results by dataset
│   │   ├── 📁 Botswana Dataset/
│   │   ├── 📁 India Pines Dataset/
│   │   └── ...
│   ├── 📊 GAN Botswana.ipynb             # GAN training notebooks
│   ├── 📊 GAN IndianPines.ipynb
│   └── 📄 requirements.txt
├── 📁 HIC-CLI/                           # Command Line Interface version
│   ├── 📁 .ipynb_checkpoints/            # Jupyter notebook checkpoints
│   ├── 📁 checkpoints/                   # Model checkpoints
│   ├── 📁 Datasets/                      # CLI datasets
│   ├── 📊 Clear GPU Memory.ipynb         # GPU memory management
│   ├── 🐍 custom_datasets.py             # Custom dataset implementations
│   ├── 🐍 datasets.py                    # Dataset loading utilities
│   ├── 🐍 inference.py                   # Model inference script
│   ├── 🐍 main.py                        # Main CLI script
│   ├── 🐍 models.py                      # Model architectures
│   ├── 🐍 utils.py                       # Utility functions
│   └── 📄 requirements.txt               # CLI dependencies
├── 📁 static/                            # Web interface static files
├── 📁 templates/                         # HTML templates
│   ├── 📄 home.html                      # Main dashboard
│   ├── 📄 train.html                     # Model training interface
│   ├── 📄 prediction.html                # Classification interface
│   ├── 📄 register.html                  # User registration
│   ├── 📄 about.html                     # Project information
│   └── 📄 contact.html                   # Contact page
├── 🐍 app.py                             # Main Flask application
├── 🐍 face_verification.py               # Cryptographic authentication
├── 🐍 mlpipeline.py                      # ML training pipeline
├── 🐍 prediction.py                      # Prediction pipeline
├── 📄 requirements.txt                   # Python dependencies
├── 📄 README.md                          # Project documentation
├── 📄 LICENSE                            # MIT License
├── 📄 AUTHORS                            # Contributors
└── 📄 .gitignore                         # Git ignore rules
```

## 🚀 Quick Start

### **Prerequisites**
```bash
# Python 3.8+ required
python --version

# Install CUDA for GPU acceleration (optional but recommended)
nvidia-smi  # Check GPU availability
```

### **Installation**
```bash
# Clone the repository
git clone https://github.com/TirumalaManav/HyperspectralNexusMLOps
cd HyperspectralNexusMLOps

# Create virtual environment
python -m venv hic_env
source hic_env/bin/activate  # On Windows: hic_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For GAN architecture
cd GAN
pip install -r requirements.txt
cd ..

# For CLI version
cd HIC-CLI
pip install -r requirements.txt
cd ..
```

### **Dataset Setup**
```bash
# Download hyperspectral datasets from Google Drive
 https://drive.google.com/drive/folders/1cVRjWpbnXWryTWPunhDqZqW_jJaxqS7y?usp=drive_link
#
# Alternatively, datasets are publicly available on Google or you can create
# custom datasets based on your requirements
#
# Place .mat files in respective dataset folders:
# - IndianPines: Indian_pines_corrected.mat, Indian_pines_gt.mat
# - PaviaU: PaviaU.mat, PaviaU_gt.mat
# - Botswana: Botswana.mat, Botswana_gt.mat
# - KSC: KSC.mat, KSC_gt.mat
# - Salinas: Salinas_corrected.mat, Salinas_gt.mat
```

## 🎯 Usage Guide

### **1. Web Interface (Production)**
```bash
# Start the Flask application
python app.py

# Open browser and navigate to
http://localhost:5000

# Features available:
# - User Registration with Face Verification
# - Secure Login with Cryptographic Authentication
# - Model Training Interface
# - Real-time Classification
# - Results Visualization
```

### **2. Command Line Interface (CLI)**
```bash
# Navigate to CLI directory
cd HIC-CLI

# Start a Visdom server for visualization
python -m visdom.server

# Open browser and navigate to
http://localhost:8097  # or http://localhost:9999 if using Docker

# Run the main CLI script
python main.py

# The most useful arguments are:
# --model to specify the model (e.g. 'svm', 'nn', 'HICNN', 'HICAE', 'hamida', 'lee', 'chen', 'li')
# --dataset to specify which dataset to use (e.g. 'PaviaC', 'PaviaU', 'IndianPines', 'KSC', 'Botswana')
# --cuda switch to run the neural nets on GPU. The tool fallbacks on CPU if not specified.

# For more parameters information:
python main.py -h
```

#### **CLI Examples:**
```bash
# Example 1: SVM on Indian Pines dataset
python main.py --model SVM --dataset IndianPines --training_sample 0.3
# This runs a grid search on SVM on the Indian Pines dataset, using 30% of the
# samples for training and the rest for testing. Results are displayed in the visdom panel.

# Example 2: Basic Neural Network on Pavia University
python main.py --model nn --dataset PaviaU --training_sample 0.1 --cuda 0
# This runs on GPU a basic 4-layers fully connected neural network on the Pavia
# University dataset, using 10% of the samples for training.

# Example 3: HICNN on Pavia University
python main.py --model HICNN --dataset PaviaU --training_sample 0.5 --patch_size 7 --epoch 50 --cuda 0
# This runs on GPU the 3D CNN from Hamida et al. on the Pavia University dataset
# with a patch size of 7, using 50% of the samples for training and optimizing for 50 epochs.

# Example 4: HICAE (Autoencoder) model
python main.py --model HICAE --dataset IndianPines --training_sample 0.4 --cuda 0
# This runs the Convolutional Autoencoder model for feature learning and classification.
```

### **3. Training Models**

#### **CNN Architecture (HICNN)**
```bash
# Access training through web interface or direct script
# Select architecture: HICNN
# Choose layers: 3, 5, or 7
# Select optimizer: Adam, SGD, Adadelta, Adagrad
# Set hyperparameters and start training
```

#### **Autoencoder Architecture (HICAE)**
```bash
# Use mlpipeline.py for autoencoder training
# Features: Encoder-Decoder with classification head
# Reconstruction + Classification loss optimization
```

#### **GAN Architecture (HIC-GAN)**
```bash
# Navigate to GAN folder
cd GAN

# Run specific dataset training
jupyter notebook "GAN IndianPines.ipynb"
jupyter notebook "GAN Botswana.ipynb"
jupyter notebook "GAN Salinas.ipynb"

# Monitor training through Excel metrics
# Results saved in Gan Results/ folder
```

### **4. Model Prediction**
```bash
# Web interface prediction
# Upload hyperspectral image → Select model → Get classification

# Direct script usage
python prediction.py --model HICNN --image path/to/hyperspectral.mat
```

## 🔧 Configuration

### **Model Parameters**
```python
# HICNN Configuration
PATCH_SIZE = 5          # Spatial patch size
LEARNING_RATE = 0.01    # Initial learning rate
BATCH_SIZE = 100        # Training batch size
EPOCHS = 200            # Training epochs

# HICAE Configuration
RECONSTRUCTION_WEIGHT = 0.5  # Reconstruction loss weight
CLASSIFICATION_WEIGHT = 0.5  # Classification loss weight

# GAN Configuration
Z_DIM = 100             # Noise vector dimension
H_DIM = 512             # Hidden layer dimension
C_DIM = 16              # Class condition dimension
```

### **Optimizer Settings**
```python
# Available optimizers with typical settings
OPTIMIZERS = {
    'Adam': {'lr': 0.001, 'betas': (0.9, 0.999)},
    'SGD': {'lr': 0.01, 'momentum': 0.9, 'weight_decay': 0.0005},
    'Adadelta': {'lr': 1.0, 'rho': 0.9},
    'Adagrad': {'lr': 0.01, 'weight_decay': 0.01}
}
```
## 🔐 Security Features

### **Cryptographic Face Verification**
- **Registration**: Secure face template storage with encryption
- **Authentication**: Real-time face verification for login
- **Privacy**: Encrypted .bin files for biometric data protection
- **Anti-spoofing**: Advanced verification algorithms

### **Data Protection**
```python
# Cryptographic implementation highlights
# - AES encryption for biometric templates
# - Secure hash functions for data integrity
# - Protected reference image storage
# - Real-time authentication pipeline
```

## 🚀 Deployment

### **Local Deployment**
```bash
# Development server
python app.py

# Production server with Gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### **Docker Deployment**
```dockerfile
# Dockerfile example
FROM python:3.8-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

## 🤝 Contributing

### **Development Workflow**
```bash
# Fork the repository
git fork https://github.com/TirumalaManav/HyperspectralNexusMLOps.git

# Create feature branch
git checkout -b feature/new-architecture

# Make changes and commit
git commit -m "Add new hyperspectral architecture"

# Push and create pull request
git push origin feature/new-architecture
```

### **Code Standards**
- **PEP 8**: Python code formatting
- **Type Hints**: Function annotations
- **Documentation**: Comprehensive docstrings
- **Testing**: Unit tests for critical functions


## 📈 Performance Optimization

### **GPU Acceleration**
```python
# CUDA optimization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

### **Memory Management**
```python
# Efficient data loading
DataLoader(dataset, batch_size=batch_size,
          num_workers=4, pin_memory=True)

# Gradient checkpointing for large models
torch.utils.checkpoint.checkpoint(model_segment, input)
```


## 🐛 Troubleshooting

### **Common Issues**
```bash
# CUDA out of memory
# Solution: Reduce batch size or use gradient accumulation

# Dataset loading errors
# Solution: Verify .mat file format and path

# Authentication failures
# Solution: Check camera permissions and lighting

# Training instability
# Solution: Adjust learning rate and use gradient clipping
```


## 📞 Support & Contact

### **Primary Authors**
- **Tirumala Manav** (Developer)
  - 📧 Email: thirumalamanav123@gmail.com
  - 🏫 Institution: Hyderabad Institute of Technology and Management

- **Bhaskar Das** (Co-Developer)
  - 🏫 Institution: Hyderabad Institute of Technology and Management

- **Dhananjoy Bhakta** (Co-Developer)
  - 🏫 Institution: Indian Institute of Information Technology, Ranchi, India

- **Lalan Kumar** (Co-Developer)
  - 🏫 Institution: Hyderabad Institute of Technology and Management

### **Technical Support**
- 📋 **Issues**: GitHub Issues for bug reports
- 💬 **Discussions**: GitHub Discussions for questions
- 📧 **Email**: thirumalamanav123@gmail.com for collaboration
- 🌐 **Website**: [Project Documentation](https://github.com/TirumalaManav/HyperspectralNexusMLOps)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
