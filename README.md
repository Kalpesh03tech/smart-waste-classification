# ♻️ Smart Waste Classification System

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An end-to-end intelligent computer vision application designed to automatically classify municipal solid waste items into **Glass**, **Metal**, **Paper**, and **Plastic**, providing real-time recycling and disposal instructions.

Powered by a fine-tuned **MobileNetV2** deep convolutional neural network and wrapped in an intuitive **Streamlit** web interface, this tool bridges the gap between machine learning classification and actionable environmental sustainability.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Repository Structure](#-repository-structure)
- [Dataset Details](#-dataset-details)
- [Model Architecture & Training Pipeline](#-model-architecture--training-pipeline)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
  - [1. Running the Web Application](#1-running-the-web-application)
  - [2. Training the Model from Scratch](#2-training-the-model-from-scratch)
- [Safety & Confidence Thresholding](#-safety--confidence-thresholding)
- [Material Disposal & Recycling Guidance](#-material-disposal--recycling-guidance)
- [Troubleshooting & FAQs](#-troubleshooting--faqs)
- [Future Enhancements](#-future-enhancements)
- [License](#-license)

---

## 🔍 Overview

Improper waste segregation significantly impedes recycling operations, increases landfill burden, and damages processing equipment. The **Smart Waste Classification System** addresses this problem by:

1. **Automating Detection**: Classifying waste items from standard photos (drag-and-drop or device capture).
2. **Preventing Misdirection**: Applying an automated confidence threshold (default 60%) so doubtful images prompt manual verification rather than incorrect bin disposal.
3. **Educating Users**: Presenting immediate, actionable guidelines (rinsing, removing caps, separating contaminants) tailored to the predicted material.

---

## ✨ Key Features

- **High-Efficiency Deep Learning Backbone**: Built on `MobileNetV2` with ImageNet pre-trained weights for fast inference and lightweight deployment.
- **Two-Phase Transfer Learning Pipeline**:
  - *Phase 1*: Trains custom classification head with base model frozen.
  - *Phase 2*: Fine-tunes the top 30 convolutional layers with a reduced learning rate for optimal domain adaptation.
- **Data Augmentation**: Integrated on-the-fly transformations (random horizontal flips, rotations, zooms, and contrast shifts) to prevent overfitting.
- **Class-Imbalance Compensation**: Automatically computes inverse-frequency class weights during training to balance underrepresented categories.
- **Interactive Streamlit Web Dashboard**:
  - Drag-and-drop file uploader supporting `JPG`, `JPEG`, `PNG`, `BMP`, and `WEBP`.
  - Image integrity verification to prevent corrupted inputs.
  - Real-time probability distributions for all classes via interactive progress bars.
  - Dynamic display of disposal guidelines.
- **Model Caching**: Streamlit resource caching (`@st.cache_resource`) ensures the heavy neural network is loaded into memory only once.

---

## 🏗 System Architecture

```mermaid
flowchart TD
    A[User Uploads Image] --> B[Streamlit Web App]
    B --> C[Image Preprocessing & Validation\n224x224 RGB, [-1, 1] Range]
    C --> D[MobileNetV2 Deep Learning Model]
    D --> E[Softmax Output Probabilities]
    E --> F{Confidence >= 60%?}
    F -- Yes --> G[Display Predicted Class & Confidence]
    G --> H[Show Probability Distribution Breakdown]
    H --> I[Display Actionable Disposal / Recycling Guidance]
    F -- No --> J[Display Uncertainty Warning & Best Guess]
    J --> K[Withhold Disposal Instructions & Request Manual Inspection]
```

---

## 📁 Repository Structure

```text
Smart Waste Classification/
├── app.py                     # Streamlit web application & inference interface
├── train.py                   # Complete training, fine-tuning & evaluation pipeline
├── requirements.txt           # Project dependencies
├── README.md                  # Detailed documentation and user guide
├── dataset/                   # Add your own class-organized training images
└── models/                    # Saved model artifacts & metrics
    ├── best_model.keras       # Trained Keras model checkpoint (HDF5/Keras 3)
    ├── class_names.json       # Exported class labels mapping
    └── confusion_matrix.png   # Test set evaluation confusion matrix
```

---

## 📊 Dataset Details

The original training images are not included in this repository. To retrain the model, provide a dataset organized in standard directory-based categorical format:

| Class | Item Types Included | Sample Count | Share |
|---|---|---|---|
| **Glass** | Clear & colored bottles, jars, glassware | ~501 | ~25.2% |
| **Metal** | Aluminium soda cans, food tins, aerosol cans | ~410 | ~20.6% |
| **Paper** | Cardboard boxes, office paper, newspapers, magazines | ~594 | ~29.9% |
| **Plastic** | PET beverage bottles, HDPE milk jugs, plastic wrappers | ~482 | ~24.3% |
| **Total** | Multi-class domestic waste | **~1,987** | **100%** |

### Data Splitting Strategy:
1. **85%** allocated to Training + Validation, **15%** strictly held out for Test evaluation.
2. The 85% partition is sub-divided into approximately **70% Training** and **15% Validation**.
3. Deterministic caching and fixed random seed (`seed=42`) ensure reproducible splits across runs.

---

## 🧠 Model Architecture & Training Pipeline

### Backbone
- **Base Architecture**: `MobileNetV2` (pre-trained on ImageNet).
- **Input Resolution**: `224 × 224 × 3` pixels.
- **Normalization**: Scaled to $[-1, 1]$ using standard MobileNetV2 preprocessing.

### Custom Classification Head
```text
Input (224, 224, 3)
  │
  ▼
Data Augmentation (Flip, Rotation ±15%, Zoom ±10%, Contrast ±10%)
  │
  ▼
MobileNetV2 Feature Extractor (154 layers)
  │
  ▼
Global Average Pooling 2D
  │
  ▼
Dropout (0.3)
  │
  ▼
Dense (128 units, ReLU activation)
  │
  ▼
Dropout (0.2)
  │
  ▼
Dense (4 units, Softmax activation)
```

### Two-Phase Training Strategy

| Parameter | Phase 1 (Head Warmup) | Phase 2 (Fine-Tuning) |
|---|---|---|
| **Backbone State** | Frozen (weights locked) | Top 30 layers unfrozen |
| **Optimizer** | Adam | Adam |
| **Learning Rate** | `1e-3` | `1e-5` |
| **Epochs** | Default: 20 | Default: +10 |
| **Loss Function** | Sparse Categorical Crossentropy | Sparse Categorical Crossentropy |
| **Batch Size** | 32 | 32 |
| **Callbacks** | `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau` | `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau` |

---

## 💻 Installation & Setup

### 1. Clone or Navigate to the Repository
```bash
git clone <repository-url>
cd smart-waste-classification
```

### 2. Create and Activate a Virtual Environment
Using a virtual environment is strongly recommended to isolate dependencies:

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**On Windows:**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Key Dependencies:
- `tensorflow>=2.13.0,<2.19.0`
- `streamlit>=1.28.0`
- `Pillow>=10.0.0`
- `numpy>=1.24.0`
- `scikit-learn>=1.3.0`
- `matplotlib>=3.7.0`

---

## 🚀 Usage Guide

### 1. Running the Web Application

To launch the interactive classification interface:

```bash
streamlit run app.py
```

Once started, the application will automatically open in your default browser at:
```text
Local URL: http://localhost:8501
Network URL: http://<your-local-ip>:8501
```

#### How to Use the UI:
1. Click **Browse files** or drag-and-drop a waste image (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`).
2. Verify the preview of your uploaded image.
3. Click the **🔍 Classify** button.
4. Review the prediction outcome, confidence rating, class probabilities bar breakdown, and disposal advice.

---

### 2. Training the Model from Scratch

If you wish to retrain the neural network on the current dataset or a custom dataset:

```bash
python train.py --data_dir dataset/
```

#### Custom Training Hyperparameters:
You can pass custom arguments to control epochs, batch size, and learning rates:

```bash
python train.py \
  --data_dir dataset/ \
  --epochs 25 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --fine_tune_epochs 15 \
  --fine_tune_lr 0.00001
```

#### CLI Options:
| Argument | Type | Default | Description |
|---|---|---|---|
| `--data_dir` | `str` | *Required* | Path to root directory containing class folders (`glass`, `metal`, `paper`, `plastic`). |
| `--epochs` | `int` | `20` | Maximum epochs for Phase 1 head training. |
| `--batch_size` | `int` | `32` | Number of samples per batch. |
| `--learning_rate` | `float` | `0.001` | Initial learning rate for classifier head. |
| `--fine_tune_lr` | `float` | `0.00001` | Reduced learning rate for Phase 2 fine-tuning. |
| `--fine_tune_epochs` | `int` | `10` | Extra epochs dedicated to fine-tuning top layers. |

Training automatically outputs:
- The best performing model checkpoint to `models/best_model.keras`.
- The class labels mapping to `models/class_names.json`.
- A publication-ready confusion matrix visualization to `models/confusion_matrix.png`.
- Detailed test metrics (precision, recall, F1-score) in the console.

---

## 🛡️ Safety & Confidence Thresholding

Standard classifiers force an argmax prediction even when completely uncertain. To make this tool practical and safe:

- **Confidence Threshold = 60% (`CONFIDENCE_THRESHOLD = 0.60`)**:
  - If `max(probabilities) >= 0.60`: The classification is accepted as high confidence. Disposal guidance is displayed.
  - If `max(probabilities) < 0.60`: The prediction is flagged as **Uncertain**. The user is warned, disposal guidance is withheld, and manual material inspection is recommended.

This failsafe prevents items like hazardous materials, ceramics, or multi-layer packaging from contaminating dedicated recycling bins.

---

## ♻️ Material Disposal & Recycling Guidance

| Material | Disposal Recommendation |
|---|---|
| **Glass** | Rinse containers clean and detach caps or corks. Place in the glass recycling bin. Never mix with ceramics, Pyrex, window panes, or mirrors. |
| **Metal** | Empty and rinse food tins and beverage cans. Crush cans to save collection volume. Aerosol cans must be fully emptied before binning. |
| **Paper** | Keep clean and dry. Flatten cardboard cartons. Remove plastic film wrap, window envelopes, and bubble mailers. Soiled or greasy pizza boxes belong in general waste/compost. |
| **Plastic** | Check the SPI resin code (Numbers 1-7). Codes #1 (PET) and #2 (HDPE) are standard curbside items. Rinse thoroughly; soft films/grocery bags usually require designated store drop-offs. |

---

## 🔧 Troubleshooting & FAQs

### Q1: `Model file not found at models/best_model.keras`
> **Solution**: You need to train the model first by running `python train.py --data_dir dataset/` or verify that the pre-trained `.keras` file exists in the `models/` directory.

### Q2: Double Normalization warning or low accuracy during inference
> **Solution**: `app.py` automatically checks whether the loaded model includes internal preprocessing layers before applying `keras.applications.mobilenet_v2.preprocess_input`. Ensure you keep `train.py` and `app.py` matching.

### Q3: Apple Silicon (M1/M2/M3) or GPU Acceleration
> **Tip**: TensorFlow on macOS uses `tensorflow-metal` for hardware acceleration. If you experience issues, standard CPU execution runs MobileNetV2 with low latency (<50ms per image).

### Q4: How do I change the confidence cutoff?
> **Solution**: Open `app.py` and modify line 16:
> ```python
> CONFIDENCE_THRESHOLD = 0.70  # Change to 70% or any desired threshold
> ```

---

## 🔮 Future Enhancements

- [ ] **Additional Categories**: Expand classes to Organic/Compost, Electronic waste (E-waste), Hazardous, and Textiles.
- [ ] **Live Camera Feed**: Integrate Streamlit's `st.camera_input` for immediate webcam snapping.
- [ ] **Edge Deployment**: Export model to **TensorFlow Lite (TFLite)** and **ONNX** formats for mobile and Raspberry Pi deployment.
- [ ] **Barcode & Resin Code OCR**: Integrate an optical reader to read embossed plastic resin numbers (#1 to #7) directly.

---

## 📄 License

This project is open-source and distributed under the [MIT License](LICENSE). Contributions, bug reports, and pull requests are welcome!


source .venv/bin/activate

streamlit run app.py