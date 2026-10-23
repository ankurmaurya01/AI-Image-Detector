# 🔍 AI-Generated Image Detector

A deep learning classifier built with **PyTorch** to distinguish between authentic (real) photographs and AI-generated (synthetic / fake) images.

---

## 📁 Project Directory Structure

```text
ai-image-detector/
│
├── database/                   # Dataset root (binary classification)
│   ├── train/                  # Training set
│   │   ├── real/               # Authentic training images
│   │   └── fake/               # AI-generated training images
│   ├── valid/                  # Validation set (evaluated after each epoch)
│   │   ├── real/               # Authentic validation images
│   │   └── fake/               # AI-generated validation images
│   └── test/                   # Test set (unseen holdout for final evaluation)
│       ├── real/               # Authentic test images
│       └── fake/               # AI-generated test images
│
├── models/                     # Saved model checkpoints & training curves
│   └── .gitkeep
├── notebooks/                  # Jupyter experimentation & visualization
│   └── experimentation.ipynb
├── src/                        # Modular source code
│   ├── __init__.py
│   ├── data_loader.py          # Dataset loading, augmentations & robust image decoding
│   ├── model.py                # CNN architectures (ResNet / EfficientNet backbones)
│   ├── train.py                # Training loop, validation, metrics & checkpointing
│   ├── evaluate.py             # Evaluation: Accuracy, Precision, Recall, F1 & Confusion Matrix
│   ├── predict.py              # Single image inference CLI (Real vs AI-generated)
│   └── utils.py                # Config loader, device selector, seeding & plotting
├── requirements.txt            # Python dependencies
├── config.yaml                 # Central hyperparameter & path configuration
└── README.md                   # Complete documentation
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.10 or higher
- NVIDIA GPU with CUDA support recommended (CPU is automatically supported as fallback)

### 2. Set Up a Virtual Environment

Open your terminal in the project directory:

```bash
cd ai-image-detector

# Create a virtual environment
python -m venv venv

# Activate virtual environment:
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Windows (Command Prompt):
venv\Scripts\activate.bat
# On Linux / macOS:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **PyTorch with CUDA**: If you want GPU acceleration, install the CUDA-enabled build of PyTorch according to your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/):
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

---

## 🖼️ Populating the Dataset

Place your images into their respective folders inside `database/`:

- `database/train/real/`: Authentic photos for training
- `database/train/fake/`: AI-generated images (e.g. Midjourney, Stable Diffusion, DALL-E, Flux) for training
- `database/valid/real/`: Authentic photos for epoch-by-epoch validation
- `database/valid/fake/`: AI-generated images for validation
- `database/test/real/`: Authentic photos for final evaluation
- `database/test/fake/`: AI-generated images for final evaluation

**Supported Image Formats**: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tiff`.  
The data loader automatically converts all images (including RGBA and grayscale) to 3-channel RGB.

---

## 🛠️ Configuration (`config.yaml`)

All hyperparameters, paths, and settings are managed in `config.yaml`:

```yaml
paths:
  train_dir: "database/train"
  valid_dir: "database/valid"
  test_dir: "database/test"
  model_save_dir: "models"
  best_model_path: "models/best_model.pth"
  history_save_path: "models/training_history.json"
  confusion_matrix_path: "models/confusion_matrix.png"

dataset:
  classes: ["real", "fake"]
  class_to_idx:
    real: 0
    fake: 1
  image_size: 224
  batch_size: 32
  num_workers: 2

augmentation:
  random_horizontal_flip_prob: 0.5
  random_rotation_degrees: 15
  color_jitter:
    brightness: 0.1
    contrast: 0.1
    saturation: 0.1
    hue: 0.05

model:
  backbone: "resnet50"       # Options: resnet18, resnet34, resnet50, efficientnet_b0, efficientnet_b2
  pretrained: true           # true = Transfer Learning (ImageNet weights), false = Train from scratch
  num_classes: 2
  dropout_rate: 0.3

training:
  num_epochs: 20
  learning_rate: 0.0001
  weight_decay: 0.0001
  optimizer: "adamw"          # Options: adam, adamw, sgd
  scheduler: "cosine"        # Options: cosine, steplr, none
  step_size: 7
  gamma: 0.1
  seed: 42
  device: "auto"             # Options: auto, cuda, cpu
```

---

## 🚀 Usage Guide

### 1. Training the Model

To start training with the default settings in `config.yaml`:

```bash
python src/train.py
```

#### Optional CLI Overrides:
You can override configuration parameters directly from the command line:

```bash
# Train with custom epochs, learning rate, and batch size
python src/train.py --epochs 25 --lr 0.0002 --batch-size 16

# Train using an EfficientNet-B0 backbone
python src/train.py --backbone efficientnet_b0 --epochs 15
```

**Training Features**:
- Evaluates on the validation set after every epoch.
- Automatically saves the best model checkpoint to `models/best_model.pth` whenever validation accuracy improves.
- Saves epoch loss and accuracy history to `models/training_history.json`.
- Plots and saves training/validation curves to `models/training_curves.png`.

---

### 2. Evaluating the Model

After training, evaluate the model on unseen images in `database/test`:

```bash
python src/evaluate.py
```

You can optionally specify a custom weights file or config:

```bash
python src/evaluate.py --weights models/best_model.pth --config config.yaml
```

**Output Metrics**:
- Overall Accuracy
- Macro & Per-class Precision, Recall, and F1-Score
- Formatted Scikit-Learn Classification Report
- Formatted Confusion Matrix
- Saves high-resolution confusion matrix heatmap to `models/confusion_matrix.png`
- Saves detailed JSON metrics to `models/evaluation_metrics.json`

---

### 3. Predicting on a Single Image

To classify a new image:

```bash
python src/predict.py --image path/to/example.jpg
```

#### Output Example:
```text
==================================================
           DETECTION RESULT
==================================================
 Image       : samples/portrait.jpg
 Prediction  : AI-generated
 Confidence  : 97.42%
--------------------------------------------------
 Probabilities Breakdown:
   - Real          : 2.58%
   - AI-generated  : 97.42%
==================================================
```

#### Programmatic Python Usage:
```python
from src.utils import load_config
from src.predict import DetectorPredictor

config = load_config("config.yaml")
predictor = DetectorPredictor(config=config, weights_path="models/best_model.pth")

result = predictor.predict("path/to/my_image.png")
print(result["label"])       # "Real" or "AI-generated"
print(result["confidence"])  # e.g. 98.5
```

---

### 4. Graphical User Interfaces (GUIs)

Two graphical interface options are included out of the box with zero extra dependencies:

#### 🌐 Option A: Modern Web GUI (Browser App)
Launch a local interactive web interface featuring drag-and-drop upload, image preview, animated confidence meters, and class probability bars:

```bash
python app.py
```
This automatically opens your default browser at `http://127.0.0.1:5000`.

#### 🖥️ Option B: Native Desktop GUI (Tkinter Window)
Launch an OS desktop window with a native file picker, preview display, and instant detection badges:

```bash
python src/desktop_gui.py
```

---

## 📓 Interactive Experimentation

Launch Jupyter to explore the dataset and test architectures interactively:

```bash
jupyter notebook notebooks/experimentation.ipynb
```

The notebook guides you through:
1. Inspecting loaded configurations.
2. Visualizing data augmentation effects on images.
3. Instantiating ResNet / EfficientNet models and inspecting parameter counts.
4. Performing dummy tensor sanity checks.

---

## 💡 Practical Recommendations

1. **Balanced Datasets**: Aim for roughly equal numbers of real and fake images in both train and validation sets.
2. **Diverse Generators**: Train on images generated by multiple AI models (Midjourney, DALL-E 3, Stable Diffusion XL, Flux) to ensure the detector generalizes rather than memorizing single-generator artifacts.
3. **Resolution Consistency**: The default input size is `224x224`. If training on high-resolution images, you can test higher dimensions (e.g. `384` or `512`) in `config.yaml` if GPU memory allows.
