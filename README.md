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

#### 🌐 Option A: Modern Web GUI & Admin Dashboard
Launch the web interface featuring drag-and-drop upload, image preview, animated confidence meters, class probability bars, and a secure owner-only admin portal:

```bash
python app.py
```
- **Public Detector**: `http://127.0.0.1:5000/` (completely clean UI, no admin links visible)
- **Admin Portal**: `http://127.0.0.1:5000/admin` (password-protected, single-admin access)

---

## 🔐 Admin Portal & Security System

The project includes an **admin-only management portal** at `/admin` designed for the project owner to monitor user submissions, inspect predictions, manage dataset expansion, and analyze traffic metrics.

### 1. Key Features
- **Automatic Submission Logging**: Every user upload is stored under `uploads/` with a cryptographically randomized UUID filename (preventing filename collision and path manipulation). Metadata (UTC timestamp, prediction label, confidence score, class probabilities, client IP address, session ID, original filename, and file size) is recorded in SQLite (`admin_data.db`) via SQLAlchemy.
- **Admin Analytics Dashboard (`/admin`)**:
  - Live summary statistics: Total Submissions, Real vs AI % distribution, and Average Model Confidence.
  - Interactive Daily Submissions activity chart (Chart.js).
  - Search by filename or IP address.
  - Multi-criteria filtering by verdict (Real vs AI) and date range.
  - Sorting by newest, oldest, confidence, and file size.
  - Low-memory responsive pagination and lazy-loaded thumbnails.
- **User Feedback & Accuracy Auditing**:
  - **In-Session Prediction Feedback**: After an image is analyzed, users can vote whether the prediction was accurate (👍 Accurate / 👎 Inaccurate), submit ground-truth corrections (Authentic vs AI), and provide notes (e.g. generator used or visual artifacts).
  - **General Feedback & Ratings**: Users can submit overall experience ratings (1-5 stars ⭐), category tags (Model Accuracy, Feature Request, Bug Report), and comments.
  - **Admin Feedback Center**: The admin console includes an agreement KPI card, feedback-based table filtering (allowing instant identification of mispredicted images for retraining), and a dedicated User Voice Inbox.
- **Training Dataset Ingestion ("Move to Dataset")**: One-click action to transfer submitted images directly into `database/train/REAL` or `database/train/FAKE` for ongoing model fine-tuning and active learning.
- **Data Export & Deletion**: Export all submission logs as a formatted `.csv` file (`/admin/export-csv`) including all user feedback, claimed labels, and comments, or permanently purge records along with their stored image files.
- **Multi-User Concurrency**: Built-in threading synchronization locks inference passes so up to 10+ simultaneous users can analyze images concurrently without exceeding Render's 512 MB memory limit.

---

### 2. Setting Up Admin Credentials (`.env`)

Admin credentials and secrets are **never hardcoded** and are kept out of version control via `.gitignore`.

#### Step 1: Run the Interactive Password Generator
Run the built-in CLI tool to generate a cryptographically salted password hash (`scrypt` / `pbkdf2`) and a secure session key:

```bash
python generate_admin_hash.py
```
This utility prompts for your desired admin username and password, generates a 64-character random `SECRET_KEY`, and creates or updates your local `.env` file automatically:

```ini
SECRET_KEY=94b7e8d2...
ADMIN_USERNAME=admin
ADMIN_PASSWORD_HASH=scrypt:32768:8:1$7f3b...
MAX_CONTENT_LENGTH=10485760
```

#### Step 2: Accessing the Dashboard
1. Start the server:
   ```bash
   python app.py
   # Or with production Gunicorn:
   gunicorn app:app --workers 1 --threads 8 --timeout 120
   ```
2. Navigate to `http://127.0.0.1:5000/admin`.
3. You will be redirected to `/admin/login`. Enter your admin username and password to log in.

---

### 3. Security Hardening Measures
- **No Public Registration**: Single-operator admin access only. There is no sign-up or registration route.
- **Hidden Admin Footprint**: The public detector (`/`) has zero links or references to `/admin` or `/admin/login`.
- **Brute-Force Protection & Lockout**: Tracks consecutive failed login attempts per client IP. After **5 failed attempts**, the IP is locked out for **15 minutes**.
- **CSRF Defense**: All form submissions (login, logout, dataset movement, record deletion) are protected against Cross-Site Request Forgery via `Flask-WTF` CSRF tokens.
- **Private Image Storage**: The `uploads/` directory is isolated from public static file serving. Stored images and thumbnails can only be retrieved by authenticated admin sessions (`/admin/image/<uuid>` and `/admin/thumbnail/<uuid>`).
- **File Validation**: Enforces strict MIME and extension validation (JPEG, PNG, WebP), 10 MB payload limits, and Pillow image verification before saving.
- **Privacy Notice**: The public detector displays an explicit privacy notice informing users that submitted images are stored for security and quality evaluation purposes.

---

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
