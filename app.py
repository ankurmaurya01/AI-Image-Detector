"""
AI-Generated Image Detector - Web GUI Application
Flask web application compatible with local execution and cloud serverless platforms (Vercel).
"""

import base64
import gc
import io
import json
import os
import sys
import tempfile
import webbrowser
from flask import Flask, request, jsonify, render_template_string

# Restrict multithreading to stay under 512MB RAM on Render
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Add current directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.utils import load_config, get_device
from src.predict import DetectorPredictor
from src.gradcam import GradCAM
from src.data_loader import safe_image_loader, get_transforms
from src.model import build_model
import torch
import torch.nn.functional as F
import urllib.request

torch.set_num_threads(1)

PORT = int(os.environ.get("PORT", 5000))
HOST = "0.0.0.0"

config = load_config("config.yaml")
weights_path = config["paths"].get("best_model_path", "models/best_model.pth")
MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://github.com/ankurmaurya01/AI-Image-Detector/releases/download/v1.0.0/best_model.pth"
)

# Global model references (loaded lazily to ensure immediate port binding)
_predictor = None
_gradcam = None

def get_model():
    global _predictor, _gradcam
    if _predictor is not None:
        return _predictor, _gradcam

    torch.set_num_threads(1)

    if not os.path.exists(weights_path) and MODEL_URL:
        try:
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            print(f"[INFO] Attempting to download weights from {MODEL_URL}...")
            urllib.request.urlretrieve(MODEL_URL, weights_path)
            if os.path.exists(weights_path) and os.path.getsize(weights_path) < 1000000:
                os.remove(weights_path)
            else:
                print("[INFO] Model weights downloaded successfully.")
        except Exception as dl_err:
            print(f"[INFO] Model weights not downloaded via URL ({dl_err}).")

    try:
        if os.path.exists(weights_path):
            _predictor = DetectorPredictor(config=config, weights_path=weights_path)
            _gradcam = GradCAM(_predictor.model)
            print(f"[INFO] Loaded trained model & Grad-CAM from {weights_path}")
        else:
            raise FileNotFoundError(f"Checkpoint not found at '{weights_path}'")
    except Exception as e:
        print(f"[WARNING] Could not load checkpoint ({e}). Initializing fallback demo mode...")
        
        class FallbackPredictor:
            def __init__(self, cfg):
                self.config = cfg
                self.device = get_device("auto")
                self.model = build_model(cfg).to(self.device)
                self.model.eval()
                img_size = cfg.get("dataset", {}).get("image_size", 224)
                self.transform = get_transforms(split="test", image_size=img_size)

            def predict(self, image_path):
                pil_img = safe_image_loader(image_path)
                tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    outputs = self.model(tensor)
                    probs = F.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
                real_prob = float(probs[0]) * 100
                fake_prob = float(probs[1]) * 100
                label = "AI-generated" if fake_prob >= 50.0 else "Real"
                confidence = fake_prob if label == "AI-generated" else real_prob
                return {
                    "label": label,
                    "confidence": round(confidence, 2),
                    "probabilities": {
                        "Real": round(real_prob, 2),
                        "AI-generated": round(fake_prob, 2),
                    }
                }

        _predictor = FallbackPredictor(config)
        _gradcam = GradCAM(_predictor.model)
        print("[INFO] Fallback model & Grad-CAM initialized successfully.")

    gc.collect()
    return _predictor, _gradcam



HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Image Detector</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0f19;
      --card: #131b2e;
      --border: #1e293b;
      --text: #f8fafc;
      --muted: #94a3b8;
      --primary: #6366f1;
      --primary-hover: #4f46e5;
      --real: #10b981;
      --fake: #f43f5e;
      --ring: rgba(99, 102, 241, 0.25);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }
    body {
      background: radial-gradient(circle at 50% 0%, #172554 0%, var(--bg) 75%);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 40px 20px;
    }
    .container {
      width: 100%;
      max-width: 860px;
      display: flex;
      flex-direction: column;
      gap: 24px;
    }
    header {
      text-align: center;
      margin-bottom: 8px;
    }
    .badge {
      display: inline-block;
      padding: 6px 14px;
      border-radius: 9999px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      background: rgba(99, 102, 241, 0.15);
      color: #818cf8;
      border: 1px solid rgba(99, 102, 241, 0.3);
      margin-bottom: 12px;
    }
    h1 {
      font-size: 32px;
      font-weight: 800;
      letter-spacing: -0.02em;
      background: linear-gradient(135deg, #fff 0%, #cbd5e1 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 8px;
    }
    p.subtitle {
      color: var(--muted);
      font-size: 15px;
    }
    .main-card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 32px;
      box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
    }
    .dropzone {
      border: 2px dashed #334155;
      border-radius: 16px;
      padding: 40px 20px;
      text-align: center;
      cursor: pointer;
      transition: all 0.2s ease;
      background: rgba(15, 23, 42, 0.4);
      position: relative;
    }
    .dropzone:hover, .dropzone.dragover {
      border-color: var(--primary);
      background: rgba(99, 102, 241, 0.08);
      transform: translateY(-2px);
    }
    .dropzone-icon {
      font-size: 42px;
      margin-bottom: 12px;
    }
    .dropzone-text {
      font-size: 16px;
      font-weight: 600;
      color: var(--text);
      margin-bottom: 6px;
    }
    .dropzone-sub {
      font-size: 13px;
      color: var(--muted);
    }
    input[type="file"] {
      display: none;
    }
    .preview-section {
      display: none;
      flex-direction: column;
      align-items: center;
      margin-top: 24px;
      gap: 20px;
    }
    .preview-wrapper {
      position: relative;
      max-width: 360px;
      max-height: 360px;
      border-radius: 14px;
      overflow: hidden;
      border: 1px solid #334155;
      box-shadow: 0 10px 25px rgba(0,0,0,0.3);
    }
    .preview-wrapper img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }
    .btn-group {
      display: flex;
      gap: 12px;
    }
    button {
      padding: 12px 24px;
      border-radius: 12px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
      border: none;
    }
    .btn-primary {
      background: var(--primary);
      color: white;
      box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4);
    }
    .btn-primary:hover {
      background: var(--primary-hover);
      transform: translateY(-1px);
    }
    .btn-secondary {
      background: #1e293b;
      color: var(--muted);
    }
    .btn-secondary:hover {
      background: #334155;
      color: white;
    }
    .loading-spinner {
      display: none;
      align-items: center;
      gap: 10px;
      color: #cbd5e1;
      font-size: 14px;
      margin-top: 16px;
    }
    .spinner {
      width: 20px;
      height: 20px;
      border: 3px solid rgba(255,255,255,0.2);
      border-top-color: var(--primary);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .result-card {
      display: none;
      margin-top: 28px;
      border-radius: 16px;
      padding: 24px;
      border: 1px solid #1e293b;
      background: rgba(15, 23, 42, 0.7);
      animation: fadeIn 0.3s ease;
    }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
    .result-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 20px;
    }
    .verdict-tag {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 18px;
      border-radius: 10px;
      font-size: 18px;
      font-weight: 800;
      letter-spacing: -0.01em;
    }
    .verdict-real {
      background: rgba(16, 185, 129, 0.15);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .verdict-fake {
      background: rgba(244, 63, 94, 0.15);
      color: #fb7185;
      border: 1px solid rgba(244, 63, 94, 0.3);
    }
    .confidence-value {
      font-size: 28px;
      font-weight: 800;
      color: var(--text);
    }
    .bar-group {
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .bar-item {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .bar-label {
      display: flex;
      justify-content: space-between;
      font-size: 13px;
      font-weight: 600;
    }
    .bar-track {
      height: 10px;
      background: #1e293b;
      border-radius: 9999px;
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      border-radius: 9999px;
      transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .fill-real { background: linear-gradient(90deg, #059669, #10b981); }
    .fill-fake { background: linear-gradient(90deg, #e11d48, #f43f5e); }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="badge">PyTorch Deep Learning</div>
      <h1>AI-Generated Image Detector</h1>
      <p class="subtitle">Upload any image to verify if it is authentic or synthetically generated by AI</p>
    </header>

    <div class="main-card">
      <div class="dropzone" id="dropzone" onclick="document.getElementById('fileInput').click()">
        <div class="dropzone-icon">🖼️</div>
        <div class="dropzone-text">Click to choose or drag & drop an image</div>
        <div class="dropzone-sub">Supports JPG, PNG, WEBP, BMP</div>
        <input type="file" id="fileInput" accept="image/*" onchange="handleFile(this.files[0])">
      </div>

      <div class="preview-section" id="previewSection">
        <div class="preview-wrapper">
          <img id="previewImg" src="" alt="Preview">
        </div>
        <div class="btn-group">
          <button class="btn-primary" id="analyzeBtn" onclick="analyzeImage()">Analyze Image</button>
          <button class="btn-secondary" onclick="resetUpload()">Upload Another</button>
        </div>
        <div class="loading-spinner" id="spinner">
          <div class="spinner"></div>
          <span>Processing with ResNet50 neural network...</span>
        </div>
      </div>

      <div class="result-card" id="resultCard">
        <div class="result-header">
          <div>
            <div style="font-size: 12px; color: var(--muted); margin-bottom: 4px; text-transform: uppercase; font-weight: 700;">Detection Verdict</div>
            <div class="verdict-tag" id="verdictBadge">AI-Generated</div>
          </div>
          <div style="text-align: right;">
            <div style="font-size: 12px; color: var(--muted); margin-bottom: 4px; text-transform: uppercase; font-weight: 700;">Confidence</div>
            <div class="confidence-value" id="confidenceVal">98.5%</div>
          </div>
        </div>

        <div class="bar-group">
          <div class="bar-item">
            <div class="bar-label">
              <span>Authentic (Real Photo)</span>
              <span id="realProb">1.5%</span>
            </div>
            <div class="bar-track">
              <div class="bar-fill fill-real" id="realBar" style="width: 1.5%;"></div>
            </div>
          </div>
          <div class="bar-item">
            <div class="bar-label">
              <span>AI-Generated (Synthetic)</span>
              <span id="fakeProb">98.5%</span>
            </div>
            <div class="bar-track">
              <div class="bar-fill fill-fake" id="fakeBar" style="width: 98.5%;"></div>
            </div>
          </div>
        </div>

        <div id="gradcamContainer" style="display: none; margin-top: 24px; padding-top: 20px; border-top: 1px solid #1e293b;">
          <div style="font-size: 13px; font-weight: 700; color: #cbd5e1; margin-bottom: 6px; display: flex; align-items: center; gap: 6px;">
            <span>🔬 Explainable AI (Grad-CAM Heatmap)</span>
          </div>
          <p style="font-size: 12px; color: var(--muted); margin-bottom: 14px; line-height: 1.5;">
            Visualizes where the neural network focused. As detailed in the CIFAKE research paper, authentic images trigger holistic features, whereas AI-generated images trigger localized background defects and texture artifacts.
          </p>
          <div style="display: flex; justify-content: center;">
            <div style="max-width: 320px; border-radius: 12px; overflow: hidden; border: 1px solid #334155; box-shadow: 0 8px 20px rgba(0,0,0,0.3);">
              <img id="gradcamImg" src="" alt="Grad-CAM Heatmap" style="width: 100%; display: block;">
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    let selectedFile = null;
    const dropzone = document.getElementById('dropzone');

    ['dragenter', 'dragover'].forEach(name => {
      dropzone.addEventListener(name, (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
    });
    ['dragleave', 'drop'].forEach(name => {
      dropzone.addEventListener(name, (e) => { e.preventDefault(); dropzone.classList.remove('dragover'); });
    });
    dropzone.addEventListener('drop', (e) => {
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFile(e.dataTransfer.files[0]);
      }
    });

    function handleFile(file) {
      if (!file || !file.type.startsWith('image/')) {
        alert('Please upload a valid image file.');
        return;
      }
      selectedFile = file;
      const reader = new FileReader();
      reader.onload = (e) => {
        document.getElementById('previewImg').src = e.target.result;
        dropzone.style.display = 'none';
        document.getElementById('previewSection').style.display = 'flex';
        document.getElementById('resultCard').style.display = 'none';
      };
      reader.readAsDataURL(file);
    }

    function resetUpload() {
      selectedFile = null;
      document.getElementById('fileInput').value = '';
      dropzone.style.display = 'block';
      document.getElementById('previewSection').style.display = 'none';
      document.getElementById('resultCard').style.display = 'none';
    }

    async function analyzeImage() {
      if (!selectedFile) return;

      const btn = document.getElementById('analyzeBtn');
      const spinner = document.getElementById('spinner');
      btn.disabled = true;
      spinner.style.display = 'flex';
      document.getElementById('resultCard').style.display = 'none';

      try {
        const formData = new FormData();
        formData.append('image', selectedFile);

        const res = await fetch('/predict', {
          method: 'POST',
          body: formData
        });
        const data = await res.json();

        if (data.error) {
          alert('Error: ' + data.error);
          return;
        }

        const isReal = data.label === 'Real';
        const badge = document.getElementById('verdictBadge');
        badge.textContent = isReal ? 'Authentic Real' : 'AI-Generated';
        badge.className = 'verdict-tag ' + (isReal ? 'verdict-real' : 'verdict-fake');

        document.getElementById('confidenceVal').textContent = data.confidence.toFixed(1) + '%';
        
        const realPct = data.probabilities['Real'] || 0;
        const fakePct = data.probabilities['AI-generated'] || 0;

        document.getElementById('realProb').textContent = realPct.toFixed(1) + '%';
        document.getElementById('realBar').style.width = realPct + '%';

        document.getElementById('fakeProb').textContent = fakePct.toFixed(1) + '%';
        document.getElementById('fakeBar').style.width = fakePct + '%';

        if (data.gradcam_b64) {
          document.getElementById('gradcamImg').src = data.gradcam_b64;
          document.getElementById('gradcamContainer').style.display = 'block';
        } else {
          document.getElementById('gradcamContainer').style.display = 'none';
        }

        document.getElementById('resultCard').style.display = 'block';
      } catch (err) {
        alert('Analysis failed: ' + err.message);
      } finally {
        btn.disabled = false;
        spinner.style.display = 'none';
      }
    }
  </script>
</body>
</html>
"""

app = Flask(__name__)
application = app  # WSGI compatibility alias

@app.route("/", methods=["GET"])
@app.route("/index.html", methods=["GET"])
def home():
    return render_template_string(HTML_PAGE)

@app.route("/predict", methods=["POST"])
def predict():
    pred, cam = get_model()
    if pred is None:
        return jsonify({"error": "Model initialization failed."}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded in request."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = pred.predict(tmp_path)

        # Generate Grad-CAM Explainable AI Heatmap
        if cam is not None:
            try:
                pil_img = safe_image_loader(tmp_path)
                tensor = pred.transform(pil_img).unsqueeze(0).to(pred.device)
                heatmap = cam.generate_heatmap(tensor)
                overlay = cam.overlay_heatmap(pil_img, heatmap, colormap_name="jet", alpha=0.55)

                buf = io.BytesIO()
                overlay.save(buf, format="JPEG", quality=90)
                result["gradcam_b64"] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception as ge:
                print(f"[WARNING] Grad-CAM generation error: {ge}")

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


if __name__ == "__main__":
    url = f"http://{HOST}:{PORT}"
    print("\n" + "=" * 60)
    print(f" 🚀 AI-Generated Image Detector Web GUI is LIVE!")
    print(f" 👉 URL: {url}")
    print(f" Press Ctrl+C in this terminal to stop the server.")
    print("=" * 60 + "\n")

    try:
        webbrowser.open(url)
    except Exception:
        pass

    app.run(host=HOST, port=PORT, debug=False)

