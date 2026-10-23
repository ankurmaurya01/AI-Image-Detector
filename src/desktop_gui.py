"""
AI-Generated Image Detector - Native Desktop GUI (Tkinter)
Provides an OS desktop window with file browsing, image preview,
and real-time classification readout.
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import DetectorPredictor
from src.utils import load_config


class DetectorDesktopApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AI-Generated Image Detector")
        self.root.geometry("640x740")
        self.root.minsize(560, 680)
        self.root.configure(bg="#0f172a")

        self.selected_image_path = None
        self.predictor = None

        self._setup_styles()
        self._build_ui()
        self._load_model_async()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TProgressbar", thickness=12, troughcolor="#1e293b", background="#6366f1")

    def _build_ui(self):
        # Header Frame
        header = tk.Frame(self.root, bg="#0f172a", pady=15)
        header.pack(fill=tk.X)

        tk.Label(
            header,
            text="AI-Generated Image Detector",
            font=("Segoe UI", 18, "bold"),
            fg="#f8fafc",
            bg="#0f172a",
        ).pack()

        tk.Label(
            header,
            text="PyTorch ResNet50 Classifier",
            font=("Segoe UI", 10),
            fg="#94a3b8",
            bg="#0f172a",
        ).pack(pady=(2, 0))

        # Content Card
        card = tk.Frame(self.root, bg="#1e293b", padx=20, pady=20, relief=tk.FLAT)
        card.pack(fill=tk.BOTH, expand=True, padx=25, pady=(0, 20))

        # Canvas for Image Preview
        self.canvas_frame = tk.Frame(card, bg="#0b0f19", width=340, height=300)
        self.canvas_frame.pack(pady=10)
        self.canvas_frame.pack_propagate(False)

        self.image_label = tk.Label(
            self.canvas_frame,
            text="No image selected\nClick 'Choose Image' below",
            font=("Segoe UI", 11),
            fg="#64748b",
            bg="#0b0f19",
        )
        self.image_label.pack(expand=True)

        # File Path Label
        self.path_label = tk.Label(
            card,
            text="",
            font=("Segoe UI", 9),
            fg="#94a3b8",
            bg="#1e293b",
            wraplength=480,
        )
        self.path_label.pack(pady=4)

        # Button Group
        btn_frame = tk.Frame(card, bg="#1e293b")
        btn_frame.pack(pady=10)

        self.btn_browse = tk.Button(
            btn_frame,
            text="📁 Choose Image",
            font=("Segoe UI", 10, "bold"),
            bg="#334155",
            fg="white",
            activebackground="#475569",
            activeforeground="white",
            relief=tk.FLAT,
            padx=16,
            pady=8,
            cursor="hand2",
            command=self.choose_image,
        )
        self.btn_browse.grid(row=0, column=0, padx=8)

        self.btn_detect = tk.Button(
            btn_frame,
            text="⚡ Analyze Image",
            font=("Segoe UI", 10, "bold"),
            bg="#6366f1",
            fg="white",
            activebackground="#4f46e5",
            activeforeground="white",
            relief=tk.FLAT,
            padx=18,
            pady=8,
            cursor="hand2",
            state=tk.DISABLED,
            command=self.analyze_image,
        )
        self.btn_detect.grid(row=0, column=1, padx=8)

        # Status & Results Container
        self.result_frame = tk.Frame(card, bg="#0f172a", padx=16, pady=16, relief=tk.FLAT)
        self.result_frame.pack(fill=tk.X, pady=(10, 0))

        self.verdict_label = tk.Label(
            self.result_frame,
            text="Ready",
            font=("Segoe UI", 15, "bold"),
            fg="#94a3b8",
            bg="#0f172a",
        )
        self.verdict_label.pack()

        self.confidence_label = tk.Label(
            self.result_frame,
            text="",
            font=("Segoe UI", 11),
            fg="#cbd5e1",
            bg="#0f172a",
        )
        self.confidence_label.pack(pady=4)

        # Progress bar for probabilities
        self.prob_bar = ttk.Progressbar(self.result_frame, orient=tk.HORIZONTAL, mode="determinate")
        self.prob_bar.pack(fill=tk.X, pady=8)

        self.prob_detail = tk.Label(
            self.result_frame,
            text="",
            font=("Segoe UI", 9),
            fg="#64748b",
            bg="#0f172a",
        )
        self.prob_detail.pack()

    def _load_model_async(self):
        def _load():
            try:
                config = load_config("config.yaml")
                weights_path = config["paths"].get("best_model_path", "models/best_model.pth")
                self.predictor = DetectorPredictor(config=config, weights_path=weights_path)
                self.verdict_label.config(text="Model Loaded & Ready", fg="#34d399")
            except Exception as e:
                self.verdict_label.config(text="Model Load Failed", fg="#f43f5e")
                messagebox.showerror("Error", f"Failed to load model:\n{e}")

        threading.Thread(target=_load, daemon=True).start()

    def choose_image(self):
        path = filedialog.askopenfilename(
            title="Select an image to analyze",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp *.bmp *.tiff"), ("All files", "*.*")],
        )
        if not path:
            return

        self.selected_image_path = path
        self.path_label.config(text=os.path.basename(path))

        # Show thumbnail preview
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((320, 280))
            self.tk_img = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.tk_img, text="")
            self.btn_detect.config(state=tk.NORMAL)
            self.verdict_label.config(text="Click 'Analyze Image'", fg="#94a3b8")
            self.confidence_label.config(text="")
            self.prob_detail.config(text="")
            self.prob_bar["value"] = 0
        except Exception as e:
            messagebox.showerror("Error", f"Could not load preview: {e}")

    def analyze_image(self):
        if not self.selected_image_path or not self.predictor:
            return

        self.btn_detect.config(state=tk.DISABLED, text="Analyzing...")
        self.verdict_label.config(text="Processing...", fg="#cbd5e1")

        def _predict():
            try:
                result = self.predictor.predict(self.selected_image_path)
                label = result["label"]
                conf = result["confidence"]
                probs = result["probabilities"]

                is_real = label == "Real"
                color = "#34d399" if is_real else "#fb7185"

                self.root.after(0, lambda: self._show_result(label, conf, probs, color, is_real))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Prediction Error", str(e)))
            finally:
                self.root.after(0, lambda: self.btn_detect.config(state=tk.NORMAL, text="⚡ Analyze Image"))

        threading.Thread(target=_predict, daemon=True).start()

    def _show_result(self, label, conf, probs, color, is_real):
        status_text = f"✔ Authentic (Real Photo)" if is_real else "⚠ AI-Generated Image"
        self.verdict_label.config(text=status_text, fg=color)
        self.confidence_label.config(text=f"Confidence: {conf:.2f}%", fg="#f8fafc")

        real_p = probs.get("Real", 0.0)
        fake_p = probs.get("AI-generated", 0.0)
        self.prob_bar["value"] = real_p if is_real else fake_p

        self.prob_detail.config(
            text=f"Probabilities:  Real = {real_p:.1f}%  |  AI-Generated = {fake_p:.1f}%"
        )


def main():
    root = tk.Tk()
    app = DetectorDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
