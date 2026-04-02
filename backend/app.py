# =============================================================================
# backend/app.py
# Flask REST API for the XAI System
#
# Flow:
#   Browser POST /api/predict
#     -> Load classifier (AnimalClassifier / BrainTumor / Caltech)
#     -> Load model weights from outputs/
#     -> Inference: image -> prediction
#     -> XAI: image -> heatmap
#     -> Return JSON (base64 images + metrics)
# =============================================================================

# -----------------------------------------------------------------------------
# [1] IMPORTS
# -----------------------------------------------------------------------------
import sys
import traceback
import time
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import matplotlib
matplotlib.use("Agg")   # use non-interactive backend; avoids GUI crash on server
import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from torchvision import transforms

# -----------------------------------------------------------------------------
# [2] PATHS & CONSTANTS
# -----------------------------------------------------------------------------
BASE_DIR    = Path(__file__).resolve().parent.parent   # IPEM/
OUTPUTS_DIR = BASE_DIR / "outputs"
DATA_DIR    = BASE_DIR / "data"
UPLOAD_DIR  = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Add src/ to Python path so we can import existing modules without moving them
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

# Import XAI explanation functions from existing code (sample_xAI.py)
# Must be done AFTER sys.path is updated
from sample_xAI import (
    explain_with_lime,
    explain_with_shap,
    explain_with_gradcam,
    explain_with_rise,
    explain_with_ipem,
)

IMG_SIZE = 224
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Standard ImageNet normalization — same as used during training
TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# -----------------------------------------------------------------------------
# [3] HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def pil_to_base64(img: Image.Image) -> str:
    """
    Convert a PIL Image to a base64-encoded PNG string.

    Why base64?
      JSON only carries text, not binary. base64 encodes image bytes
      as ASCII so they can be embedded in JSON and used directly as:
        <img src="data:image/png;base64,...">
    """
    buf = io.BytesIO()         # in-memory buffer (no disk I/O)
    img.save(buf, format="PNG")
    buf.seek(0)                # rewind to the beginning
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def numpy_to_base64(arr: np.ndarray) -> str:
    """Convert a numpy array (H, W, 3) uint8 to a base64 PNG string."""
    return pil_to_base64(Image.fromarray(arr.astype(np.uint8)))


def heatmap_overlay_to_base64(
    org_img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    cmap: str = "jet"
) -> str:
    """
    Blend the original image with a colormap-applied heatmap and encode as base64.

    Steps:
      1. Resize heatmap to match original image dimensions
      2. Normalize heatmap to [0, 1]
      3. Apply matplotlib colormap (jet: blue -> yellow -> red)
      4. Blend with original image using alpha transparency
      5. Encode to base64
    """
    H, W = org_img.shape[:2]

    # Resize heatmap to image size
    hm = cv2.resize(heatmap.astype(np.float32), (W, H))

    # Normalize to [0, 1]
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)

    # Apply colormap; drop alpha channel (last column)
    colormap   = plt.get_cmap(cmap)
    hm_colored = (colormap(hm)[:, :, :3] * 255).astype(np.uint8)

    # Weighted blend: original * (1-alpha) + heatmap * alpha
    blended = (org_img * (1 - alpha) + hm_colored * alpha).astype(np.uint8)
    return numpy_to_base64(blended)


def load_classifier(dataset: str, model_name: str):
    """
    Instantiate the correct classifier for the given dataset and load
    pre-trained weights from outputs/{dataset}/{model_name}_best_model.pt.

    Note: do NOT put load_trained_model() inside the if/elif branches —
    that would make it dead code (unreachable after a return statement).
    """
    from animal_classifier import AnimalImageClassifier
    from brain_tumor_classifier import BrainTumorClassifier
    from caltech_classifier import CaltechImageClassifier

    kwargs = dict(
        data_dir=str(DATA_DIR),
        output_dir=str(OUTPUTS_DIR),
        args_model=model_name,
    )

    # Select the right classifier class
    if dataset == "animals":
        clf = AnimalImageClassifier(**kwargs)
    elif dataset == "brain-tumor":
        clf = BrainTumorClassifier(**kwargs)
    elif dataset == "caltech-101":
        clf = CaltechImageClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: '{dataset}'")

    # Load saved weights and switch to eval mode
    clf.load_trained_model()
    clf.model.eval()
    return clf

# -----------------------------------------------------------------------------
# [4] FLASK APP & ROUTES
# -----------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)   # allow React (port 5173) to call Flask (port 5000)


@app.route("/api/health", methods=["GET"])
def health():
    """Health check — returns server status and active device (cpu/cuda)."""
    return jsonify({"status": "ok", "device": str(device)})


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Main endpoint: receive an image + config, run inference + XAI, return results.

    Request (multipart/form-data):
      image      : image file (JPG / PNG)
      dataset    : 'animals' | 'brain-tumor' | 'caltech-101'
      model_name : 'efficientnet_b3' | 'resnet50' | 'transformer'
      xai_method : 'lime' | 'shap' | 'gradcam' | 'rise' | 'ipem'

    Response (JSON):
      {
        "prediction": { "class_name", "confidence", "top5" },
        "images":     { "original", "heatmap", "perturbed" },  // base64 PNG
        "metrics":    { "explanation_time" }
      }
    """
    try:
        # ── 1. Parse request ──────────────────────────────────────────────────
        if "image" not in request.files:
            return jsonify({"error": "Missing image file"}), 400

        image_file = request.files["image"]
        dataset    = request.form.get("dataset",    "animals")
        model_name = request.form.get("model_name", "efficientnet_b3")
        xai_method = request.form.get("xai_method", "gradcam")

        # GradCAM requires Conv2D layers — incompatible with ViT (Transformer)
        if xai_method == "gradcam" and model_name == "transformer":
            return jsonify({
                "error": "GradCAM is not compatible with the Transformer model. "
                         "Please choose LIME, SHAP, RISE, or IPEM instead."
            }), 400

        # ── 2. Preprocess image ───────────────────────────────────────────────
        pil_img     = Image.open(image_file.stream).convert("RGB")
        pil_resized = pil_img.resize((IMG_SIZE, IMG_SIZE))
        org_img_np  = np.array(pil_resized)                           # (H,W,3) uint8
        img_tensor  = TRANSFORM(pil_resized).unsqueeze(0).to(device)  # (1,C,H,W) normalized

        # img_np for LIME: raw float [0,1] HWC array (LIME's predict_fn handles its own normalization)
        # Do NOT use the ImageNet-normalized tensor here — that would have values in [-2, 2]
        img_np = np.array(pil_resized).astype(np.float32) / 255.0

        # ── 3. Load classifier & model weights ───────────────────────────────
        clf         = load_classifier(dataset, model_name)
        class_names = clf.class_names

        # ── 4. Inference: image -> class probabilities ────────────────────────
        with torch.no_grad():
            logits = clf.model(img_tensor)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]  # shape: (num_classes,)

        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

        # Build top-5 list (or fewer if the dataset has < 5 classes)
        top_k   = min(5, len(class_names))
        top_idx = np.argsort(probs)[::-1][:top_k]
        top5    = [[class_names[i], round(float(probs[i]), 4)] for i in top_idx]

        # ── 5. XAI: image -> saliency heatmap ────────────────────────────────
        # All explain_with_*() functions share the same signature:
        #   (clf, img, class_names, output_dir, args_dataset, org_img)
        # output_dir / args_dataset are only used to save debug images; we use
        # UPLOAD_DIR as a temporary scratch directory.
        tmp_dir = str(UPLOAD_DIR)
        start   = time.time()

        if xai_method == "lime":
            # LIME expects raw (H,W,C) float numpy, NOT ImageNet-normalized
            _, heatmap = explain_with_lime(
                clf, img_np, class_names,
                output_dir=tmp_dir, args_dataset="tmp", org_img=pil_resized
            )

        elif xai_method == "shap":
            # SHAP works on the normalized tensor (1,C,H,W)
            _, heatmap = explain_with_shap(
                clf, img_tensor, class_names,
                output_dir=tmp_dir, args_dataset="tmp", org_img=pil_resized
            )

        elif xai_method == "gradcam":
            # GradCAM returns the heatmap directly (not a tuple)
            heatmap = explain_with_gradcam(
                clf, img_tensor, class_names,
                output_dir=tmp_dir, args_dataset="tmp", org_img=pil_resized
            )

        elif xai_method == "rise":
            heatmap = explain_with_rise(
                clf, img_tensor, class_names,
                output_dir=tmp_dir, args_dataset="tmp", org_img=pil_resized
            )

        elif xai_method == "ipem":
            heatmap = explain_with_ipem(
                clf, img_tensor, class_names,
                output_dir=tmp_dir, args_dataset="tmp", org_img=pil_resized
            )

        xai_time = round(time.time() - start, 2)

        # ── 6. Build visualization images ─────────────────────────────────────
        original_b64 = numpy_to_base64(org_img_np)
        heatmap_b64  = heatmap_overlay_to_base64(org_img_np, heatmap)

        # Perturbed image: Gaussian-blur the most salient region (heatmap >= 0.7)
        H, W       = org_img_np.shape[:2]
        hm_resized = cv2.resize(heatmap.astype(np.float32), (W, H))
        hm_norm    = (hm_resized - hm_resized.min()) / (hm_resized.max() - hm_resized.min() + 1e-8)
        mask       = (hm_norm >= 0.7).astype(np.uint8)
        blurred    = cv2.GaussianBlur(org_img_np, (11, 11), sigmaX=50, sigmaY=50)
        mask_3d    = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        perturbed  = org_img_np.copy()
        perturbed[mask_3d == 1] = blurred[mask_3d == 1]
        perturbed_b64 = numpy_to_base64(perturbed)

        # ── 7. Return JSON response ───────────────────────────────────────────
        return jsonify({
            "prediction": {
                "class_name": class_names[pred_class],
                "confidence": round(confidence, 4),
                "top5":       top5,
            },
            "images": {
                "original":  original_b64,
                "heatmap":   heatmap_b64,
                "perturbed": perturbed_b64,
            },
            "metrics": {
                "explanation_time": xai_time,
            },
        })

    except FileNotFoundError as e:
        # Typically means the model has not been trained yet (no .pt file)
        return jsonify({"error": f"Model weights not found: {e}"}), 404

    except Exception as e:
        traceback.print_exc()   # print full traceback to terminal for debugging
        return jsonify({"error": str(e)}), 500

# -----------------------------------------------------------------------------
# [5] ENTRY POINT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Flask server running at http://localhost:5000")
    print(f"SRC_DIR  : {SRC_DIR}")
    print(f"DATA_DIR : {DATA_DIR}")
    print(f"OUTPUTS  : {OUTPUTS_DIR}")
    print(f"Device   : {device}")
    app.run(debug=True, port=5000)