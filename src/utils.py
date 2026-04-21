import numpy as np
import torch
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_with_model(model, img_tensor):
    """Run a forward pass and return the predicted class, confidence, and probability vector.
    
    Args:
        model: Trained PyTorch model used for inference.
        img_tensor: Input image tensor to evaluate.
    
    Return:
        tuple[int, float, np.ndarray]: Predicted class index, confidence score, and class probabilities.
    """
    model.eval()
    with torch.no_grad():
        logits = model(img_tensor.to(device))
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()
    
    return pred_class, confidence, probs.cpu().numpy()[0]

# --------- Vectorize the heatmap for similarity ----------
def vectorize_explanation(heatmap, k=2000):
    """Flatten a heatmap and keep only the top-k most important values by magnitude.
    
    Args:
        heatmap: Explanation map to vectorize.
        k: Number of strongest absolute values to preserve.
    
    Return:
        np.ndarray: Sparse one-dimensional vector representation of the heatmap.
    """
    flat = heatmap.flatten()
    idx = np.argsort(-np.abs(flat))[:k]
    vec = np.zeros_like(flat)
    vec[idx] = flat[idx]
    return vec

def predict_proba_fn(model, imgs: np.ndarray):
    """Convert NumPy images into tensors and return class probabilities for a model.
    
    Args:
        model: Trained PyTorch model used for inference.
        imgs: Image batch stored as NumPy arrays in HWC format.
    
    Return:
        np.ndarray: Predicted class probabilities for each input image.
    """
    device = next(model.parameters()).device

    # Convert NumPy arrays to torch tensors
    if imgs.ndim == 3:   # (H,W,C)
        imgs = np.expand_dims(imgs, axis=0)  # -> (1,H,W,C)

    # Scale to [0, 1] if needed
    if imgs.max() > 1.0:
        imgs = imgs / 255.0

    # (N,H,W,C) -> (N,C,H,W)
    imgs_t = torch.from_numpy(imgs).permute(0,3,1,2).float().to(device)

    with torch.no_grad():
        logits = model(imgs_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    return probs

def _make_perturbation(block, mode='noise', ksize=(11, 11), sigma=50):
    """Apply a perturbation strategy to an image block for faithfulness-style evaluations.
    
    Args:
        block: Image region to perturb.
        mode: Perturbation strategy such as noise, zero, mean, or blur.
        ksize: Gaussian kernel size used when blur mode is selected.
        sigma: Gaussian standard deviation used for blur mode.
    
    Return:
        np.ndarray: Perturbed image block.
    """
    if mode == 'noise':
        noise = np.random.normal(0, 0.1, block.shape)
        return np.clip(block + noise, 0, 1)
    elif mode == 'zero':
        return np.zeros_like(block)
    elif mode == 'mean':
        return np.full_like(block, np.mean(block))
    elif mode == "blur":
        k_h = min(ksize[0], block.shape[0] | 1)  # |1 ensures the kernel size remains odd
        k_w = min(ksize[1], block.shape[1] | 1)
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(block, (k_w, k_h), sigma)
        return blurred
    else:
        return block