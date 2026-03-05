import numpy as np
import torch
from pathlib import Path
from torchvision import transforms
from PIL import Image
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_sample_from_animals(data_dir, img_size=224):
    """Lấy ngẫu nhiên 1 ảnh từ 1 lớp bất kỳ trong dataset động vật.

    Hỗ trợ cả hai trường hợp:
    - data_dir trỏ tới thư mục gốc (ví dụ: "data"), có thư mục con "animals"
    - data_dir trỏ trực tiếp tới thư mục dataset (ví dụ: "data/animals")
    """
    rng = np.random.default_rng()
    root = Path(data_dir)
    base_dir = root / "animals" if (root / "animals").exists() else root

    if not base_dir.exists():
        raise ValueError(f"Không tìm thấy thư mục dataset: {base_dir}")

    # Liệt kê các thư mục class
    class_dirs = [p for p in base_dir.iterdir() if p.is_dir()]
    if not class_dirs:
        raise ValueError(f"Không tìm thấy thư mục lớp trong: {base_dir}")

    # Chọn ngẫu nhiên một class và sau đó là một ảnh trong class đó
    selected_class_dir = rng.choice(class_dirs)
    selected_class = selected_class_dir.name

    # Hỗ trợ nhiều phần mở rộng ảnh
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    image_files = []
    for pat in patterns:
        image_files.extend(selected_class_dir.glob(pat))

    if not image_files:
        raise ValueError(f"Không tìm thấy ảnh nào trong lớp: {selected_class}")

    selected_file = rng.choice(image_files)
    print(f"Đã chọn lớp: {selected_class} | Ảnh: {selected_file.name}")

    # Load và preprocess ảnh (giữ RGB, không Grayscale)
    org_image = Image.open(selected_file).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_tensor = transform(org_image).unsqueeze(0)
    img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    return img_tensor, img_np, selected_file.name, org_image.resize((img_size, img_size)), selected_class

def load_sample_from_caltech(data_dir, img_size=224):
    """Lấy ngẫu nhiên 1 ảnh từ 1 lớp bất kỳ trong dataset Caltech-101"""
    rng = np.random.default_rng()
    root = Path(data_dir)
    base_dir = root / "caltech-101" if (root / "caltech-101").exists() else root
    

    if not base_dir.exists():
        raise ValueError(f"Không tìm thấy thư mục dataset: {base_dir}")

    # Liệt kê các thư mục class
    class_dirs = [p for p in base_dir.iterdir() if p.is_dir()]

    if not class_dirs:
        raise ValueError(f"Không tìm thấy thư mục lớp trong: {base_dir}")

    # Chọn ngẫu nhiên một class và sau đó là một ảnh trong class đó
    selected_class_dir = rng.choice(class_dirs)
    selected_class = selected_class_dir.name

    # Hỗ trợ nhiều phần mở rộng ảnh
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    image_files = []
    for pat in patterns:
        image_files.extend(selected_class_dir.glob(pat))

    if not image_files:
        raise ValueError(f"Không tìm thấy ảnh nào trong lớp: {selected_class}")

    selected_file = rng.choice(image_files)
    print(f"Đã chọn lớp: {selected_class} | Ảnh: {selected_file.name}")

    # Load và preprocess ảnh (giữ RGB, không Grayscale)
    org_image = Image.open(selected_file).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_tensor = transform(org_image).unsqueeze(0)
    img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    return img_tensor, img_np, selected_file.name, org_image.resize((img_size, img_size)), selected_class

def load_sample_from_brain_tumor(data_dir, img_size=224):
    """Lấy ngẫu nhiên 1 ảnh từ 1 lớp bất kỳ trong dataset brain-tumor"""
    rng = np.random.default_rng()
    root = Path(data_dir)
    base_dir = root / "brain-tumor" if (root / "brain-tumor").exists() else root
    

    if not base_dir.exists():
        raise ValueError(f"Không tìm thấy thư mục dataset: {base_dir}")
    testing_dir = base_dir / "Testing"
    # Liệt kê các thư mục class
    class_dirs = [p for p in testing_dir.iterdir() if p.is_dir()]

    if not class_dirs:
        raise ValueError(f"Không tìm thấy thư mục lớp trong: {testing_dir}")

    # Chọn ngẫu nhiên một class và sau đó là một ảnh trong class đó
    selected_class_dir = rng.choice(class_dirs)
    selected_class = selected_class_dir.name

    # Hỗ trợ nhiều phần mở rộng ảnh
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    image_files = []
    for pat in patterns:
        image_files.extend(selected_class_dir.glob(pat))

    if not image_files:
        raise ValueError(f"Không tìm thấy ảnh nào trong lớp: {selected_class}")

    selected_file = rng.choice(image_files)
    print(f"Đã chọn lớp: {selected_class} | Ảnh: {selected_file.name}")

    # Load và preprocess ảnh (giữ RGB, không Grayscale)
    org_image = Image.open(selected_file).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_tensor = transform(org_image).unsqueeze(0)
    img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    return img_tensor, img_np, selected_file.name, org_image.resize((img_size, img_size)), selected_class

def predict_with_model(model, img_tensor):
    """Dự đoán với mô hình"""
    model.eval()
    with torch.no_grad():
        logits = model(img_tensor.to(device))
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()
    
    return pred_class, confidence, probs.cpu().numpy()[0]


# --------- Vector hóa heatmap để tính sim ----------
def vectorize_explanation(heatmap, k=2000):
    flat = heatmap.flatten()
    idx = np.argsort(-np.abs(flat))[:k]
    vec = np.zeros_like(flat)
    vec[idx] = flat[idx]
    return vec

def predict_proba_fn(model, imgs: np.ndarray):
    """
    Hàm cho LIME: nhận batch ảnh numpy (N,H,W,C) -> trả về xác suất (N,num_classes).
    """
    device = next(model.parameters()).device

    # chuyển numpy -> torch tensor
    if imgs.ndim == 3:   # (H,W,C)
        imgs = np.expand_dims(imgs, axis=0)  # -> (1,H,W,C)

    # scale về [0,1] nếu chưa
    if imgs.max() > 1.0:
        imgs = imgs / 255.0

    # (N,H,W,C) -> (N,C,H,W)
    imgs_t = torch.from_numpy(imgs).permute(0,3,1,2).float().to(device)

    with torch.no_grad():
        logits = model(imgs_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    return probs

def _make_perturbation(block, mode='noise', ksize=(11, 11), sigma=50):
    """Tạo perturbation cho một block"""
    if mode == 'noise':
        noise = np.random.normal(0, 0.1, block.shape)
        return np.clip(block + noise, 0, 1)
    elif mode == 'zero':
        return np.zeros_like(block)
    elif mode == 'mean':
        return np.full_like(block, np.mean(block))
    elif mode == "blur":
        k_h = min(ksize[0], block.shape[0] | 1)  # |1 để đảm bảo lẻ
        k_w = min(ksize[1], block.shape[1] | 1)
        # Áp dụng Gaussian blur
        blurred = cv2.GaussianBlur(block, (k_w, k_h), sigma)
        return blurred
    else:
        return block