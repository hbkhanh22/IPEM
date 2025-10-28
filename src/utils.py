import numpy as np
import torch
from pathlib import Path
from torchvision import transforms
from PIL import Image


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