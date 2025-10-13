"""
Script để lấy một mẫu từ tập COVID và sử dụng 3 mô hình LIME, SHAP và PEBEX 
để dự đoán và giải thích quyết định mô hình
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import transforms
from PIL import Image
from skimage.transform import resize

from pebex_explainer import PEBEXExplainer
from lime import lime_image
from skimage.segmentation import mark_boundaries, slic
import shap
import cv2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_sample_from_covid(data_dir, img_size=224):
    """Lấy một mẫu ngẫu nhiên từ thư mục COVID"""
    non_covid_dir = Path(data_dir) / "COVID"
    
    # Lấy danh sách tất cả file ảnh
    image_files = list(non_covid_dir.glob("*.png"))
    
    if not image_files:
        raise ValueError("Không tìm thấy ảnh nào trong thư mục COVID")
    
    # Chọn ngẫu nhiên một ảnh
    #selected_file = random.choice(image_files)
    selected_file = np.random.default_rng().choice(image_files)

    print(f"Đã chọn ảnh: {selected_file.name}")
    
    # Load và preprocess ảnh
    org_image = Image.open(selected_file).convert('RGB')
    
    # Transform giống như trong training
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img_tensor = transform(org_image).unsqueeze(0)  # Thêm batch dimension
    img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, C) cho visualization
    
    return img_tensor, img_np, selected_file.name, org_image.resize((img_size, img_size))

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

def explain_with_lime(clf, img_np, class_names, output_dir, args_dataset, org_img):
    """Giải thích bằng LIME"""
    output_dir = f"{output_dir}/{args_dataset}"
    print("🔍 Đang chạy LIME...")
    print(f"   - Input shape: {img_np.shape}")
    print(f"   - Output dir: {output_dir}")
    
    def predict_proba_fn(imgs):
        """Hàm dự đoán cho LIME"""
        device = next(clf.model.parameters()).device
        
        if imgs.ndim == 3:
            imgs = np.expand_dims(imgs, axis=0)
        
        # if imgs.max() > 1.0:
        #     imgs = imgs / 255.0
        
        imgs_t = torch.from_numpy(imgs).permute(0, 3, 1, 2).float().to(device)
        
        with torch.no_grad():
            logits = clf.model(imgs_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        
        return probs
    
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_np,
        classifier_fn=predict_proba_fn,
        top_labels=len(class_names),
        hide_color=0,
        num_samples=2000,
    )
    
    # Lấy mask cho class được dự đoán
    pred_class = np.argmax(predict_proba_fn(np.array([img_np])))
    temp, mask = explanation.get_image_and_mask(
        label=pred_class, 
        positive_only=True, 
        num_features=2, 
        hide_rest=True
    )

    print(mask.shape)
    
    # Lưu kết quả
    output_path = Path(output_dir) / "lime"
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"   - Tạo thư mục: {output_path}")
    save_path = output_path / "lime_explanation.png"
    visualize_counterfactual_explanation(
        classifier=clf,
        org_img=org_img,
        heatmap=mask,
        model_name="LIME",
        pred_class=pred_class,
        save_path=save_path
    )
    return explanation, mask

def explain_with_shap(clf, img_tensor, class_names, output_dir, args_dataset, org_img, batch_size=8, top_k=3):
    """Giải thích bằng SHAP"""
    print("🔍 Đang chạy SHAP...")
    output_dir = f"{output_dir}/{args_dataset}"
    
    device = next(clf.model.parameters()).device
    # Tạo background (có thể dùng ảnh trung bình hoặc ảnh khác)
    background = torch.zeros_like(img_tensor).to(device)
    
    explainer = shap.GradientExplainer(clf.model, background, batch_size=batch_size)
    shap_values = explainer.shap_values(img_tensor.to(device))
    
    # Xử lý shap_values (có thể là list hoặc array)
    if isinstance(shap_values, list):
        shap_array = np.array(shap_values)  # (num_classes, 1, C, H, W)
    else:
        shap_array = shap_values  # (1, C, H, W, num_classes) hoặc dạng khác
    # Lấy class được dự đoán
    pred_class, _, _ = predict_with_model(clf.model, img_tensor)
    org_img_np = np.array(org_img)
    # Tạo heatmap cho class được dự đoán dưới dạng ma trận (H, W)
    if shap_array.ndim == 5 and shap_array.shape[0] > 1 and shap_array.shape[1] == 1:
        # (num_classes, 1, C, H, W)
        class_shap = shap_array[pred_class, 0]            # (C, H, W)
        heatmap = np.sum(np.abs(class_shap), axis=0)      # (H, W)
    elif shap_array.ndim == 5 and shap_array.shape[0] == 1:
        # (1, C, H, W, num_classes)
        class_shap = shap_array[0, :, :, :, pred_class]   # (C, H, W)
        heatmap = np.sum(np.abs(class_shap), axis=0)      # (H, W)
    else:
        # Fallback: cố gắng đưa về (H, W)
        arr = np.array(shap_array)
        # Nếu trục cuối là số lớp, chọn theo lớp dự đoán
        if arr.ndim >= 4 and arr.shape[-1] == len(class_names):
            arr = arr[..., pred_class]
        # Đưa về (C, H, W) nếu có thể
        while arr.ndim > 2:
            arr = np.sum(np.abs(arr), axis=0)
        heatmap = arr
            
    # Resize heatmap về cùng kích thước ảnh gốc
    heatmap_resized = resize(
        heatmap,
        (org_img_np.shape[0], org_img_np.shape[1]),
        preserve_range=True
    )
    
    # # Tạo superpixel với kích thước vừa phải bằng SLIC
    # img_float = org_img_np / 255.0
    # segments = slic(img_float, n_segments=50, compactness=30, sigma=1, start_label=0)
    # # Tạo heatmap highlight
    # highlight_map = np.zeros_like(heatmap_resized, dtype=np.float32)

    # for seg_id in np.unique(segments):
    #     mask_sp = (segments == seg_id)
    #     if np.any(mask_sp):
    #         highlight_map[mask_sp] = heatmap_resized[mask_sp]

    # 4️⃣ Làm mịn heatmap (giống Grad-CAM smoothing)
    from scipy.ndimage import gaussian_filter
    heatmap_smooth = gaussian_filter(heatmap_resized, sigma=4)

    # 5️⃣ Chuẩn hóa [0, 1]
    heatmap_smooth -= heatmap_smooth.min()
    heatmap_smooth /= (heatmap_smooth.max() + 1e-8)

    # Lưu kết quả
    output_path = Path(output_dir) / "shap"
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = output_path / "shap_explanation.png"
    visualize_counterfactual_explanation(
        classifier=clf,
        org_img=org_img,
        heatmap=heatmap_smooth,
        model_name="SHAP",
        pred_class=pred_class,
        save_path=save_path
    )

    return shap_values, heatmap_smooth


def explain_with_pebex(clf, img_tensor, class_names, output_dir, args_dataset, org_img):
    """Giải thích bằng PEBEX với visualization tương tự SHAP"""
    print("🔍 Đang chạy PEBEX...")
    output_dir = f"{output_dir}/{args_dataset}"
    pebex = PEBEXExplainer(clf.model, class_names)
    heatmap, pred_class = pebex.explain_one(img_tensor.squeeze(0), mode="black")

    # Lưu kết quả
    output_path = Path(output_dir) / "pebex"
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = output_path / "pebex_explanation.png"
    visualize_counterfactual_explanation(
        classifier=clf,
        org_img=org_img,
        heatmap=heatmap,
        model_name="PeBEx",
        pred_class=predict_with_model(clf.model, img_tensor)[0],
        save_path=save_path
    )

    return heatmap

def  visualize_counterfactual_explanation(classifier, org_img, heatmap, model_name, pred_class, save_path=None, percentile_threshold=95, alpha=0.5, cmap='jet'):
    """
    Visualize heatmap (PEBEX) giống saliency map overlay lên ảnh gốc.
    
    Args:
        org_img (PIL.Image or np.ndarray): ảnh gốc
        heatmap (np.ndarray): ma trận heatmap từ PEBEX
        save_path (str, optional): nếu muốn lưu ảnh
        alpha (float): độ trong suốt của heatmap
        cmap (str): colormap để hiển thị heatmap
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Chuyển ảnh về numpy
    if not isinstance(org_img, np.ndarray):
        org_img = np.array(org_img)

    H, W = org_img.shape[:2]
    # # Resize heatmap để match với ảnh gốc
    # heatmap_resized = resize(heatmap, (H, W), preserve_range=True)

    if model_name == "LIME":
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        mask = resize(heatmap_norm, (H, W), preserve_range=True).astype(np.uint8)
    elif model_name == "SHAP":
        thresh_val = np.percentile(heatmap, percentile_threshold)
        mask = (heatmap >= thresh_val).astype(np.uint8)
    else:
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        thresh_val = np.percentile(heatmap_norm, percentile_threshold)
        mask = (heatmap_norm >= thresh_val).astype(np.uint8)

    blurred = cv2.GaussianBlur(org_img, (11, 11), 50)

    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    perturbed_img = org_img.copy()
    noise = np.random.uniform(0, 255, org_img.shape).astype(np.float32)

    perturbed_img[mask_3d == 1] = blurred[mask_3d == 1]
    # perturbed_img[mask_3d == 1] = 0

    # 5. Chuẩn bị input cho model
    perturbed_tensor = torch.from_numpy(perturbed_img.transpose(2,0,1)).unsqueeze(0).float().to(device)
    perturbed_tensor = perturbed_tensor / 255.0
 
    classifier.model.eval()
    with torch.no_grad():
        outputs = classifier.model(perturbed_tensor)
        new_class = torch.argmax(outputs, dim=1).item()

    fig, ax = plt.subplots(1, 3, figsize=(15, 6))

    # Ảnh gốc
    ax[0].imshow(org_img)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    # Overlay saliency map
    ax[1].imshow(org_img, alpha=1.0)
    hm = ax[1].imshow(heatmap, cmap=cmap, alpha=alpha)
    ax[1].set_title(f"{model_name} Saliency Map - Pred: {classifier.class_names[pred_class]}")
    ax[1].axis("off")
    fig.colorbar(hm, ax=ax[1], fraction=0.046, pad=0.04)

    ax[2].imshow(perturbed_img, alpha=1.0)
    ax[2].set_title(f"Perturbed Image - Pred: {classifier.class_names[new_class]}")
    ax[2].axis("off")

    # Căn chỉnh layout để 3 ảnh bằng nhau
    plt.subplots_adjust(wspace=0.05, hspace=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)