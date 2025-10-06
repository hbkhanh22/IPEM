"""
Script để lấy một mẫu từ tập COVID và sử dụng 3 mô hình LIME, SHAP và PEBEX 
để dự đoán và giải thích quyết định mô hình
"""
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import transforms
from PIL import Image
from skimage.transform import resize
import argparse
import cv2

import sys
sys.path.append('src')
from covid_classifier import COVIDClassifier
from animal_classifier import AnimalImageClassifier
from pebex_explainer import PEBEXExplainer
from caltech_classifier import CaltechImageClassifier
from brain_tumor_classifier import BrainTumorClassifier
from lime import lime_image
from skimage.segmentation import mark_boundaries, slic
import shap


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
    #pred_class = predict_with_model(clf.model, img_np)[0]
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

def explain_with_shap(clf, img_tensor, class_names, output_dir, args_dataset, org_img, top_k=3):
    """Giải thích bằng SHAP"""
    print("🔍 Đang chạy SHAP...")
    output_dir = f"{output_dir}/{args_dataset}"
    
    device = next(clf.model.parameters()).device
    # Tạo background (có thể dùng ảnh trung bình hoặc ảnh khác)
    background = torch.zeros_like(img_tensor).to(device)
    
    explainer = shap.GradientExplainer(clf.model, background)
    shap_values = explainer.shap_values(img_tensor.to(device))
    
    # Xử lý shap_values (có thể là list hoặc array)
    if isinstance(shap_values, list):
        shap_array = np.array(shap_values)  # (num_classes, 1, C, H, W)
    else:
        shap_array = shap_values  # (1, C, H, W, num_classes) hoặc dạng khác
    #shap_array.resize((org_img.shape))
    # Lấy class được dự đoán
    pred_class, _, _ = predict_with_model(clf.model, img_tensor)
    
    #print(shap_array.shape)
    
    #img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
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
        if arr.ndim >= 3:
            # Ưu tiên coi 2 trục cuối là (H, W)
            while arr.ndim > 3:
                arr = np.sum(np.abs(arr), axis=0)
            if arr.shape[0] in (1, 3):
                heatmap = np.sum(np.abs(arr), axis=0)
            else:
                # Nếu không rõ kênh, tổng mọi trục trừ 2 trục cuối
                while arr.ndim > 2:
                    arr = np.sum(np.abs(arr), axis=0)
                heatmap = arr
        else:
            # Không suy luận được -> tạo zeros để tránh lỗi
            heatmap = np.zeros((224, 224), dtype=float)
            
    # Resize heatmap về cùng kích thước ảnh gốc
    heatmap_resized = resize(
        heatmap,
        (org_img_np.shape[0], org_img_np.shape[1]),
        preserve_range=True
    )
    
    # Tạo superpixel với kích thước vừa phải bằng SLIC
    img_float = (org_img_np.astype(np.float32) / 255.0)
    segments = slic(img_float, n_segments=50, compactness=10, sigma=1, start_label=0)

    # Tính giá trị trung bình SHAP cho mỗi superpixel
    sp_importance = {}
    for seg_id in np.unique(segments):
        mask_sp = (segments == seg_id)
        if np.any(mask_sp):
            sp_importance[seg_id] = float(np.mean(heatmap_resized[mask_sp]))

    # Chọn top-k superpixel quan trọng nhất
    top_segments = sorted(sp_importance.items(), key=lambda x: -abs(x[1]))[:top_k]
    top_ids = set([seg_id for seg_id, _ in top_segments])

    # Tạo heatmap highlight chỉ cho top-k
    highlight_map = np.zeros_like(heatmap_resized, dtype=np.float32)
    for seg_id in top_ids:
        mask_sp = (segments == seg_id)
        highlight_map[mask_sp] = heatmap_resized[mask_sp]

    # Lưu kết quả
    output_path = Path(output_dir) / "shap"
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = output_path / "shap_explanation.png"
    visualize_counterfactual_explanation(
        classifier=clf,
        org_img=org_img,
        heatmap=highlight_map,
        model_name="SHAP",
        pred_class=pred_class,
        save_path=save_path
    )

    return shap_values, highlight_map

def explain_with_pebex(clf, img_tensor, class_names, output_dir, args_dataset, org_img, top_k=3):
    """Giải thích bằng PEBEX với visualization tương tự SHAP"""
    print("🔍 Đang chạy PEBEX...")
    output_dir = f"{output_dir}/{args_dataset}"
    pebex = PEBEXExplainer(clf.model, class_names)
    heatmap, pred_class = pebex.explain_one(img_tensor.squeeze(0), mode="black")
    
    # Lưu kết quả
    output_path = Path(output_dir) / "pebex"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Lưu kết quả
    output_path = Path(output_dir) / "pebex"
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = output_path / "pebex_explanation.png"
    #visualize_saliency(org_img, heatmap, "PeBEx", class_names[pred_class], save_path)
    visualize_counterfactual_explanation(
        classifier=clf,
        org_img=org_img,
        heatmap=heatmap,
        model_name="PeBEx",
        pred_class=predict_with_model(clf.model, img_tensor)[0],
        save_path=save_path
    )

    return heatmap

def  visualize_counterfactual_explanation(classifier, org_img, heatmap, model_name, pred_class, save_path=None, percentile_threshold=70, alpha=0.5, cmap='jet'):
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
    # Resize heatmap để match với ảnh gốc
    heatmap_resized = resize(heatmap, (H, W), preserve_range=True)

    # Counterfactual explanation
    heatmap_norm = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min())

    if model_name == "LIME" or model_name == "SHAP":
        mask = resize(heatmap_norm, (H, W), preserve_range=True).astype(np.uint8)
    else:
        thresh_val = np.percentile(heatmap_norm, percentile_threshold)
        mask = (heatmap_norm >= thresh_val).astype(np.uint8)

    blurred = cv2.GaussianBlur(org_img, (11, 11), 50)

    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    perturbed_img = org_img.copy()
    perturbed_img[mask_3d == 1] = blurred[mask_3d == 1]

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
    hm = ax[1].imshow(heatmap_resized, cmap=cmap, alpha=alpha)
    ax[1].set_title(f"{model_name} Saliency Map - Pred: {classifier.class_names[pred_class]}")
    ax[1].axis("off")
    fig.colorbar(hm, ax=ax[1], fraction=0.046, pad=0.04)

    ax[2].imshow(perturbed_img, alpha=1.0)
    ax[2].set_title(f"Perturbed Image - Pred: {classifier.class_names[new_class]}")
    ax[2].axis("off")

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def main(args_dataset, args_model_name):
    # Cấu hình
    data_dir = "data"  # có thể là "data" hoặc "data/animals"
    output_dir = "outputs"  # Sử dụng thư mục outputs chính
    sample_output_dir = "outputs/sample_explanation"  # Thư mục cho kết quả mẫu
    img_size = 224
    
    print("🚀 Bắt đầu script...")
    
    try:
        if args_dataset.lower() == 'animals': 
            clf = AnimalImageClassifier(data_dir=data_dir, output_dir=output_dir, img_size=img_size, args_model=args_model_name)

            # Lấy mẫu ngẫu nhiên từ tập động vật
            print("\n🖼️  Lấy mẫu ngẫu nhiên từ dataset động vật...")
            img_tensor, img_np, filename, org_img, selected_class = load_sample_from_animals(data_dir, img_size)
            print(f"   - Ảnh gốc shape: {np.array(org_img).shape}")
        elif args_dataset.lower() == 'sars-cov2-ct':
        # Khởi tạo classifier
            clf = COVIDClassifier(data_dir=data_dir, output_dir=output_dir, img_size=img_size, args_model=args_model_name)

            # Lấy mẫu từ COVID
            print("\n🖼️  Lấy mẫu từ tập COVID...")
            img_tensor, img_np, filename, org_img = load_sample_from_covid(data_dir, img_size)
            print(f"   - Ảnh gốc shape: {np.array(org_img).shape}")
        elif args_dataset.lower() == 'caltech-101':
            clf = CaltechImageClassifier(data_dir=data_dir, output_dir=output_dir, img_size=img_size, args_model=args_model_name)
            # Lấy mẫu từ Caltech-101
            print("\n🖼️  Lấy mẫu từ tập Caltech-101...")
            img_tensor, img_np, filename, org_img, selected_class = load_sample_from_caltech(data_dir, img_size)
            print(f"   - Ảnh gốc shape: {np.array(org_img).shape}")
        elif args_dataset.lower() == 'brain-tumor':
            clf = BrainTumorClassifier(data_dir=data_dir, output_dir=output_dir, img_size=img_size, args_model=args_model_name)
            # Lấy mẫu từ thư mục Testing trong brain-tumor
            print("\n🖼️  Lấy mẫu từ thư mục Testing của tập Brain-tumor...")
            img_tensor, img_np, filename, org_img, selected_class = load_sample_from_brain_tumor(data_dir, img_size)
            print(f"   - Ảnh gốc shape: {np.array(org_img).shape}")

        clf._build_dataloaders()
        clf.load_trained_model()
        
        print("📊 Thông tin mô hình:")
        print(f"   - Classes: {clf.class_names}")
        print(f"   - Device: {device}")
        
        # Set random seed
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Dự đoán với mô hình
        print("\n🔮 Dự đoán với mô hình...")
        pred_class, confidence, probs = predict_with_model(clf.model, img_tensor)
        # Xác định index lớp đúng dựa trên tên class được chọn
        if selected_class in clf.class_names:
            true_class = clf.class_names.index(selected_class)
        else:
            raise ValueError(f"Lớp '{selected_class}' không nằm trong class_names của mô hình: {clf.class_names}")
        
        print(f"   - Ảnh: {filename}")
        print(f"   - True class: {clf.class_names[true_class]} (index: {true_class})")
        print(f"   - Predicted class: {clf.class_names[pred_class]} (index: {pred_class})")
        print(f"   - Confidence: {confidence:.4f}")
        print(f"   - Prediction correct: {pred_class == true_class}")
        
        # In xác suất cho tất cả classes
        print("\n📈 Xác suất dự đoán:")
        for i, (class_name, prob) in enumerate(zip(clf.class_names, probs)):
            print(f"   - {class_name}: {prob:.4f}")
        
        # Nếu dự đoán đúng, giải thích quyết định
        if pred_class == true_class:
            # LIME
            lime_explanation, lime_mask = explain_with_lime(clf, img_np, clf.class_names, sample_output_dir, args_dataset, org_img)
            # SHAP
            shap_values, shap_heatmap = explain_with_shap(clf, img_tensor, clf.class_names, sample_output_dir, args_dataset, org_img)

            # PEBEX
            pebex_heatmap = explain_with_pebex(clf, img_tensor, clf.class_names, sample_output_dir, args_dataset, org_img)
            
            print(f"\n💾 Kết quả đã được lưu trong: {sample_output_dir}")
            print("   - LIME: outputs/sample_explanation/lime/lime_explanation.png")
            print("   - SHAP: outputs/sample_explanation/shap/shap_explanation.png")
            print("   - PEBEX: outputs/sample_explanation/pebex/pebex_explanation.png")
            
        else:
            print(f"\n❌ Dự đoán sai! Không thể giải thích quyết định.")
            print("   Hãy thử với mẫu khác...")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
        input("Nhấn Enter để tiếp tục...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['animals', 'sars-cov2', 'caltech-101', 'brain-tumor'], default='animals')
    parser.add_argument('--model', type=str, choices=["efficientnet_b3", "resnet50", "densenet121", "mobilenet_v3", "vgg19"], default="efficientnet_b3")

    dataset = parser.parse_args().dataset
    model_name = parser.parse_args().model
    main(dataset, model_name)
