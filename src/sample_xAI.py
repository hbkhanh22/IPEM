import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.transform import resize
from torchvision import transforms
from PIL import Image
from pebex_explainer import PEBEXExplainer
from lime import lime_image
import shap
import cv2
from utils import predict_with_model
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def renormalize_image(img):
    if img is None:
        return None
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img_uint8 = img_norm.astype(np.uint8)
    return img_uint8

def explain_with_lime(clf, img_np, class_names, output_dir, args_dataset, org_img):
    """Giải thích bằng LIME"""
    if output_dir:
        output_dir = f"{output_dir}/{args_dataset}"
        output_path = Path(output_dir) / "lime"
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"   - Tạo thư mục: {output_path}")
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
    original_probs = predict_proba_fn(np.array([img_np]))[0]

    temp, mask = explanation.get_image_and_mask(
        label=pred_class, 
        positive_only=True, 
        num_features=5, 
        hide_rest=True
    )

    # save_path = output_path / "lime_explanation.png"

    visualize_counterfactual_explanation(
        classifier=clf,
        org_img=org_img,
        heatmap=mask,
        model_name="LIME",
        pred_class=pred_class,
        original_probs=original_probs,
        output_path=output_path
    )
    return explanation, mask

def explain_with_shap(clf, img_tensor, class_names, output_dir, args_dataset, org_img, batch_size=8, top_k=3):
    """Giải thích bằng SHAP"""
    print("🔍 Đang chạy SHAP...")
    if output_dir:
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
    pred_class, _, original_probs = predict_with_model(clf.model, img_tensor)
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

    # 4️⃣ Làm mịn heatmap (giống Grad-CAM smoothing)
    from scipy.ndimage import gaussian_filter
    heatmap_smooth = gaussian_filter(heatmap_resized, sigma=4)

    # 5️⃣ Chuẩn hóa [0, 1]
    heatmap_smooth -= heatmap_smooth.min()
    heatmap_smooth /= (heatmap_smooth.max() + 1e-8)

    # Lưu kết quả
    output_path = Path(output_dir) / "shap"
    output_path.mkdir(parents=True, exist_ok=True)
    # save_path = output_path / "shap_explanation.png"
    visualize_counterfactual_explanation(
        classifier=clf,
        org_img=org_img,
        heatmap=heatmap_smooth,
        model_name="SHAP",
        pred_class=pred_class,
        original_probs=original_probs,
        output_path=output_path
    )

    return shap_values, heatmap_smooth


def explain_with_pebex(clf, img_tensor, class_names, output_dir, args_dataset, org_img):
    """Giải thích bằng PEBEX với visualization tương tự SHAP"""
    print("🔍 Đang chạy PEBEX...")
    if output_dir:
        output_dir = f"{output_dir}/{args_dataset}"
        output_path = Path(output_dir) / "pebex"
        output_path.mkdir(parents=True, exist_ok=True)
        
    pebex = PEBEXExplainer(clf.model, class_names)
    heatmap, pred_class = pebex.explain_slic(img_tensor.squeeze(0), mode="blur")

    # save_path = output_path / "pebex_explanation.png"
    visualize_counterfactual_explanation(
        classifier=clf,
        org_img=org_img,
        heatmap=heatmap,
        model_name="PeBEx",
        pred_class=predict_with_model(clf.model, img_tensor)[0],
        original_probs=predict_with_model(clf.model, img_tensor)[2],
        output_path=output_path
    )

    return heatmap

def explain_with_gradcam(clf, img_tensor, class_names, output_dir, args_dataset, org_img):
    if output_dir:
        output_dir = f"{output_dir}/{args_dataset}"
        output_path = Path(output_dir) / "GradCAM"
        output_path.mkdir(parents=True, exist_ok=True)
    
    def get_last_conv_layer(model):
        """
        Trả về module Conv2d cuối + reference layer
        Dùng cho pytorch-grad-cam.
        """
        last_conv = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise ValueError("❌ Không tìm thấy Conv2D nào trong model. GradCAM yêu cầu CNN.")
        return last_conv

    target_layer = get_last_conv_layer(clf.model)
    
    cam = GradCAM(model=clf.model, target_layers=[target_layer])

    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        logits = clf.model(img_tensor)
        y_model = logits.argmax(1).cpu().numpy()

    target = [ClassifierOutputTarget(y_model)]
    heatmap = cam(input_tensor=img_tensor, targets=target)[0]

    visualize_counterfactual_explanation(
        classifier=clf,
        org_img=org_img,
        heatmap=heatmap,
        model_name="GradCAM",
        pred_class=predict_with_model(clf.model, img_tensor)[0],
        original_probs=predict_with_model(clf.model, img_tensor)[2],
        output_path=output_path
    )

    return heatmap


def  visualize_counterfactual_explanation(classifier, org_img, heatmap, model_name, pred_class, original_probs, output_path=None, percentile_threshold=50, alpha=0.5, cmap='jet'):
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
    
    renormalized_heatmap = renormalize_image(heatmap)
    cv2.imwrite(f"{output_path}/{model_name}_mask.png", renormalized_heatmap)

    H, W = org_img.shape[:2]

    # Resize heatmap để match với ảnh gốc
    if model_name == "LIME":
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        mask = resize(heatmap_norm, (H, W), preserve_range=True).astype(np.uint8)

    else:
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        # thresh_val = np.percentile(heatmap_norm, 80)
        # mean_val = np.mean(heatmap_norm)
        mask = (heatmap_norm >= 0.8).astype(np.uint8)
        # mask = heatmap_norm.astype(np.uint8)

    blurred = cv2.GaussianBlur(org_img, (11, 11), sigmaX=50, sigmaY=50)

    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    perturbed_img = org_img.copy()
    # noise = np.random.uniform(0, 255, org_img.shape).astype(np.float32)

    # perturbed_img[mask_3d == 1] = noise[mask_3d == 1]
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
    ax[1].set_title(f"{model_name} Saliency Map - Pred: {classifier.class_names[pred_class]}\n"
                    f"Probability: {original_probs[pred_class]:.2f}")
    ax[1].axis("off")
    fig.colorbar(hm, ax=ax[1], fraction=0.046, pad=0.04)

    ax[2].imshow(perturbed_img, alpha=1.0)
    ax[2].set_title(f"Perturbed Image - Pred: {classifier.class_names[new_class]}\n"
                    f"Probability of {classifier.class_names[pred_class]} class: {torch.softmax(outputs, dim=1)[0][pred_class]:.2f}")
    ax[2].axis("off")

    # Căn chỉnh layout để 3 ảnh bằng nhau
    plt.subplots_adjust(wspace=0.05, hspace=0)
    plt.tight_layout()

    if output_path:
        plt.savefig(f"{output_path}/{model_name}_explanation.png", dpi=200, bbox_inches="tight")

    plt.show()
    plt.close(fig)


def make_perturbation(img):
    perturbation_value = []
    for z in range(img.shape[2]):
        perturbation_value.append(np.mean(img[:,:,z]))

    new_img = np.ones(img.shape)
    for z in range(img.shape[2]):
        new_img[:,:,z] = new_img[:,:,z] * perturbation_value[z]

    return new_img

def make_perturbation(img, mode='blur', ksize=(11, 11), sigma=50):
    """
    Tạo nhiễu ảnh bằng các phương pháp khác nhau
    """
    if mode == 'blur':
        # Kiểm tra để đảm bảo kernel không lớn hơn block
        k_h = min(ksize[0], img.shape[0] | 1)  # |1 để đảm bảo lẻ
        k_w = min(ksize[1], img.shape[1] | 1)

        # Áp dụng Gaussian blur
        blurred = cv2.GaussianBlur(img, (k_w, k_h), sigma)

        return blurred
    elif mode == 'noise':
        noise = np.random.normal(0, 0.1, img.shape)
        return np.clip(img + noise, 0, 1)
    elif mode == "zero":
        return np.zeros_like(img)
    elif mode == "mean":
        return np.full_like(img, np.mean(img))
    else:
        return img

def AOPC_MoRF(clf, img_map_path, img_path, mode='blur', block_size=8, block_per_row=28, percentile=None, img_size=224, verbose=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_AOPC = None
    pct = -1 if percentile is None else percentile
    blocks_dict = {}
    key = 0
    img = cv2.imread(img_map_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_size, img_size))

    for i in range(0, img.shape[0], block_size):
        for j in range(0, img.shape[1], block_size):
            block = img[i: i + block_size, j : j + block_size]
            block_sum = np.sum(block)
            blocks_dict[key] = (block, block_sum)
            key += 1
    
    sorted_blocks = sorted(blocks_dict.items(), key=lambda x:x[1][1], reverse=True)
    total_sum_blocks = sum(block_sum[1] for block_sum in list(blocks_dict.values()))

    img_pred = cv2.imread(img_path)[:, :, ::-1]
    img_pred = cv2.resize(img_pred, (img_size, img_size))
    img_pred = np.float32(img_pred) / 255.0
    img_pred_aopc = img_pred.copy()
    # print(img_pred_aopc.shape)
    # Nếu ảnh đang là float (0–1), cần scale lại 0–255 để tạo PIL Image
    img_pred_pil = Image.fromarray((img_pred * 255).astype(np.uint8))

    # img_pred_aopc = img_pred.copy()
    # print(img_pred_aopc.shape)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = transform(img_pred_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = clf.model(input_tensor)

    probs_k0, indices = torch.topk(torch.nn.Softmax(dim=-1)(output), k=1)
    class_origin = indices[0]
    print(int(class_origin))

    count_blocks = 0
    if pct >= 0:
        sum_blocks = 0
        breaking_pct_value = (pct / 100) * total_sum_blocks
    
    if verbose:
        print(f"Total sum: {total_sum_blocks}")
        if pct >= 0:
            print(f"Breaking value {breaking_pct_value}")

    num_classes = output.shape[1]
    k = min(1000, num_classes)

    AOPC = []
    for i in range(len(blocks_dict.items())):
        ref_block_index = sorted_blocks[i][0]
        row = ref_block_index // block_per_row
        col = ref_block_index % block_per_row

        img_pred_aopc[row * block_size : row * block_size + block_size, col * block_size : col * block_size + block_size] = make_perturbation(img_pred_aopc[row * block_size : row * block_size + block_size, col * block_size : col * block_size + block_size], mode=mode)

        # input_tensor = transform(img_pred_pil.copy()).unsqueeze(0).to(device)
        # img_pred_aopc_uint8 = (img_pred_aopc * 255).astype(np.uint8)
        img_pred_pil_aopc = Image.fromarray((img_pred_aopc * 255).astype(np.uint8))
        input_tensor = transform(img_pred_pil_aopc).unsqueeze(0).to(device)

        with torch.no_grad():
            output = clf.model(input_tensor)
        
        probs, indices = torch.topk(torch.nn.Softmax(dim=-1)(output), k=k)
        class_probs = dict(zip(indices.squeeze().tolist(), probs.squeeze().tolist()))
        class_prob_after = class_probs.get(int(class_origin), 0.0)

        probs_k0_val = probs_k0.squeeze().tolist()
        delta = float(probs_k0_val) - float(class_prob_after)
        # AOPC.append(probs_k0_val - class_probs[int(class_origin)])

        AOPC.append(delta)

        count_blocks += 1
        if pct >= 0:
            sum_blocks += np.sum(sorted_blocks[i][1][0])
            if sum_blocks >= breaking_pct_value:
                if verbose:
                    print(f"Breaking after {count_blocks} evaluated blocks")
                break

    total_AOPC = (class_origin.item(), AOPC, np.sum(AOPC)/count_blocks)
    return total_AOPC
