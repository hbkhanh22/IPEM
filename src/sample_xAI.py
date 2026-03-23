import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.transform import resize
from torchvision import transforms
from PIL import Image
from ipem_explainer import IPEMExplainer
from lime import lime_image
import shap
import cv2
from utils import predict_with_model
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import time

def renormalize_image(img):
    if img is None:
        return None
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img_uint8 = img_norm.astype(np.uint8)
    return img_uint8

def explain_with_lime(clf, img_np, class_names, output_dir, args_dataset, org_img):
    """Giải thích bằng LIME"""
    start_time = time.time()
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
    
    pred_class = np.argmax(predict_proba_fn(np.array([img_np])))
    original_probs = predict_proba_fn(np.array([img_np]))[0]

    # ── Tạo continuous heatmap từ superpixel weights ──────────────────────────
    # Segments: mỗi pixel mang id của superpixel nó thuộc về
    segments = explanation.segments                          # shape (H, W), int

    # local_exp[pred_class] = list of (segment_id, weight)
    seg_weight_dict = dict(explanation.local_exp[pred_class])

    # Khởi tạo heatmap rỗng, chỉ gán weight cho các segment có trong dict
    heatmap_raw = np.zeros(segments.shape, dtype=np.float32)
    for seg_id, weight in seg_weight_dict.items():
        heatmap_raw[segments == seg_id] = weight

    # Giữ lại phần đóng góp dương (tương tự positive_only=True)
    heatmap_pos = np.maximum(heatmap_raw, 0)

    # Normalize về [0, 1]
    max_val = heatmap_pos.max()
    if max_val > 0:
        heatmap_continuous = heatmap_pos / max_val
    else:
        heatmap_continuous = heatmap_pos          # toàn 0, tránh chia 0
    # ─────────────────────────────────────────────────────────────────────────

    end_time = time.time()
    explanation_time = end_time - start_time

    visualize_counterfactual_explanation(
        classifier=clf,
        org_img=org_img,
        heatmap=heatmap_continuous,
        model_name="LIME",
        pred_class=pred_class,
        original_probs=original_probs,
        output_path=output_path,
        explanation_time=explanation_time
    )
    return explanation, heatmap_continuous

def explain_with_shap(clf, img_tensor, class_names, output_dir, args_dataset, org_img, batch_size=8):
    print("🔍 Đang chạy SHAP...")
    start_time = time.time()
    if output_dir:
        output_dir = f"{output_dir}/{args_dataset}"

    device = next(clf.model.parameters()).device

    # ✅ FIX 1: Dùng blurred image làm background thay vì zeros
    import torchvision.transforms.functional as TF
    img_squeezed = img_tensor.squeeze(0)  # (C, H, W)
    blurred_bg = TF.gaussian_blur(img_squeezed, kernel_size=51, sigma=10.0)

    # Tạo n background đa dạng để ổn định SHAP
    n_bg = 20
    noise_scale = 0.05
    backgrounds = torch.stack([
        blurred_bg + noise_scale * torch.randn_like(blurred_bg)
        for _ in range(n_bg)
    ]).to(device)  # (20, C, H, W)

    explainer = shap.GradientExplainer(clf.model, backgrounds, batch_size=batch_size)
    shap_values = explainer.shap_values(img_tensor.to(device))

    # Xử lý shap_values
    if isinstance(shap_values, list):
        shap_array = np.array([
            sv.detach().cpu().numpy() if torch.is_tensor(sv) else sv
            for sv in shap_values
        ])
    else:
        shap_array = shap_values

    pred_class, _, original_probs = predict_with_model(clf.model, img_tensor)
    org_img_np = np.array(org_img)

    if shap_array.ndim == 5 and shap_array.shape[0] > 1:
        class_shap = shap_array[pred_class, 0]       # (C, H, W)
    elif shap_array.ndim == 5 and shap_array.shape[0] == 1:
        class_shap = shap_array[0, :, :, :, pred_class]
    else:
        raise ValueError(f"Unsupported SHAP shape: {shap_array.shape}")

    # ✅ FIX 2: Dùng absolute SHAP + channel max thay vì weighted sum
    # Cách A: Absolute mean across channels (highlight mọi vùng ảnh hưởng)
    shap_abs = np.abs(class_shap)                    # (C, H, W)
    heatmap = shap_abs.max(axis=0)                   # (H, W) — dùng max thay mean

    # ✅ FIX 3: Chỉ giữ top-k% vùng quan trọng nhất (loại bỏ nhiễu nền)
    threshold = np.percentile(heatmap, 80)           # Chỉ giữ top 20%
    heatmap = np.where(heatmap >= threshold, heatmap, 0)

    # Resize
    heatmap_resized = resize(
        heatmap,
        (org_img_np.shape[0], org_img_np.shape[1]),
        preserve_range=True
    )

    # ✅ FIX 4: Giảm sigma smoothing để giữ biên rõ hơn
    from scipy.ndimage import gaussian_filter
    heatmap_smooth = gaussian_filter(heatmap_resized, sigma=2)  # sigma nhỏ hơn

    # Normalize [0, 1]
    heatmap_smooth -= heatmap_smooth.min()
    heatmap_smooth /= (heatmap_smooth.max() + 1e-8)

    # ✅ FIX 5: Tăng contrast bằng power transform
    heatmap_smooth = np.power(heatmap_smooth, 0.5)   # sqrt làm nổi bật vùng cao

    output_path = Path(output_dir) / "shap"
    output_path.mkdir(parents=True, exist_ok=True)

    end_time = time.time()
    explanation_time = end_time - start_time
    visualize_counterfactual_explanation(
        classifier=clf,
        org_img=org_img,
        heatmap=heatmap_smooth,
        model_name="SHAP",
        pred_class=pred_class,
        original_probs=original_probs,
        output_path=output_path,
        explanation_time=explanation_time
    )
    return shap_values, heatmap_smooth

def explain_with_ipem(clf, img_tensor, class_names, output_dir, args_dataset, org_img):
    """Giải thích bằng IPEM với visualization tương tự SHAP"""
    print("🔍 Đang chạy IPEM...")
    start_time = time.time()
    if output_dir:
        output_dir = f"{output_dir}/{args_dataset}"
        output_path = Path(output_dir) / "ipem"
        output_path.mkdir(parents=True, exist_ok=True)

    ipem = IPEMExplainer(clf.model, class_names)
    heatmap, pred_class = ipem.explain_by_slic(img_tensor.squeeze(0))
    end_time = time.time()
    explanation_time = end_time - start_time
    # save_path = output_path / "ipem_explanation.png"
    visualize_counterfactual_explanation(
        classifier=clf,
        org_img=org_img,
        heatmap=heatmap,
        model_name="IPEM",
        pred_class=predict_with_model(clf.model, img_tensor)[0],
        original_probs=predict_with_model(clf.model, img_tensor)[2],
        output_path=output_path,
        explanation_time=explanation_time
    )

    return heatmap

def explain_with_gradcam(clf, img_tensor, class_names, output_dir, args_dataset, org_img):
    start_time = time.time()
    print("🔍 Đang chạy GradCAM...")
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
    end_time = time.time()
    explantion_time = end_time - start_time
    visualize_counterfactual_explanation(
        classifier=clf,
        org_img=org_img,
        heatmap=heatmap,
        model_name="GradCAM",
        pred_class=predict_with_model(clf.model, img_tensor)[0],
        original_probs=predict_with_model(clf.model, img_tensor)[2],
        output_path=output_path,
        explanation_time=explantion_time
    )

    return heatmap


# def visualize_counterfactual_explanation(classifier, org_img, heatmap, model_name, pred_class, original_probs, output_path=None, explanation_time=None, percentile_threshold=50, alpha=0.5, cmap='jet'):
#     """
#     Visualize heatmap (ipem) giống saliency map overlay lên ảnh gốc.
    
#     Args:
#         org_img (PIL.Image or np.ndarray): ảnh gốc
#         heatmap (np.ndarray): ma trận heatmap từ ipem
#         save_path (str, optional): nếu muốn lưu ảnh
#         alpha (float): độ trong suốt của heatmap
#         cmap (str): colormap để hiển thị heatmap
#     """

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Chuyển ảnh về numpy
#     if not isinstance(org_img, np.ndarray):
#         org_img = np.array(org_img)
    
#     renormalized_heatmap = renormalize_image(heatmap)
#     cv2.imwrite(f"{output_path}/{model_name}_mask.png", renormalized_heatmap)

#     H, W = org_img.shape[:2]

#     # Resize heatmap để match với ảnh gốc
#     if model_name == "LIME":
#         heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
#         mask = resize(heatmap_norm, (H, W), preserve_range=True).astype(np.uint8)

#     else:
#         heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
#         # thresh_val = np.percentile(heatmap_norm, 80)
#         # mean_val = np.mean(heatmap_norm)
#         mask = (heatmap_norm >= 0.8).astype(np.uint8)
#         # mask = heatmap_norm.astype(np.uint8)

#     # blurred = cv2.GaussianBlur(org_img, (11, 11), sigmaX=50, sigmaY=50)

#     mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
#     perturbed_img = org_img.copy()
#     # noise = np.random.uniform(0, 255, org_img.shape).astype(np.float32)

#     # perturbed_img[mask_3d == 1] = noise[mask_3d == 1]
#     # perturbed_img[mask_3d == 1] = blurred[mask_3d == 1]
#     perturbed_img[mask_3d == 1] = 0
#     # 5. Chuẩn bị input cho model
#     perturbed_tensor = torch.from_numpy(perturbed_img.transpose(2,0,1)).unsqueeze(0).float().to(device)
#     perturbed_tensor = perturbed_tensor / 255.0
 
#     classifier.model.eval()
#     with torch.no_grad():
#         outputs = classifier.model(perturbed_tensor)
#         new_class = torch.argmax(outputs, dim=1).item()

#     fig, ax = plt.subplots(1, 2, figsize=(10, 6))

#     # Ảnh gốc
#     ax[0].imshow(org_img)
#     ax[0].set_title("Original Image")
#     ax[0].axis("off")

#     # Overlay saliency map
#     ax[1].imshow(org_img, alpha=1.0)
#     hm = ax[1].imshow(heatmap, cmap=cmap, alpha=alpha)
#     ax[1].set_title(f"{model_name} Explanation Map - Pred: {classifier.class_names[pred_class]}\n"
#                     f"Probability: {original_probs[pred_class]:.2f} - Explanation time: {explanation_time:.2f}s")
#     ax[1].axis("off")
#     fig.colorbar(hm, ax=ax[1], fraction=0.046, pad=0.04)

#     # ax[2].imshow(perturbed_img, alpha=1.0)
#     # ax[2].set_title(f"Perturbed Image - Pred: {classifier.class_names[new_class]}\n"
#     #                 f"Probability of {classifier.class_names[pred_class]} class: {torch.softmax(outputs, dim=1)[0][pred_class]:.2f}")
#     # ax[2].axis("off")

#     # Căn chỉnh layout để 3 ảnh bằng nhau
#     plt.subplots_adjust(wspace=0.05, hspace=0)
#     plt.tight_layout()

#     if output_path:
#         plt.savefig(f"{output_path}/{model_name}_explanation.png", dpi=200, bbox_inches="tight")

#     plt.show()
#     plt.close(fig)

def visualize_counterfactual_explanation(classifier, org_img, heatmap, model_name, pred_class, original_probs, output_path=None, explanation_time=None, percentile_threshold=50, alpha=0.5, cmap='jet'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not isinstance(org_img, np.ndarray):
        org_img = np.array(org_img)
    
    renormalized_heatmap = renormalize_image(heatmap)
    cv2.imwrite(f"{output_path}/{model_name}_mask.png", renormalized_heatmap)

    H, W = org_img.shape[:2]

    # if model_name == "LIME":
    #     heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    #     mask = resize(heatmap_norm, (H, W), preserve_range=True).astype(np.uint8)
    # else:

    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    mask = (heatmap_norm >= 0.7).astype(np.uint8)

    blurred = cv2.GaussianBlur(org_img, (11, 11), sigmaX=50, sigmaY=50)
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    perturbed_img = org_img.copy()
    # perturbed_img[mask_3d == 1] = 0
    # blurred = cv2.GaussianBlur(org_img, (11, 11), sigmaX=50, sigmaY=50)
    perturbed_img[mask_3d == 1] = blurred[mask_3d == 1]

    perturbed_tensor = torch.from_numpy(perturbed_img.transpose(2,0,1)).unsqueeze(0).float().to(device)
    perturbed_tensor = perturbed_tensor / 255.0
 
    classifier.model.eval()
    with torch.no_grad():
        outputs = classifier.model(perturbed_tensor)
        new_class = torch.argmax(outputs, dim=1).item()

    # Dùng gridspec: 2 cột ảnh bằng nhau + 1 cột nhỏ cho colorbar
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[1, 0.03, 1],  # 2 ảnh bằng nhau, cột thứ 3 cho colorbar
        wspace=0.01
    )

    ax0 = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[1])  # axes riêng cho colorbar
    ax1 = fig.add_subplot(gs[2])
    # cax = fig.add_subplot(gs[2])  # axes riêng cho colorbar

    # # Ảnh gốc
    # ax0.imshow(org_img)
    # ax0.set_title("Original Image")
    # ax0.axis("off")

    # Overlay saliency map
    ax0.imshow(org_img, alpha=1.0)
    hm = ax0.imshow(heatmap, cmap=cmap, alpha=alpha)
    ax0.set_title(
        f"{model_name} Explanation Map - Pred: {classifier.class_names[pred_class]}\n"
        f"Probability: {original_probs[pred_class]:.2f} - Explanation time: {explanation_time:.2f}s"
    )
    ax0.axis("off")

    # Colorbar trên axes riêng → không ảnh hưởng kích thước ax1
    fig.colorbar(hm, cax=cax)
    
    ax1.imshow(perturbed_img, alpha=1.0)
    ax1.set_title(f"Perturbed Image - Pred: {classifier.class_names[new_class]}\n"
                    f"Probability of {classifier.class_names[pred_class]} class: {torch.softmax(outputs, dim=1)[0][pred_class]:.2f}")
    ax1.axis("off")

    plt.tight_layout()

    if output_path:
        plt.savefig(f"{output_path}/{model_name}_explanation.png", dpi=200, bbox_inches="tight")

    plt.show()
    plt.close(fig)

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
    # print(int(class_origin))

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
