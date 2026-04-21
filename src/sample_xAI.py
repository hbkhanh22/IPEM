import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import transforms
from PIL import Image
from ipem_explainer import IPEMExplainer
from rise_explainer import RISE
from lime import lime_image
import shap
import cv2
from utils import predict_with_model
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def renormalize_image(img):
    """Normalize an image or heatmap to the 8-bit range used for visualization and file export.
    
    Args:
        img: Image or heatmap array to normalize.
    
    Return:
        np.ndarray | None: Uint8 image when input is valid, otherwise None.
    """
    if img is None:
        return None
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img_uint8 = img_norm.astype(np.uint8)
    return img_uint8

def explain_with_lime(clf, img_np, class_names, output_dir, args_dataset, org_img):
    """Generate a LIME explanation, convert it into a dense heatmap, and save the visualization outputs.
    
    Args:
        clf: Classifier object that owns the trained model.
        img_np: Input image as a NumPy array in HWC format.
        class_names: Ordered list of class names.
        output_dir: Base directory used to store artifacts.
        args_dataset: Dataset name used to organize output folders.
        org_img: Original image used for visualization.
    
    Return:
        tuple: The raw LIME explanation object and its normalized continuous heatmap.
    """
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
        """Convert LIME image samples into class probabilities using the classifier model.
        
        Args:
            imgs: Batch of perturbed images produced by LIME.
        
        Return:
            np.ndarray: Predicted class probabilities for the provided images.
        """
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

    # -- Build a continuous heatmap from superpixel weights -------------------------
    # Each pixel stores the id of the superpixel it belongs to
    segments = explanation.segments                          # shape (H, W), int

    # local_exp[pred_class] = list of (segment_id, weight)
    seg_weight_dict = dict(explanation.local_exp[pred_class])

    # Initialize an empty heatmap and assign weights only to segments in the dictionary
    heatmap_raw = np.zeros(segments.shape, dtype=np.float32)
    for seg_id, weight in seg_weight_dict.items():
        heatmap_raw[segments == seg_id] = weight

    # Keep only positive contributions, similar to positive_only=True
    heatmap_pos = np.maximum(heatmap_raw, 0)

    # Normalize to the [0, 1] range
    max_val = heatmap_pos.max()
    if max_val > 0:
        heatmap_continuous = heatmap_pos / max_val
    else:
        heatmap_continuous = heatmap_pos          # all zeros, which avoids division by zero
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

def explain_with_ipem(clf, img_tensor, class_names, output_dir, args_dataset, org_img):
    """Generate an IPEM explanation and render the resulting counterfactual-style visualization.
    
    Args:
        clf: Classifier object that owns the trained model.
        img_tensor: Input image tensor prepared for the model.
        class_names: Ordered list of class names.
        output_dir: Base directory used to store artifacts.
        args_dataset: Dataset name used to organize output folders.
        org_img: Original image used for visualization.
    
    Return:
        np.ndarray: IPEM heatmap for the input image.
    """
    print("🔍 Đang chạy IPEM...")
    start_time = time.time() 
    if output_dir:
        output_dir = f"{output_dir}/{args_dataset}"
        output_path = Path(output_dir) / "ipem"
        output_path.mkdir(parents=True, exist_ok=True)

    ipem = IPEMExplainer(clf.model, class_names)
    heatmap = ipem.explain_by_watershed(img_tensor.squeeze(0))
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
    """Generate a GradCAM explanation for the current prediction and save the visualization outputs.
    
    Args:
        clf: Classifier object that owns the trained model.
        img_tensor: Input image tensor prepared for the model.
        class_names: Ordered list of class names.
        output_dir: Base directory used to store artifacts.
        args_dataset: Dataset name used to organize output folders.
        org_img: Original image used for visualization.
    
    Return:
        np.ndarray: GradCAM heatmap for the selected target class.
    """
    start_time = time.time()
    print("🔍 Đang chạy GradCAM...")
    if output_dir:
        output_dir = f"{output_dir}/{args_dataset}"
        output_path = Path(output_dir) / "GradCAM"
        output_path.mkdir(parents=True, exist_ok=True)
    
    def get_last_conv_layer(model):
        """Locate the final convolutional layer required by GradCAM.
        
        Args:
            model: Model whose modules are inspected.
        
        Return:
            nn.Module: Last convolutional layer found in the model.
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


def explain_with_rise(clf, img_tensor, class_names, output_dir, args_dataset, org_img):
    """Generate a RISE explanation, select the predicted class heatmap, and save the visualization outputs.
    
    Args:
        clf: Classifier object that owns the trained model.
        img_tensor: Input image tensor prepared for the model.
        class_names: Ordered list of class names.
        output_dir: Base directory used to store artifacts.
        args_dataset: Dataset name used to organize output folders.
        org_img: Original image used for visualization.
    
    Return:
        np.ndarray: RISE heatmap for the predicted class.
    """
    start_time = time.time()
    print("🔍 Đang chạy RISE...")
    if output_dir:
        output_dir = f"{output_dir}/{args_dataset}"
        output_path = Path(output_dir) / "RISE"
        output_path.mkdir(parents=True, exist_ok=True)
    
    rise_explainer = RISE(model=clf.model, n_masks=10000, p=0.5, input_size=(224, 224), initial_mask_size=(7,7), n_batch=64, mask_path=None)
    heatmap_tensor = rise_explainer.explain(img_tensor)
    
    # Get the prediction details to select the heatmap of the predicted class
    pred_class, _, original_probs = predict_with_model(clf.model, img_tensor)
    
    # Convert the tensor to a NumPy array with shape (H, W)
    heatmap = heatmap_tensor[pred_class].cpu().numpy()
    
    end_time = time.time()
    explantion_time = end_time - start_time
    visualize_counterfactual_explanation(
        classifier=clf,
        org_img=org_img,
        heatmap=heatmap,
        model_name="RISE",
        pred_class=pred_class,
        original_probs=original_probs,
        output_path=output_path,
        explanation_time=explantion_time
    )
    return heatmap

def visualize_counterfactual_explanation(classifier, org_img, heatmap, model_name, pred_class, original_probs, output_path=None, explanation_time=None, percentile_threshold=50, alpha=0.5, cmap='jet'):
    """Overlay an explanation heatmap on the original image and save the visualization artifacts.
    
    Args:
        classifier: Classifier wrapper that provides the model and class names.
        org_img: Original image used as the visualization background.
        heatmap: Explanation heatmap to overlay.
        model_name: Name of the explanation method being visualized.
        pred_class: Predicted class index for the original image.
        original_probs: Probability vector predicted for the original image.
        output_path: Optional directory used to save generated figures.
        explanation_time: Optional explanation runtime in seconds.
        percentile_threshold: Threshold parameter reserved for advanced filtering.
        alpha: Opacity used for the heatmap overlay.
        cmap: Matplotlib colormap name used to render the heatmap.
    
    Return:
        None.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not isinstance(org_img, np.ndarray):
        org_img = np.array(org_img)
    
    renormalized_heatmap = renormalize_image(heatmap)
    cv2.imwrite(f"{output_path}/{model_name}_mask.png", renormalized_heatmap)

    H, W = org_img.shape[:2]
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

    # Use gridspec with two equally sized image columns and one narrow colorbar column
    fig, ax0 = plt.subplots(figsize=(15, 6))
    # gs = fig.add_gridspec(
    #     1, 3,
    #     width_ratios=[1, 0.03, 1],  # two equal image columns and one colorbar column
    #     wspace=0.01
    # )

    # ax0 = fig.add_subplot(gs[0])
    # cax = fig.add_subplot(gs[1])  # dedicated axes for the colorbar
    # ax1 = fig.add_subplot(gs[2])
    # cax = fig.add_subplot(gs[2])  # dedicated axes for the colorbar

    # # Original image
    # ax0.imshow(org_img)
    # ax0.set_title("Original Image")
    # ax0.axis("off")

    # Overlay saliency map
    ax0.imshow(org_img, alpha=1.0)
    hm = ax0.imshow(heatmap, cmap=cmap, alpha=alpha)
    ax0.axis("off")

    if output_path:
        fig.savefig(
            f"{output_path}/{model_name}_explanation_clean.png",
            dpi=200,
            bbox_inches="tight",
            pad_inches=0
        )

    ax0.set_title(
        f"{model_name} Explanation Map - Pred: {classifier.class_names[pred_class]}\n"
        f"Probability: {original_probs[pred_class]:.2f} - Explanation time: {explanation_time:.2f}s"
    )

    # # Place the colorbar on dedicated axes so it does not affect ax1 size
    # fig.colorbar(hm, cax=cax)
    
    # ax1.imshow(perturbed_img, alpha=1.0)
    # ax1.set_title(f"Perturbed Image - Pred: {classifier.class_names[new_class]}\n"
    #                 f"Probability of {classifier.class_names[pred_class]} class: {torch.softmax(outputs, dim=1)[0][pred_class]:.2f}")
    # ax1.axis("off")

    plt.tight_layout()

    if output_path:
        plt.savefig(f"{output_path}/{model_name}_explanation.png", dpi=200, bbox_inches="tight")

    fig.colorbar(hm, ax=ax0, fraction=0.046, pad=0.04)
    plt.show()
    plt.close(fig)

def make_perturbation(img, mode='blur', ksize=(11, 11), sigma=50):
    """Apply a perturbation strategy to an image for AOPC-style faithfulness analysis.
    
    Args:
        img: Image region to perturb.
        mode: Perturbation strategy such as blur, noise, zero, or mean.
        ksize: Gaussian kernel size used when blur mode is selected.
        sigma: Gaussian standard deviation used for blur mode.
    
    Return:
        np.ndarray: Perturbed image.
    """
    if mode == 'blur':
        # Ensure the kernel is not larger than the block
        k_h = min(ksize[0], img.shape[0] | 1)  # |1 ensures the kernel size remains odd
        k_w = min(ksize[1], img.shape[1] | 1)

        # Apply Gaussian blur
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
    """Compute an AOPC MoRF curve by progressively perturbing the most relevant image blocks.
    
    Args:
        clf: Classifier object that owns the trained model.
        img_map_path: Path to the explanation heatmap image.
        img_path: Path to the original image.
        mode: Perturbation strategy applied to selected blocks.
        block_size: Side length of each square perturbation block.
        block_per_row: Number of blocks contained in each image row.
        percentile: Optional cumulative importance threshold used to stop early.
        img_size: Spatial size used to resize inputs before evaluation.
        verbose: Whether to print detailed progress information.
    
    Return:
        tuple: Original class index, AOPC curve values, and mean AOPC score.
    """
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
    # If the image is in float format (0-1), rescale it to 0-255 before creating a PIL image
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

def insertion_deletion_score(model, img_tensor, heatmap, target_class, steps=20, mode="insertion"):
    """Compute insertion or deletion faithfulness scores by revealing or removing pixels in importance order.
    
    Args:
        model: Trained model used for scoring perturbed images.
        img_tensor: Original image tensor in CHW or NCHW format.
        heatmap: Explanation heatmap associated with the image.
        target_class: Class index whose probability is tracked.
        steps: Number of perturbation steps used to build the curve.
        mode: Either insertion or deletion.
    
    Return:
        tuple[float, list[float]]: AUC score and per-step probabilities.
    """
    device = next(model.parameters()).device

    # Prepare the transform for compatibility with the existing pipeline
    transform = transforms.Compose([
        transforms.Resize((img_tensor.shape[1] if img_tensor.dim() > 2 else img_tensor.shape[2], img_tensor.shape[2] if img_tensor.dim() > 2 else img_tensor.shape[1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Prepare the baseline image (all zeros) for both insertion and deletion
    baseline = torch.zeros_like(img_tensor).to(device)

    # Prepare the heatmap by validating it and converting it to a single channel
    hm = np.array(heatmap)
    if hm.size == 0:
        raise ValueError("Heatmap is empty or invalid for resizing")
    # If heatmap has channels (e.g., C,H,W or H,W,C), average to single channel
    if hm.ndim == 3:
        # try to detect channel-first (C,H,W) or channel-last (H,W,C)
        if hm.shape[0] <= 4 and hm.shape[0] != hm.shape[1]:
            # likely (C,H,W)
            hm = hm.mean(axis=0)
        else:
            # likely (H,W,C)
            hm = hm.mean(axis=2)
    # Replace NaN/inf with zeros
    hm = np.nan_to_num(hm, nan=0.0, posinf=0.0, neginf=0.0)
    # Normalize to [0,1] if possible
    if hm.max() - hm.min() > 0:
        hm = (hm - hm.min()) / (hm.max() - hm.min())
    else:
        hm = np.zeros_like(hm)

    # Resize heatmap to match image size
    # Determine target width/height from img_tensor shape
    if img_tensor.dim() == 4:
        _, C, H, W = img_tensor.shape
    elif img_tensor.dim() == 3:
        C, H, W = img_tensor.shape
    else:
        raise ValueError(f"Unsupported img_tensor shape: {img_tensor.shape}")

    target_w, target_h = int(W), int(H)
    try:
        # cv2.resize expects (width, height); convert to uint8 to avoid issues
        heatmap_resized = cv2.resize((hm * 255).astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    except Exception as e:
        raise RuntimeError(f"Failed to resize heatmap: {e}")

    heatmap_flat = heatmap_resized.flatten()
    indices = np.argsort(-heatmap_flat)  # sort in descending order

    # Compute probabilities across the perturbation steps
    probs = []
    for step in range(steps + 1):
        fraction = step / steps
        num_pixels = int(fraction * len(heatmap_flat))

        mask = np.zeros_like(heatmap_flat)
        mask[indices[:num_pixels]] = 1.0
        mask = mask.reshape(heatmap_resized.shape)

        # Create a mask tensor with a shape that supports broadcasting
        mask_t = torch.tensor(mask, dtype=img_tensor.dtype, device=device)
        if img_tensor.dim() == 4:
            # make shape (1,1,H,W) so it broadcasts to (1,C,H,W)
            mask_t = mask_t.unsqueeze(0).unsqueeze(1)
        else:
            # img_tensor dim==3 (C,H,W) -> make mask (1,H,W) which will broadcast across channels
            mask_t = mask_t.unsqueeze(0)

        # Create the perturbed image
        if mode == "insertion":
            # start from baseline and insert pixels from original
            perturbed_img = baseline * (1 - mask_t) + img_tensor * mask_t
        else:  # deletion
            # start from original and replace masked pixels with baseline
            perturbed_img = img_tensor * (1 - mask_t) + baseline * mask_t

        # Ensure perturbed_img has batch dimension when passing to model
        if perturbed_img.dim() == 3:
            perturbed_in = perturbed_img.unsqueeze(0)
        else:
            perturbed_in = perturbed_img

        with torch.no_grad():
            output = model(perturbed_in)
            prob = torch.softmax(output, dim=1)[0, target_class].item()

        probs.append(prob)
    # Compute the AUC
    auc_score = np.trapezoid(probs, dx=1.0/steps)
    return auc_score, probs
