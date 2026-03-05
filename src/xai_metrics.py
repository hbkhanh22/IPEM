import numpy as np
import torch
from typing import List
from lime import lime_image
import shap
from pebex_explainer import PEBEXExplainer  # đã viết ở trên
from sklearn.metrics import accuracy_score
import time
import cv2
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch.nn as nn
from utils import _make_perturbation, vectorize_explanation, predict_proba_fn

class XAIEvaluator:
    def __init__(self, model: torch.nn.Module, class_names: List[str]):
        self.model = model.eval()   
        self.class_names = class_names

    # --------- Các hàm tính metric chung ----------
    @staticmethod
    def fidelity(y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def comprehensibility(used_features):
        return np.mean(used_features), np.std(used_features)

    @staticmethod
    def consistency(expl_vecs):
        sims = []
        for i in range(len(expl_vecs)-1):
            sims.append(np.dot(expl_vecs[i], expl_vecs[i+1]) /
                        (np.linalg.norm(expl_vecs[i]) * np.linalg.norm(expl_vecs[i+1]) + 1e-8))
        return np.mean(sims), np.std(sims)

    @staticmethod
    def stability(expl_vecs):
        diffs = []
        for i in range(len(expl_vecs)-1):
            diffs.append(np.linalg.norm(expl_vecs[i]-expl_vecs[i+1]))
        return np.mean(diffs)  # càng nhỏ càng ổn định

    @staticmethod
    def similarity(expl_vecs):
        return XAIEvaluator.consistency(expl_vecs)

    @staticmethod
    def robustness(expl_vecs):
        norms = [np.linalg.norm(v) for v in expl_vecs]
        return np.mean(norms)

    @staticmethod
    def fcc_score(fid, comp, cons):
        return {
            "FCC": fid * comp * cons
        }
    
    # -----------CAM Metrics ----------
    @staticmethod
    def average_drop_increase(original_probs, masked_probs):
        """
        Tính Average Drop và Increase từ xác suất gốc và xác suất sau khi mask.
        Cả 2 inputs đều là list hoặc numpy array.
        """
        original_probs = np.array(original_probs)
        masked_probs = np.array(masked_probs)

        drops = np.maximum(0, original_probs - masked_probs)
        avg_drop = np.mean(drops / (original_probs + 1e-8)) * 100  # phần trăm

        increases = (masked_probs > original_probs).astype(np.float32)
        increase_rate = np.mean(increases) * 100  # phần trăm

        return avg_drop, increase_rate
    
    @staticmethod
    def insertion_deletion_score(self, img_tensor, heatmap, target_class, steps=20, mode="insertion"):
        """
        Tính Insertion/Deletion score cho một ảnh và heatmap.
        - img_tensor: torch.Tensor, shape (C,H,W), ảnh gốc.
        - heatmap: numpy array, shape (H,W), heatmap giải thích.
        - target_class: int, lớp mục tiêu để theo dõi xác suất.
        - steps: số bước perturbation.
        - mode: "insertion" hoặc "deletion".
        Trả về AUC score.
        """
        device = next(self.model.parameters()).device

        # Chuẩn bị transform
        transform = transforms.Compose([
            transforms.Resize((img_tensor.shape[1], img_tensor.shape[2])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Chuẩn bị ảnh nền (đen)
        if mode == "insertion":
            base_img = torch.zeros_like(img_tensor).to(device)
        else:  # deletion
            base_img = img_tensor.clone().to(device)

        # Chuẩn bị heatmap
        heatmap_resized = cv2.resize(heatmap, (img_tensor.shape[2], img_tensor.shape[1]))
        heatmap_flat = heatmap_resized.flatten()
        indices = np.argsort(-heatmap_flat)  # sắp xếp giảm dần

        # Tính xác suất qua các bước
        probs = []
        for step in range(steps + 1):
            fraction = step / steps
            num_pixels = int(fraction * len(heatmap_flat))

            mask = np.zeros_like(heatmap_flat)
            mask[indices[:num_pixels]] = 1.0
            mask = mask.reshape(heatmap_resized.shape)

            # Tạo ảnh perturbed
            if mode == "insertion":
                perturbed_img = base_img * (1 - torch.tensor(mask).to(device)) + img_tensor * torch.tensor(mask).to(device)
            else:  # deletion
                perturbed_img = img_tensor * (1 - torch.tensor(mask).to(device)) + base_img * torch.tensor(mask).to(device)

            perturbed_img = perturbed_img.unsqueeze(0)  # thêm batch dim

            with torch.no_grad():
                output = self.model(perturbed_img)
                prob = torch.softmax(output, dim=1)[0, target_class].item()

            probs.append(prob)
        # Tính AUC
        auc_score = np.trapz(probs, dx=1.0/steps)
        return auc_score
    
    # --------- AOPC MoRF cho toàn bộ test_loader ----------
    def _compute_single_aopc(self, img_tensor, explanation, original_class, original_prob, block_size=8, percentile=None, verbose=False):
        """Tính AOPC cho một ảnh đơn lẻ"""
        device = next(self.model.parameters()).device
        
        # Xử lý img_tensor
        if isinstance(img_tensor, torch.Tensor):
            if img_tensor.dim() == 3:  # (C, H, W)
                img_size = img_tensor.shape[-1]  # giả sử ảnh vuông
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            else:
                raise ValueError(f"Unsupported img_tensor shape: {img_tensor.shape}")
        else:
            img_np = img_tensor
            img_size = img_np.shape[0]  # giả sử ảnh vuông
        
        block_per_row = img_size // block_size
        
        # Chuyển explanation thành numpy array
        if isinstance(explanation, torch.Tensor):
            explanation = explanation.detach().cpu().numpy()
        
        # Resize explanation về đúng kích thước
        if explanation.shape != (img_size, img_size):
            explanation = cv2.resize(explanation, (img_size, img_size))
        
        # Tạo blocks từ explanation
        blocks_dict = {}
        key = 0
        
        for i in range(0, img_size, block_size):
            for j in range(0, img_size, block_size):
                block = explanation[i:i + block_size, j:j + block_size]
                block_sum = np.sum(block)
                blocks_dict[key] = (block, block_sum)
                key += 1
        
        # Sắp xếp blocks theo importance
        sorted_blocks = sorted(blocks_dict.items(), key=lambda x: x[1][1], reverse=True)
        total_sum_blocks = sum(block_sum[1] for block_sum in list(blocks_dict.values()))
        
        # Tính percentile breaking point nếu có
        breaking_pct_value = None
        if percentile is not None:
            breaking_pct_value = (percentile / 100) * total_sum_blocks
        
        # Chuẩn bị ảnh để perturb
        img_pred_aopc = img_np.copy()
        
        # Transform để đưa vào model
        transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Xử lý original_class và original_prob
        if isinstance(original_class, torch.Tensor):
            original_class = original_class.item()
        if isinstance(original_prob, torch.Tensor):
            original_prob = original_prob.cpu().numpy()
        
        # Tính AOPC
        AOPC = []
        count_blocks = 0
        sum_blocks = 0
        
        for i in range(len(blocks_dict)):
            ref_block_index = sorted_blocks[i][0]
            row = ref_block_index // block_per_row
            col = ref_block_index % block_per_row
            
            # Perturb block này
            img_pred_aopc[row * block_size:row * block_size + block_size, 
                         col * block_size:col * block_size + block_size] = _make_perturbation(
                img_pred_aopc[row * block_size:row * block_size + block_size, 
                             col * block_size:col * block_size + block_size]
            )
            
            # Đưa vào model
            img_pil = Image.fromarray((img_pred_aopc * 255).astype(np.uint8))
            input_tensor = transform(img_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = self.model(input_tensor)
            
            probs = torch.softmax(output, dim=1)
            class_prob_after = probs[0, original_class].item()
            
            # Tính delta
            original_prob_val = original_prob[original_class]
            delta = original_prob_val - class_prob_after
            AOPC.append(delta)
            
            count_blocks += 1
            
            # Kiểm tra breaking condition
            if percentile is not None:
                sum_blocks += np.sum(sorted_blocks[i][1][0])
                if sum_blocks >= breaking_pct_value:
                    break
        
        return (original_class, AOPC, np.sum(AOPC) / count_blocks)

    # --------- LIME ----------
    def evaluate_with_lime(self, samples, num_samples=1000, compute_aopc=True, block_size=8, percentile=None):
        device = next(self.model.parameters()).device

        explainer = lime_image.LimeImageExplainer()
        results = {}

        y_model, y_local = [], []
        expl_vecs, used_feats = [], []
        all_aopc_scores = []  # lưu original probabilities
        orig_probs_list, masked_probs_list = [], []
        masks_list, imgs_list = [], []
        
        time_start = time.time()
        for idx, (img_t, label) in enumerate(samples):
            # img_t shape: (C,H,W)
            if img_t.dim() == 2:  # grayscale
                img_t = img_t.unsqueeze(0)  # thêm channel

            img_np = img_t.permute(1,2,0).cpu().numpy()  # (H,W,C)

            # Chạy lime
            explanation = explainer.explain_instance(
                img_np,
                classifier_fn=lambda ims: predict_proba_fn(self.model, ims),
                top_labels=len(self.class_names),
                hide_color=0,
                num_samples=num_samples,
            )

            top_label = np.argmax(predict_proba_fn(self.model, np.array([img_np]))[0])

            # lấy mask để vector hóa
            temp, mask = explanation.get_image_and_mask(
                label=top_label, positive_only=True, num_features=8, hide_rest=False
            )

            masks_list.append(mask)
            imgs_list.append(img_t)
            expl_vecs.append(vectorize_explanation(mask))
            used_feats.append(np.count_nonzero(mask))

            y_model.append(label)
            y_local.append(top_label)
            
            # Lưu data để tính AOPC
            if compute_aopc:
                with torch.no_grad():
                    img_tensor = img_t.unsqueeze(0).to(device)
                    original_output = self.model(img_tensor)
                    original_prob = torch.softmax(original_output, dim=1)[0]
                # compute masked prob for average drop/increase
                try:
                    bin_mask = (mask > 0).astype(np.float32)
                    mask_t = torch.from_numpy(bin_mask).to(device).unsqueeze(0).unsqueeze(0)
                    mask_t = mask_t.expand(1, img_tensor.shape[1], bin_mask.shape[0], bin_mask.shape[1])
                    with torch.no_grad():
                        masked_img = img_tensor * mask_t
                        masked_out = self.model(masked_img)
                        masked_prob = torch.softmax(masked_out, dim=1)[0, top_label].item()
                    orig_probs_list.append(float(original_prob[top_label].item()))
                    masked_probs_list.append(float(masked_prob))
                except Exception:
                    pass
                _, AOPC_list, mean_AOPC = self._compute_single_aopc(
                    img_tensor=img_tensor.squeeze(0),
                    explanation=mask,
                    original_class=top_label,
                    original_prob=original_prob,
                    block_size=block_size,
                    percentile=percentile,
                    verbose=False
                )
                
                all_aopc_scores.append(mean_AOPC)
            
        mu_comp, _ = self.comprehensibility(used_feats)
        mu_cons, _ = self.consistency(expl_vecs)
        time_end = time.time()
        exp_time = time_end - time_start

        results = {
            "LIME_Fidelity": self.fidelity(y_model, y_local),
            "LIME_Comprehensibility": mu_comp,
            "LIME_Consistency": mu_cons,
            #"LIME_LocalAcc": self.local_accuracy(y_model, y_local),
            "LIME_Stability": self.stability(expl_vecs),
            # "LIME_Similarity": mu_cons,
            "LIME_Robustness": self.robustness(expl_vecs),
            "LIME_Time": exp_time,
        }

        # --- Thêm AOPC nếu có ---
        if compute_aopc and len(all_aopc_scores) > 0:
            results["LIME_AOPC_Mean"] = float(np.mean(all_aopc_scores))
        # --- Thêm Average Drop/Increase nếu có ---
        if len(orig_probs_list) > 0:
            try:
                avg_drop, inc_rate = self.average_drop_increase(orig_probs_list, masked_probs_list)
                results["LIME_AvgDrop"] = float(avg_drop)
                results["LIME_IncreaseRate"] = float(inc_rate)
            except Exception:
                pass

        # --- Thêm Insertion/Deletion nếu có ---
        lime_ins, lime_del = [], []
        try:
            for i in range(len(imgs_list)):
                img_t = imgs_list[i]
                mask = masks_list[i]
                top_label = int(y_local[i])
                # normalize mask to 2D
                if hasattr(mask, 'ndim') and mask.ndim == 3:
                    heat = mask[:, :, 0]
                else:
                    heat = mask

                try:
                    ins = self.insertion_deletion_score(img_t.to(next(self.model.parameters()).device), heat, top_label, mode="insertion")
                    de = self.insertion_deletion_score(img_t.to(next(self.model.parameters()).device), heat, top_label, mode="deletion")
                    lime_ins.append(float(ins))
                    lime_del.append(float(de))
                except Exception:
                    continue
        except Exception:
            pass

        print(lime_ins, lime_del)
        
        if len(lime_ins) > 0:
            results["LIME_Insertion"] = float(np.mean(lime_ins))
        if len(lime_del) > 0:
            results["LIME_Deletion"] = float(np.mean(lime_del))
        
        results.update(self.fcc_score(
            results["LIME_Fidelity"],
            results["LIME_Comprehensibility"],
            results["LIME_Consistency"]
        ))
        return results


    # --------- SHAP ----------
    def evaluate_with_shap(self, loader, batch_size: int = 8, noise_sigma: float = 0.02, large_mask_prob: float = 0.2, 
                          compute_aopc=True, block_size=8, percentile=None):
        """
        Evaluate SHAP explanations on one batch from `loader`.
        - Hỗ trợ SHAP trả về 5D (N, C, H, W, num_classes), list (num_classes of (N,C,H,W)), hoặc 4D (N,C,H,W).
        - Tính Fidelity (accuracy giữa y_model và y_pred_from_shap), Comprehensibility, Consistency,
        LocalAcc, Stability, Similarity, Robustness, và FCC.
        """

        device = next(self.model.parameters()).device

        # 1) Lấy batch mẫu
        imgs, _ = next(iter(loader))
        imgs = imgs.to(device)

        N = imgs.shape[0]

        # pred_class, _, original_probs = predict_with_model(self.model, imgs)

        # 2) Chọn background cho SHAP (dùng 2 ảnh đầu trong batch nếu có)
        bg = imgs[:2].to(device)

        # 3) Tạo GradientExplainer và tính shap_values cho batch gốc
        time_start = time.time()
        print(f"Starting SHAP ...")
        explainer = shap.GradientExplainer(self.model, batch_size=batch_size, data=bg)
       
        raw_shap = explainer.shap_values(imgs)  # có thể là list / tensor / numpy
        print(f"SHAP completed in {time.time() - time_start} seconds")
        
        def to_shap_per_class_array(svalues):
            """
            Chuẩn hóa SHAP output về dạng:
            (num_classes, N, C, H, W)
            """
            # Case 1: list[num_classes] of (N,C,H,W)
            if isinstance(svalues, list):
                shap_array = np.stack([
                    sv.detach().cpu().numpy() if torch.is_tensor(sv) else sv
                    for sv in svalues
                ], axis=0)
                # -> (num_classes, N, C, H, W)

            # Case 2: tensor / numpy 5D
            elif isinstance(svalues, (np.ndarray, torch.Tensor)):
                if torch.is_tensor(svalues):
                    svalues = svalues.detach().cpu().numpy()

                if svalues.ndim == 5:
                    # Common layouts:
                    # - (N, C, H, W, num_classes) -> transpose to (num_classes, N, C, H, W)
                    # - (num_classes, N, C, H, W) -> already correct
                    # Try to disambiguate using known number of classes if available.
                    try:
                        n_classes = len(self.class_names)
                    except Exception:
                        n_classes = None

                    # If first dim equals num_classes -> assume (num_classes, N, C, H, W)
                    if n_classes is not None and svalues.shape[0] == n_classes:
                        shap_array = svalues
                    # If last dim equals num_classes -> assume (N, C, H, W, num_classes)
                    elif n_classes is not None and svalues.shape[-1] == n_classes:
                        shap_array = np.transpose(svalues, (4, 0, 1, 2, 3))
                    # Fallback: if last dim is >1 and likely class axis, transpose it
                    elif svalues.shape[-1] > 1 and svalues.shape[0] != svalues.shape[-1]:
                        shap_array = np.transpose(svalues, (4, 0, 1, 2, 3))
                    else:
                        raise ValueError(f"Ambiguous SHAP shape: {svalues.shape}")
                elif svalues.ndim == 4:
                    # Single-output model
                    shap_array = svalues[None, ...]  # (1,N,C,H,W)
                else:
                    raise ValueError(f"Unsupported SHAP ndim: {svalues.ndim}")

            else:
                raise TypeError(f"Unsupported SHAP type: {type(svalues)}")

            return shap_array
        
        shap_per_class = to_shap_per_class_array(raw_shap)  # (num_classes, N, C, H, W)
        num_classes = shap_per_class.shape[0]

        # 4) Compute model predictions & expected_value using background logits mean
        with torch.no_grad():
            y_model_logits = self.model(imgs)                       # (N, num_classes)
            y_model_probs = torch.softmax(y_model_logits, dim=1).cpu().numpy()
            y_model = np.argmax(y_model_probs, axis=1)              # (N,)

            # expected_value: mean logits on background (shape: num_classes,)
            if bg is not None and bg.shape[0] > 0:
                with torch.no_grad():
                    bg_logits = self.model(bg)                      # (B, num_classes)
                expected_value = bg_logits.mean(0).cpu().numpy()
            else:
                expected_value = np.zeros((num_classes,), dtype=float)
        print("Done expected_value")
        # 5) Tái tạo logits từ shap: y_local_logits[n, c] = expected_value[c] + sum_all_contribs_for_class_c(sample n)
        y_local_logits = np.zeros_like(y_model_probs)  # (N, num_classes)
        for c in range(num_classes):
            # shap_per_class[c] shape (N, C, H, W)
            arr = shap_per_class[c]               # numpy
            contrib = arr.reshape(N, -1).sum(axis=1)   # (N,)
            y_local_logits[:, c] = expected_value[c] + contrib
        print("Done y_local_logits")
        # softmax -> y_local labels
        y_local_probs = np.exp(y_local_logits)
        y_local_probs = y_local_probs / y_local_probs.sum(axis=1, keepdims=True)
        y_local = np.argmax(y_local_probs, axis=1)
        print("Done y_local")

        # 6) Tạo heatmaps per-sample theo predicted class (tập trung vào class dự đoán của model)
        # heatmap_i = sum_k |shap_per_class[pred_class_i][i, channel=k, :, :]| over channels
        heatmaps = []
        for i in range(N):
            p = int(y_model[i])
            arr = shap_per_class[p][i]            # (C, H, W)
            heat = np.abs(arr).sum(axis=0)        # (H, W)
            heatmaps.append(heat)
        print("Done heatmaps")
        # vectorize explanations
        expl_vecs = [vectorize_explanation(h) for h in heatmaps]
        used_feats = [int((np.abs(h) > 1e-6).sum()) for h in heatmaps]
        mu_comp, _ = self.comprehensibility(used_feats)
        mu_cons, _ = self.consistency(expl_vecs)
        print("Done mu_comp and mu_cons")
        # 7) Stability: recompute SHAP on slightly noised images and compare vectors (L2)
        # create small noise in normalized space
        imgs_noisy = (imgs + torch.randn_like(imgs) * noise_sigma).clamp(-5.0, 5.0)  # clamp in normalized space
        subset = torch.randperm(N)[: max(1, N // 10)]  # ví dụ 10%
        raw_shap_noisy = explainer.shap_values(imgs_noisy[subset])
        # raw_shap_noisy = explainer.shap_values(imgs_noisy)
        shap_noisy_per_class = to_shap_per_class_array(raw_shap_noisy)

        with torch.no_grad():
            y_logits_sub = self.model(imgs_noisy[subset])
            y_probs_sub = torch.softmax(y_logits_sub, dim=1).cpu().numpy()
            y_model_sub = np.argmax(y_probs_sub, axis=1)   # (M,)


        # noise = np.random.normal(scale=noise_sigma, size=shap_per_class.shape)
        # shap_noisy_per_class = shap_per_class + noise
        M = len(subset)
        # chọn shap theo predicted class cho từng ảnh trong subset
        sel = shap_noisy_per_class[y_model_sub, np.arange(M)]   # (M, C, H, W)
        heatmaps_noisy = np.abs(sel).sum(axis=1)                # (M, H, W)

        # vectorize explanations
        vecs_noisy = [vectorize_explanation(h) for h in heatmaps_noisy]

        # so sánh với expl_vecs tương ứng (chọn cùng subset)
        expl_vecs_sub = [expl_vecs[i] for i in subset.cpu().numpy()]
        stability_val = float(np.mean([np.linalg.norm(a - b) for a, b in zip(expl_vecs_sub, vecs_noisy)]))
        print("Done stability_val")
        
        
        # 8) Robustness: large perturbation (mask ~ large_mask_prob) then recompute SHAP, compare
        mask = (torch.rand_like(imgs[:, :1, :, :]) > (1.0 - large_mask_prob)).float().to(device)  # mask  ~large_mask_prob True
        imgs_largep = imgs * (1.0 - mask)
        subset = torch.randperm(N)[: max(1, N // 10)]  # ví dụ 10%
        raw_shap_largep = explainer.shap_values(imgs_largep[subset])
        shap_largep_per_class = to_shap_per_class_array(raw_shap_largep)


        with torch.no_grad():
            y_logits_sub = self.model(imgs_largep[subset])
            y_probs_sub = torch.softmax(y_logits_sub, dim=1).cpu().numpy()
            y_model_sub = np.argmax(y_probs_sub, axis=1)   # (M,)

        M = len(subset)
        # chọn shap theo predicted class cho từng ảnh trong subset
        sel = shap_largep_per_class[y_model_sub, np.arange(M)]   # (M, C, H, W)
        heatmaps_largep = np.abs(sel).sum(axis=1)                # (M, H, W)

        vecs_largep = [vectorize_explanation(h) for h in heatmaps_largep]

        expl_vecs_sub = [expl_vecs[i] for i in subset.cpu().numpy()]
        robustness_val = float(np.mean([np.linalg.norm(a - b) for a, b in zip(expl_vecs_sub, vecs_largep)]))
        print("Done robustness_val")

        # Tính AOPC nếu được yêu cầu
        all_aopc_scores = []
        shap_orig_probs, shap_masked_probs = [], []
        shap_ins, shap_del = [], []
        if compute_aopc and len(heatmaps) > 0:
            for i in range(N):
                img_tensor = imgs[i]
                mask = heatmaps[i]
                original_class = int(y_model[i])
                original_prob_tensor = torch.tensor(y_model_probs[i], device=device)

                # compute masked prob for average drop/increase
                try:
                    bin_mask = (mask > np.mean(mask)).astype(np.float32)
                    img_t = img_tensor.unsqueeze(0)
                    mask_t = torch.from_numpy(bin_mask).to(device).unsqueeze(0).unsqueeze(0)
                    mask_t = mask_t.expand(1, img_t.shape[1], bin_mask.shape[0], bin_mask.shape[1])
                    with torch.no_grad():
                        masked_img = img_t * mask_t
                        masked_out = self.model(masked_img)
                        masked_prob = torch.softmax(masked_out, dim=1)[0, original_class].item()
                    shap_orig_probs.append(float(y_model_probs[i][original_class]))
                    shap_masked_probs.append(float(masked_prob))
                except Exception:
                    pass

                _, _, mean_AOPC = self._compute_single_aopc(
                    img_tensor,
                    mask,
                    original_class,
                    original_prob_tensor,
                    block_size=block_size,
                    percentile=percentile,
                    verbose=False
                )
                all_aopc_scores.append(mean_AOPC)
        
        time_end = time.time()
        exp_time = time_end - time_start

        results = {
            "SHAP_Fidelity": float(self.fidelity(y_model, y_local)),
            "SHAP_Comprehensibility": float(mu_comp),
            "SHAP_Consistency": float(mu_cons),
            "SHAP_Stability": float(stability_val),
            "SHAP_Robustness": float(robustness_val),
            "SHAP_Time": exp_time,
        }

        if len(all_aopc_scores) > 0:
            results["SHAP_AOPC_Mean"] = float(np.mean(all_aopc_scores))

        # --- Thêm Average Drop/Increase nếu có ---
        if len(shap_orig_probs) > 0:
            try:
                avg_drop, inc_rate = self.average_drop_increase(shap_orig_probs, shap_masked_probs)
                results["SHAP_AvgDrop"] = float(avg_drop)
                results["SHAP_IncreaseRate"] = float(inc_rate)
            except Exception:
                pass

        # --- Thêm Insertion/Deletion nếu có ---
        try:
            for i in range(len(heatmaps)):
                try:
                    ins = self.insertion_deletion_score(imgs[i], heatmaps[i], int(y_model[i]), mode="insertion")
                    de = self.insertion_deletion_score(imgs[i], heatmaps[i], int(y_model[i]), mode="deletion")
                    shap_ins.append(float(ins))
                    shap_del.append(float(de))
                except Exception:
                    continue
        except Exception:
            pass

        if len(shap_ins) > 0:
            results["SHAP_Insertion"] = float(np.mean(shap_ins))
        if len(shap_del) > 0:
            results["SHAP_Deletion"] = float(np.mean(shap_del))

        # FCC (sửa: truyền float, không truyền list)
        fcc = self.fcc_score(
            results["SHAP_Fidelity"],
            results["SHAP_Comprehensibility"],
            results["SHAP_Consistency"]
        )
        results.update(fcc)

        return results

    # --------- PEBEX ----------
    def evaluate_with_pebex(self, loader, compute_aopc=True, block_size=8, percentile=None):
        pebex = PEBEXExplainer(self.model, self.class_names)
        device = next(self.model.parameters()).device
        all_expl_vecs, all_used_feats = [], []
        all_y_model, all_y_local = [], []
        all_aopc_scores = []  # lưu AOPC-MoRF score
        pebex_orig_probs, pebex_masked_probs = [], []
        pebex_ins, pebex_del = [], []
        time_start = time.time()

        for batch_imgs, batch_labels in loader:
            imgs = batch_imgs.to(device)
            y_model = self.model(imgs).argmax(1).cpu().numpy()
            y_local = []

            for i in range(len(imgs)):
                # explain_one bây giờ đã trả về pred_label được tính từ importance scores
                heat, pred_from_importance = pebex.explain_one_mc(imgs[i].to(device))
                
                all_expl_vecs.append(vectorize_explanation(np.abs(heat)))
                all_used_feats.append(np.count_nonzero(np.abs(heat) > 1e-6))
                y_local.append(pred_from_importance)
                
                # Lưu data để tính AOPC
                if compute_aopc:
                    # Tính original probability
                    with torch.no_grad():
                        img_tensor = imgs[i].unsqueeze(0).to(device)
                        original_output = self.model(img_tensor)
                        original_prob = torch.softmax(original_output, dim=1)[0]
                        original_class = np.argmax(original_prob.cpu().numpy())

                    # compute masked prob for average drop/increase
                    try:
                        bin_mask = (np.abs(heat) > np.mean(np.abs(heat))).astype(np.float32)
                        mask_t = torch.from_numpy(bin_mask).to(device).unsqueeze(0).unsqueeze(0)
                        mask_t = mask_t.expand(1, img_tensor.shape[1], bin_mask.shape[0], bin_mask.shape[1])
                        with torch.no_grad():
                            masked_img = img_tensor * mask_t
                            masked_out = self.model(masked_img)
                            masked_prob = torch.softmax(masked_out, dim=1)[0, original_class].item()
                        pebex_orig_probs.append(float(original_prob[original_class].item()))
                        pebex_masked_probs.append(float(masked_prob))
                    except Exception:
                        pass

                    _, _, mean_AOPC = self._compute_single_aopc(
                        img_tensor=img_tensor.squeeze(0),
                        explanation=heat,
                        original_class=original_class,
                        original_prob=original_prob,
                        block_size=block_size,
                        percentile=percentile,
                        verbose=False
                    )
                    all_aopc_scores.append(mean_AOPC)
                    # insertion/deletion
                    try:
                        ins = self.insertion_deletion_score(imgs[i], heat, original_class, mode="insertion")
                        de = self.insertion_deletion_score(imgs[i], heat, original_class, mode="deletion")
                        pebex_ins.append(float(ins))
                        pebex_del.append(float(de))
                    except Exception:
                        pass
            
            all_y_local.extend(y_local)
            all_y_model.extend(y_model)

        time_end = time.time()
        exp_time = time_end - time_start

        mu_comp, _ = self.comprehensibility(all_used_feats)
        mu_cons, _ = self.consistency(all_expl_vecs)

        results = {
            "PEBEX_Fidelity": self.fidelity(all_y_model, all_y_local),
            "PEBEX_Comprehensibility": mu_comp,
            "PEBEX_Consistency": mu_cons,
            "PEBEX_Stability": self.stability(all_expl_vecs),
            "PEBEX_Robustness": self.robustness(all_expl_vecs),
            "PEBEX_Time": exp_time,
        }

        # --- Thêm AOPC nếu có ---
        if compute_aopc and len(all_aopc_scores) > 0:
            results["PEBEX_AOPC_Mean"] = float(np.mean(all_aopc_scores))
        # --- Thêm Average Drop/Increase nếu có ---
        if len(pebex_orig_probs) > 0:
            try:
                avg_drop, inc_rate = self.average_drop_increase(pebex_orig_probs, pebex_masked_probs)
                results["PEBEX_AvgDrop"] = float(avg_drop)
                results["PEBEX_IncreaseRate"] = float(inc_rate)
            except Exception:
                pass
        # --- Thêm Insertion/Deletion nếu có ---
        if len(pebex_ins) > 0:
            results["PEBEX_Insertion"] = float(np.mean(pebex_ins))
        if len(pebex_del) > 0:
            results["PEBEX_Deletion"] = float(np.mean(pebex_del))
        results.update(self.fcc_score(results["PEBEX_Fidelity"], results["PEBEX_Comprehensibility"], results["PEBEX_Consistency"]))
        return results

    def evaluate_with_GradCAM(self, loader, compute_aopc=True, block_size=8, percentile=None):
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

        device = next(self.model.parameters()).device

        target_layer = get_last_conv_layer(self.model)

        cam = GradCAM(model=self.model, target_layers=[target_layer])

        all_expl_vecs, all_used_feats = [], []
        all_y_model, all_y_local = [], []
        all_aopc_scores = []
        gradcam_orig_probs, gradcam_masked_probs = [], []
        gradcam_ins, gradcam_del = [], []

        time_start = time.time()

        for imgs, labels in loader:
            imgs = imgs.to(device)

            with torch.no_grad():
                logits = self.model(imgs)
                y_model = logits.argmax(1).cpu().numpy()

            for i in range(len(imgs)):
                img = imgs[i].unsqueeze(0)
                target = [ClassifierOutputTarget(y_model[i])]
                heatmap = cam(input_tensor=img, targets=target)[0]

                all_expl_vecs.append(vectorize_explanation(heatmap))
                all_used_feats.append(np.count_nonzero(heatmap > 1e-6))

                all_y_local.append(int(y_model[i]))

                # AOPC
                if compute_aopc:
                    with torch.no_grad():
                        original_output = self.model(img)
                        original_prob = torch.softmax(original_output, dim=1)[0]

                    # compute masked prob for average drop/increase
                    try:
                        bin_mask = (heatmap > np.mean(heatmap)).astype(np.float32)
                        img_t = img
                        mask_t = torch.from_numpy(bin_mask).to(device).unsqueeze(0).unsqueeze(0)
                        mask_t = mask_t.expand(1, img_t.shape[1], bin_mask.shape[0], bin_mask.shape[1])
                        with torch.no_grad():
                            masked_img = img_t * mask_t
                            masked_out = self.model(masked_img)
                            masked_prob = torch.softmax(masked_out, dim=1)[0, int(y_model[i])].item()
                        gradcam_orig_probs.append(float(original_prob[int(y_model[i])].item()))
                        gradcam_masked_probs.append(float(masked_prob))
                    except Exception:
                        pass

                    _, _, mean_AOPC = self._compute_single_aopc(
                        img_tensor=imgs[i],
                        explanation=heatmap,
                        original_class=int(y_model[i]),
                        original_prob=original_prob,
                        block_size=block_size,
                        percentile=percentile,
                        verbose=False
                    )
                    all_aopc_scores.append(mean_AOPC)
                    # insertion/deletion
                    try:
                        ins = self.insertion_deletion_score(imgs[i], heatmap, int(y_model[i]), mode="insertion")
                        de = self.insertion_deletion_score(imgs[i], heatmap, int(y_model[i]), mode="deletion")
                        gradcam_ins.append(float(ins))
                        gradcam_del.append(float(de))
                    except Exception:
                        pass
            
            all_y_model.extend(y_model)

        time_end = time.time()

        mu_comp, _ = self.comprehensibility(all_used_feats)
        mu_cons, _ = self.consistency(all_expl_vecs)

        results = {
            "GradCAM_Fidelity": self.fidelity(all_y_model, all_y_local),
            "GradCAM_Comprehensibility": mu_comp,
            "GradCAM_Consistency": mu_cons,
            "GradCAM_Stability": self.stability(all_expl_vecs),
            "GradCAM_Robustness": self.robustness(all_expl_vecs),
            "GradCAM_Time": time_end - time_start,
        }

        if compute_aopc and len(all_aopc_scores) > 0:
            results["GradCAM_AOPC_Mean"] = float(np.mean(all_aopc_scores))

        # --- Thêm Average Drop/Increase nếu có ---
        if len(gradcam_orig_probs) > 0:
            try:
                avg_drop, inc_rate = self.average_drop_increase(gradcam_orig_probs, gradcam_masked_probs)
                results["GradCAM_AvgDrop"] = float(avg_drop)
                results["GradCAM_IncreaseRate"] = float(inc_rate)
            except Exception:
                pass

        # --- Thêm Insertion/Deletion nếu có ---
        if len(gradcam_ins) > 0:
            results["GradCAM_Insertion"] = float(np.mean(gradcam_ins))
        if len(gradcam_del) > 0:
            results["GradCAM_Deletion"] = float(np.mean(gradcam_del))

        results.update(self.fcc_score(
            results["GradCAM_Fidelity"],
            results["GradCAM_Comprehensibility"],
            results["GradCAM_Consistency"]
        ))

        return results

