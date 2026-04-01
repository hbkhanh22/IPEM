import numpy as np
import torch
from typing import List
from lime import lime_image
import shap
from ipem_explainer import IPEMExplainer  
from rise_explainer import RISE
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

    @staticmethod
    def _synchronize_device(device: torch.device):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps" and hasattr(torch, "mps"):
            try:
                torch.mps.synchronize()
            except Exception:
                pass

    # =====================================================================
    # --------- CÁC ĐỘ ĐO ĐÁNH GIÁ CHUNG (ĐÃ CHỈNH SỬA & BỔ SUNG) ---------
    # =====================================================================

    @staticmethod
    def gini_sparsity(heatmap):
        """
        [MỚI] Bổ sung độ đo Gini Sparsity.
        Tính độ thưa (Sparsity) theo chỉ số Gini. 
        Gini càng gần 1.0 nghĩa là Saliency Map càng thưa, tập trung vào số ít pixel quan trọng (dễ hiểu).
        """
        h = np.abs(np.array(heatmap).flatten())
        if np.sum(h) == 0:
            return 0.0
        h = np.sort(h)
        n = len(h)
        index = np.arange(1, n + 1)
        gini = (np.sum((2 * index - n  - 1) * h)) / (n * np.sum(h) + 1e-8)
        return float(gini)

    @staticmethod
    def insertion_deletion_score(model, img_tensor, heatmap, target_class, steps=20, mode="insertion"):
        """
        [MỚI] Tích hợp từ file Notebook.
        Tính Insertion/Deletion score (Faithfulness) cho một ảnh và heatmap.
        - Deletion: Loại bỏ pixel quan trọng dần -> Xác suất tụt -> Trả về Area Under Curve (Càng thấp càng tốt).
        - Insertion: Đưa vào pixel quan trọng dần -> Xác suất tăng -> Trả về Area Under Curve (Càng cao càng tốt).
        """
        device = next(model.parameters()).device
        img_tensor = img_tensor.to(device)
        baseline = torch.zeros_like(img_tensor)
        
        hm = np.array(heatmap)
        if hm.size == 0:
            raise ValueError("Heatmap is empty or invalid")
        
        if hm.ndim == 3:
            if hm.shape[0] <= 4 and hm.shape[0] != hm.shape[1]:
                hm = hm.mean(axis=0)
            else:
                hm = hm.mean(axis=2)
                
        hm = np.nan_to_num(hm, nan=0.0, posinf=0.0, neginf=0.0)
        if hm.max() - hm.min() > 0:
            hm = (hm - hm.min()) / (hm.max() - hm.min())
        else:
            hm = np.zeros_like(hm)

        if img_tensor.dim() == 4:
            _, C, H, W = img_tensor.shape
        elif img_tensor.dim() == 3:
            C, H, W = img_tensor.shape
        else:
            raise ValueError(f"Unsupported img_tensor shape: {img_tensor.shape}")

        target_w, target_h = int(W), int(H)
        heatmap_resized = cv2.resize((hm * 255).astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        
        heatmap_flat = heatmap_resized.flatten()
        indices = np.argsort(-heatmap_flat)  # Sắp xếp giảm dần

        probs = []
        for step in range(steps + 1):
            fraction = step / steps
            num_pixels = int(fraction * len(heatmap_flat))

            mask = np.zeros_like(heatmap_flat)
            mask[indices[:num_pixels]] = 1.0
            mask = mask.reshape(heatmap_resized.shape)

            mask_t = torch.tensor(mask, dtype=img_tensor.dtype, device=device)
            if img_tensor.dim() == 4:
                mask_t = mask_t.unsqueeze(0).unsqueeze(1)
            else:
                mask_t = mask_t.unsqueeze(0)

            if mode == "insertion":
                perturbed_img = baseline * (1 - mask_t) + img_tensor * mask_t
            else:  # deletion
                perturbed_img = img_tensor * (1 - mask_t) + baseline * mask_t

            perturbed_in = perturbed_img.unsqueeze(0) if perturbed_img.dim() == 3 else perturbed_img

            with torch.no_grad():
                output = model(perturbed_in)
                prob = torch.softmax(output, dim=1)[0, target_class].item()

            probs.append(prob)
            
        auc_score = np.trapz(probs, dx=1.0/steps)
        return float(auc_score)

    @staticmethod
    def average_drop_increase(original_probs, masked_probs):
        """Giữ nguyên từ bản gốc: Tính Average Drop và Increase Rate."""
        original_probs = np.array(original_probs)
        masked_probs = np.array(masked_probs)

        drops = np.maximum(0, original_probs - masked_probs)
        avg_drop = np.mean(drops / (original_probs + 1e-8)) * 100  

        increases = (masked_probs > original_probs).astype(np.float32)
        increase_rate = np.mean(increases) * 100  

        return avg_drop, increase_rate
    
    # --------- AOPC MoRF ----------
    def _compute_single_aopc(self, img_tensor, explanation, original_class, original_prob, block_size=8, percentile=None, verbose=False):
        """Giữ nguyên chuẩn logic AOPC của bạn."""
        device = next(self.model.parameters()).device
        
        if isinstance(img_tensor, torch.Tensor):
            if img_tensor.dim() == 3:  
                img_size = img_tensor.shape[-1]  
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            else: raise ValueError(f"Unsupported img_tensor shape: {img_tensor.shape}")
        else:
            img_np = img_tensor
            img_size = img_np.shape[0]  
        
        block_per_row = img_size // block_size
        
        if isinstance(explanation, torch.Tensor): explanation = explanation.detach().cpu().numpy()
        if explanation.shape != (img_size, img_size): explanation = cv2.resize(explanation, (img_size, img_size))
        
        blocks_dict = {}
        key = 0
        for i in range(0, img_size, block_size):
            for j in range(0, img_size, block_size):
                block = explanation[i:i + block_size, j:j + block_size]
                blocks_dict[key] = (block, np.sum(block))
                key += 1
        
        sorted_blocks = sorted(blocks_dict.items(), key=lambda x: x[1][1], reverse=True)
        total_sum_blocks = sum(block_sum[1] for block_sum in list(blocks_dict.values()))
        
        breaking_pct_value = None
        if percentile is not None: breaking_pct_value = (percentile / 100) * total_sum_blocks
        
        img_pred_aopc = img_np.copy()
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        if isinstance(original_class, torch.Tensor): original_class = original_class.item()
        if isinstance(original_prob, torch.Tensor): original_prob = original_prob.cpu().numpy()
        
        AOPC = []
        count_blocks = 0
        sum_blocks = 0
        
        for i in range(len(blocks_dict)):
            ref_block_index = sorted_blocks[i][0]
            row = ref_block_index // block_per_row
            col = ref_block_index % block_per_row
            
            img_pred_aopc[row * block_size:row * block_size + block_size, col * block_size:col * block_size + block_size] = _make_perturbation(
                img_pred_aopc[row * block_size:row * block_size + block_size, col * block_size:col * block_size + block_size]
            )
            
            img_pil = Image.fromarray((img_pred_aopc * 255).astype(np.uint8))
            input_tensor = transform(img_pil).unsqueeze(0).to(device)
            
            with torch.no_grad(): output = self.model(input_tensor)
            
            probs = torch.softmax(output, dim=1)
            class_prob_after = probs[0, original_class].item()
            
            delta = original_prob[original_class] - class_prob_after
            AOPC.append(delta)
            count_blocks += 1
            
            if percentile is not None:
                sum_blocks += np.sum(sorted_blocks[i][1][0])
                if sum_blocks >= breaking_pct_value: break
        
        return (original_class, AOPC, np.sum(AOPC) / count_blocks)


    # =====================================================================
    # --------- ÁP DỤNG CÁC ĐỘ ĐO CHO 4 PHƯƠNG PHÁP XAI CHUẨN MỰC ---------
    # =====================================================================

    # --------- LIME ----------
    def evaluate_with_lime(self, samples, num_samples=1000, compute_aopc=True, block_size=8, percentile=None, noise_sigma=0.02):
        device = next(self.model.parameters()).device
        explainer = lime_image.LimeImageExplainer()

        all_aopc_scores = []
        orig_probs_list, masked_probs_list = [], []
        all_gini = []
        all_insertion, all_deletion = [], []
        explanation_times = []
        
        for idx, (img_t, label) in enumerate(samples):
            if img_t.dim() == 2: img_t = img_t.unsqueeze(0)
            img_np = img_t.permute(1,2,0).cpu().numpy()

            self._synchronize_device(device)
            explanation_start = time.perf_counter()
            explanation = explainer.explain_instance(
                img_np,
                classifier_fn=lambda ims: predict_proba_fn(self.model, ims),
                top_labels=len(self.class_names),
                hide_color=0,
                num_samples=num_samples,
            )
            top_label = np.argmax(predict_proba_fn(self.model, np.array([img_np]))[0])
            temp, mask = explanation.get_image_and_mask(label=top_label, positive_only=True, num_features=8, hide_rest=False)
            self._synchronize_device(device)
            explanation_times.append(time.perf_counter() - explanation_start)

            # Sparsity / LIME_Gini
            all_gini.append(self.gini_sparsity(mask))
            
            # Insertion / Deletion AUC
            all_insertion.append(self.insertion_deletion_score(self.model, img_t, mask, top_label, steps=20, mode="insertion"))
            all_deletion.append(self.insertion_deletion_score(self.model, img_t, mask, top_label, steps=20, mode="deletion"))

            if compute_aopc:
                with torch.no_grad():
                    img_tensor = img_t.unsqueeze(0).to(device)
                    original_output = self.model(img_tensor)
                    original_prob = torch.softmax(original_output, dim=1)[0]
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
                except Exception: pass
                
                _, AOPC_list, mean_AOPC = self._compute_single_aopc(
                    img_tensor=img_tensor.squeeze(0), explanation=mask, original_class=top_label,
                    original_prob=original_prob, block_size=block_size, percentile=percentile, verbose=False
                )
                all_aopc_scores.append(mean_AOPC)
        
        # [MỚI] Local Stability - Chọn ngẫu nhiên 1 ảnh (tránh mất nhiều 1000 iter của LIME)
        stability_val = 0.0
        if len(samples) > 0:
            img_t, _ = samples[0] # lấy đại 1 ảnh
            if img_t.dim() == 2: img_t = img_t.unsqueeze(0)
            # Tạo nhiễu nhỏ trong tập
            img_noisy = (img_t + torch.randn_like(img_t) * noise_sigma).clamp(-5.0, 5.0)
            img_np_noisy = img_noisy.permute(1,2,0).cpu().numpy()
            
            expl_noisy = explainer.explain_instance(
                img_np_noisy, classifier_fn=lambda ims: predict_proba_fn(self.model, ims),
                top_labels=len(self.class_names), hide_color=0, num_samples=num_samples
            )
            top_label_noisy = np.argmax(predict_proba_fn(self.model, np.array([img_np_noisy]))[0])
            _, mask_noisy = expl_noisy.get_image_and_mask(label=top_label_noisy, positive_only=True, num_features=8, hide_rest=False)
            
            # Rerunning base sample again strictly for vector alignment (or use the one computed above)
            img_np = img_t.permute(1,2,0).cpu().numpy()
            top_label_orig = np.argmax(predict_proba_fn(self.model, np.array([img_np]))[0])
            _, mask_orig = explainer.explain_instance(img_np, classifier_fn=lambda ims: predict_proba_fn(self.model, ims),
                                                      top_labels=len(self.class_names), hide_color=0, num_samples=num_samples).get_image_and_mask(label=top_label_orig, positive_only=True, num_features=8)
            
            vec_orig = vectorize_explanation(mask_orig)
            vec_noisy = vectorize_explanation(mask_noisy)
            stability_val = float(np.linalg.norm(vec_orig - vec_noisy))

        results = {
            "LIME_Time": float(np.mean(explanation_times)) if explanation_times else 0.0,
            "LIME_Gini_Sparsity": float(np.mean(all_gini)) if len(all_gini)>0 else 0.0,
            "LIME_Insertion_AUC": float(np.mean(all_insertion)) if len(all_insertion)>0 else 0.0,
            "LIME_Deletion_AUC": float(np.mean(all_deletion)) if len(all_deletion)>0 else 0.0,
            "LIME_Local_Stability_L2": stability_val
        }
        if compute_aopc and len(all_aopc_scores) > 0:
            results["LIME_AOPC_Mean"] = float(np.mean(all_aopc_scores))
        if len(orig_probs_list) > 0:
            try:
                avg_drop, inc_rate = self.average_drop_increase(orig_probs_list, masked_probs_list)
                results["LIME_AvgDrop"] = float(avg_drop)
                results["LIME_IncreaseRate"] = float(inc_rate)
            except Exception: pass
        return results

    # --------- SHAP ----------
    def evaluate_with_shap(self, loader, batch_size: int = 8, noise_sigma: float = 0.02, large_mask_prob: float = 0.2, 
                          compute_aopc=True, block_size=8, percentile=None):
        device = next(self.model.parameters()).device
        imgs, _ = next(iter(loader))
        imgs = imgs.to(device)
        N = imgs.shape[0]

        bg = imgs[:2].to(device)
        explainer = shap.GradientExplainer(self.model, batch_size=batch_size, data=bg)
        self._synchronize_device(device)
        explanation_start = time.perf_counter()
        raw_shap = explainer.shap_values(imgs)
        self._synchronize_device(device)
        avg_explanation_time = (time.perf_counter() - explanation_start) / max(N, 1)
        
        def to_shap_per_class_array(svalues):
            if isinstance(svalues, list): return np.stack([sv.detach().cpu().numpy() if torch.is_tensor(sv) else sv for sv in svalues], axis=0)
            elif isinstance(svalues, (np.ndarray, torch.Tensor)):
                if torch.is_tensor(svalues): svalues = svalues.detach().cpu().numpy()
                if svalues.ndim == 5:
                    n_classes = len(self.class_names)
                    if svalues.shape[0] == n_classes: return svalues
                    elif svalues.shape[-1] == n_classes: return np.transpose(svalues, (4, 0, 1, 2, 3))
                    elif svalues.shape[-1] > 1 and svalues.shape[0] != svalues.shape[-1]: return np.transpose(svalues, (4, 0, 1, 2, 3))
                    else: raise ValueError(f"Ambiguous SHAP shape")
                elif svalues.ndim == 4: return svalues[None, ...]
                else: raise ValueError("Unsupported SHAP ndim")
            else: raise TypeError("Unsupported SHAP type")
        
        shap_per_class = to_shap_per_class_array(raw_shap)
        num_classes = shap_per_class.shape[0]

        with torch.no_grad():
            y_model_logits = self.model(imgs)
            y_model_probs = torch.softmax(y_model_logits, dim=1).cpu().numpy()
            y_model = np.argmax(y_model_probs, axis=1)
            expected_value = self.model(bg).mean(0).cpu().numpy() if bg.shape[0] > 0 else np.zeros((num_classes,), dtype=float)

        y_local_logits = np.zeros_like(y_model_probs)
        for c in range(num_classes):
            arr = shap_per_class[c]
            contrib = arr.reshape(N, -1).sum(axis=1)
            y_local_logits[:, c] = expected_value[c] + contrib

        y_local_probs = np.exp(y_local_logits) / np.exp(y_local_logits).sum(axis=1, keepdims=True)
        y_local = np.argmax(y_local_probs, axis=1)

        heatmaps = []
        for i in range(N):
            heatmaps.append(np.abs(shap_per_class[int(y_model[i])][i]).sum(axis=0))
            
        expl_vecs = [vectorize_explanation(h) for h in heatmaps]
        
        # Stability: recompute SHAP on slightly noised images
        imgs_noisy = (imgs + torch.randn_like(imgs) * noise_sigma).clamp(-5.0, 5.0)
        subset = torch.randperm(N)[: max(1, N // 10)]
        raw_shap_noisy = explainer.shap_values(imgs_noisy[subset])
        shap_noisy_per_class = to_shap_per_class_array(raw_shap_noisy)

        with torch.no_grad(): y_model_sub = np.argmax(torch.softmax(self.model(imgs_noisy[subset]), dim=1).cpu().numpy(), axis=1)
        heatmaps_noisy = np.abs(shap_noisy_per_class[y_model_sub, np.arange(len(subset))]).sum(axis=1)
        vecs_noisy = [vectorize_explanation(h) for h in heatmaps_noisy]
        expl_vecs_sub = [expl_vecs[i] for i in subset.cpu().numpy()]
        stability_val = float(np.mean([np.linalg.norm(a - b) for a, b in zip(expl_vecs_sub, vecs_noisy)]))
        
        # Robustness
        mask = (torch.rand_like(imgs[:, :1, :, :]) > (1.0 - large_mask_prob)).float().to(device)
        imgs_largep = imgs * (1.0 - mask)
        raw_shap_largep = explainer.shap_values(imgs_largep[subset])
        shap_largep_per_class = to_shap_per_class_array(raw_shap_largep)
        with torch.no_grad(): y_model_sub_large = np.argmax(torch.softmax(self.model(imgs_largep[subset]), dim=1).cpu().numpy(), axis=1)
        heatmaps_largep = np.abs(shap_largep_per_class[y_model_sub_large, np.arange(len(subset))]).sum(axis=1)
        vecs_largep = [vectorize_explanation(h) for h in heatmaps_largep]
        robustness_val = float(np.mean([np.linalg.norm(a - b) for a, b in zip(expl_vecs_sub, vecs_largep)]))

        # Khai báo biến mảng 
        all_aopc_scores = []
        shap_orig_probs, shap_masked_probs = [], []
        all_gini, all_insertion, all_deletion = [], [], []

        for i in range(N):
            heat = heatmaps[i]
            img_tensor = imgs[i]
            orig_class = int(y_model[i])
            
            # GINI Sparsity
            all_gini.append(self.gini_sparsity(heat))
            
            # Insertion / Deletion AUC
            all_insertion.append(self.insertion_deletion_score(self.model, img_tensor, heat, orig_class, steps=20, mode="insertion"))
            all_deletion.append(self.insertion_deletion_score(self.model, img_tensor, heat, orig_class, steps=20, mode="deletion"))

            if compute_aopc:    
                try:
                    bin_mask = (heat > np.mean(heat)).astype(np.float32)
                    img_t = img_tensor.unsqueeze(0)
                    mask_t = torch.from_numpy(bin_mask).to(device).unsqueeze(0).unsqueeze(0)
                    mask_t = mask_t.expand(1, img_t.shape[1], bin_mask.shape[0], bin_mask.shape[1])
                    with torch.no_grad():
                        masked_prob = torch.softmax(self.model(img_t * mask_t), dim=1)[0, orig_class].item()
                    shap_orig_probs.append(float(y_model_probs[i][orig_class]))
                    shap_masked_probs.append(float(masked_prob))
                except Exception: pass

                _, _, mean_AOPC = self._compute_single_aopc(img_tensor, heat, orig_class, torch.tensor(y_model_probs[i], device=device), block_size=block_size, percentile=percentile)
                all_aopc_scores.append(mean_AOPC)

        results = {
            "SHAP_Time": float(avg_explanation_time),
            "SHAP_Fidelity_Accuracy": float(accuracy_score(y_model, y_local)),
            "SHAP_Gini_Sparsity": float(np.mean(all_gini)) if all_gini else 0.0,
            "SHAP_Insertion_AUC": float(np.mean(all_insertion)) if all_insertion else 0.0,
            "SHAP_Deletion_AUC": float(np.mean(all_deletion)) if all_deletion else 0.0,
            "SHAP_Local_Stability_L2": float(stability_val),
            "SHAP_Robustness_LargeMask_L2": float(robustness_val),
        }
        if compute_aopc and len(all_aopc_scores) > 0: results["SHAP_AOPC_Mean"] = float(np.mean(all_aopc_scores))
        if len(shap_orig_probs) > 0:
            try:
                avg_drop, inc_rate = self.average_drop_increase(shap_orig_probs, shap_masked_probs)
                results["SHAP_AvgDrop"] = float(avg_drop)
                results["SHAP_IncreaseRate"] = float(inc_rate)
            except Exception: pass
        return results

    # --------- IPEM ----------
    def evaluate_with_ipem(self, loader, compute_aopc=True, block_size=8, percentile=None, noise_sigma=0.02):
        ipem = IPEMExplainer(self.model, self.class_names)
        device = next(self.model.parameters()).device
        
        all_gini, all_insertion, all_deletion = [], [], []
        all_stability = []
        all_aopc_scores = []  
        IPEM_orig_probs, IPEM_masked_probs = [], []
        explanation_times = []
        
        for batch_imgs, batch_labels in loader:
            imgs = batch_imgs.to(device)
            y_model = self.model(imgs).argmax(1).cpu().numpy()

            for i in range(len(imgs)):
                img_t = imgs[i]
                self._synchronize_device(device)
                explanation_start = time.perf_counter()
                heat, pred_from_importance = ipem.explain(img_t.to(device))
                self._synchronize_device(device)
                explanation_times.append(time.perf_counter() - explanation_start)
                heat_abs = np.abs(heat)
                
                # Gini Sparsity
                all_gini.append(self.gini_sparsity(heat_abs))
                
                # Insertion & Deletion
                all_insertion.append(self.insertion_deletion_score(self.model, img_t, heat_abs, y_model[i], steps=20, mode="insertion"))
                all_deletion.append(self.insertion_deletion_score(self.model, img_t, heat_abs, y_model[i], steps=20, mode="deletion"))

                # Local Stability
                img_noisy = (img_t + torch.randn_like(img_t) * noise_sigma).clamp(-5.0, 5.0)
                heat_noisy, _ = ipem.explain(img_noisy)
                stab = np.linalg.norm(vectorize_explanation(heat_abs) - vectorize_explanation(np.abs(heat_noisy)))
                all_stability.append(stab)
                
                # AOPC / AvgDrop
                if compute_aopc:
                    with torch.no_grad():
                        img_tensor = img_t.unsqueeze(0).to(device)
                        original_output = self.model(img_tensor)
                        original_prob = torch.softmax(original_output, dim=1)[0]
                        original_class = np.argmax(original_prob.cpu().numpy())
                    try:
                        bin_mask = (heat_abs > np.mean(heat_abs)).astype(np.float32)
                        mask_t = torch.from_numpy(bin_mask).to(device).unsqueeze(0).unsqueeze(0)
                        mask_t = mask_t.expand(1, img_tensor.shape[1], bin_mask.shape[0], bin_mask.shape[1])
                        with torch.no_grad():
                            masked_out = self.model(img_tensor * mask_t)
                            masked_prob = torch.softmax(masked_out, dim=1)[0, original_class].item()
                        IPEM_orig_probs.append(float(original_prob[original_class].item()))
                        IPEM_masked_probs.append(float(masked_prob))
                    except Exception: pass
                    _, _, mean_AOPC = self._compute_single_aopc(img_tensor.squeeze(0), heat_abs, original_class, original_prob, block_size, percentile, False)
                    all_aopc_scores.append(mean_AOPC)
        
        results = {
            "IPEM_Time": float(np.mean(explanation_times)) if explanation_times else 0.0,
            "IPEM_Gini_Sparsity": float(np.mean(all_gini)) if all_gini else 0.0,
            "IPEM_Insertion_AUC": float(np.mean(all_insertion)) if all_insertion else 0.0,
            "IPEM_Deletion_AUC": float(np.mean(all_deletion)) if all_deletion else 0.0,
            "IPEM_Local_Stability_L2": float(np.mean(all_stability)) if all_stability else 0.0,
        }
        if compute_aopc and len(all_aopc_scores) > 0: results["IPEM_AOPC_Mean"] = float(np.mean(all_aopc_scores))
        if len(IPEM_orig_probs) > 0:
            try:
                avg_drop, inc_rate = self.average_drop_increase(IPEM_orig_probs, IPEM_masked_probs)
                results["IPEM_AvgDrop"] = float(avg_drop)
                results["IPEM_IncreaseRate"] = float(inc_rate)
            except Exception: pass
        return results

    # --------- GradCAM ----------
    def evaluate_with_GradCAM(self, loader, compute_aopc=True, block_size=8, percentile=None, noise_sigma=0.02):
        def get_last_conv_layer(model):
            last_conv = None
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d): last_conv = module
            if last_conv is None: raise ValueError("❌ Không tìm thấy Conv2D nào trong model.")
            return last_conv

        device = next(self.model.parameters()).device
        target_layer = get_last_conv_layer(self.model)
        cam = GradCAM(model=self.model, target_layers=[target_layer])

        all_gini, all_insertion, all_deletion = [], [], []
        all_stability = []
        all_aopc_scores = []
        gradcam_orig_probs, gradcam_masked_probs = [], []
        explanation_times = []

        for imgs, labels in loader:
            imgs = imgs.to(device)

            with torch.no_grad():
                logits = self.model(imgs)
                y_model = logits.argmax(1).cpu().numpy()

            for i in range(len(imgs)):
                img = imgs[i].unsqueeze(0)
                target = [ClassifierOutputTarget(y_model[i])]
                self._synchronize_device(device)
                explanation_start = time.perf_counter()
                heatmap = cam(input_tensor=img, targets=target)[0]
                self._synchronize_device(device)
                explanation_times.append(time.perf_counter() - explanation_start)

                # Gini Sparsity
                all_gini.append(self.gini_sparsity(heatmap))
                
                # Insertion & Deletion
                all_insertion.append(self.insertion_deletion_score(self.model, imgs[i], heatmap, y_model[i], steps=20, mode="insertion"))
                all_deletion.append(self.insertion_deletion_score(self.model, imgs[i], heatmap, y_model[i], steps=20, mode="deletion"))

                # Local Stability
                img_noisy = (imgs[i] + torch.randn_like(imgs[i]) * noise_sigma).clamp(-5.0, 5.0).unsqueeze(0)
                heatmap_noisy = cam(input_tensor=img_noisy, targets=target)[0]
                stab = np.linalg.norm(vectorize_explanation(heatmap) - vectorize_explanation(heatmap_noisy))
                all_stability.append(stab)

                # AOPC
                if compute_aopc:
                    with torch.no_grad():
                        original_output = self.model(img)
                        original_prob = torch.softmax(original_output, dim=1)[0]
                    try:
                        bin_mask = (heatmap > np.mean(heatmap)).astype(np.float32)
                        mask_t = torch.from_numpy(bin_mask).to(device).unsqueeze(0).unsqueeze(0)
                        mask_t = mask_t.expand(1, img.shape[1], bin_mask.shape[0], bin_mask.shape[1])
                        with torch.no_grad():
                            masked_out = self.model(img * mask_t)
                            masked_prob = torch.softmax(masked_out, dim=1)[0, int(y_model[i])].item()
                        gradcam_orig_probs.append(float(original_prob[int(y_model[i])].item()))
                        gradcam_masked_probs.append(float(masked_prob))
                    except Exception: pass
                    _, _, mean_AOPC = self._compute_single_aopc(imgs[i], heatmap, int(y_model[i]), original_prob, block_size, percentile, False)
                    all_aopc_scores.append(mean_AOPC)
        
        results = {
            "GradCAM_Time": float(np.mean(explanation_times)) if explanation_times else 0.0,
            "GradCAM_Gini_Sparsity": float(np.mean(all_gini)) if all_gini else 0.0,
            "GradCAM_Insertion_AUC": float(np.mean(all_insertion)) if all_insertion else 0.0,
            "GradCAM_Deletion_AUC": float(np.mean(all_deletion)) if all_deletion else 0.0,
            "GradCAM_Local_Stability_L2": float(np.mean(all_stability)) if all_stability else 0.0,
        }
        if compute_aopc and len(all_aopc_scores) > 0: results["GradCAM_AOPC_Mean"] = float(np.mean(all_aopc_scores))
        if len(gradcam_orig_probs) > 0:
            try:
                avg_drop, inc_rate = self.average_drop_increase(gradcam_orig_probs, gradcam_masked_probs)
                results["GradCAM_AvgDrop"] = float(avg_drop)
                results["GradCAM_IncreaseRate"] = float(inc_rate)
            except Exception: pass
        return results

    def evaluate_with_rise(self, loader, compute_aopc=True, block_size=8, percentile=None, noise_sigma=0.02):
        rise_explainer = RISE(
            mode=self.model,
            n_masks=1000,
            p=0.5,
            initial_mask_size=(11, 11),
        )
        device = next(self.model.parameters()).device
        all_gini, all_insertion, all_deletion = [], [], []
        all_stability = []
        all_aopc_scores = []
        rise_orig_probs, rise_masked_probs = [], []
        explanation_times = []

        for batch_imgs, batch_labels in loader:
            imgs = batch_imgs.to(device)
            y_model = self.model(imgs).argmax(1).cpu().numpy()

            for i in range(len(imgs)):
                img_tensor = imgs[i].unsqueeze(0)
                self._synchronize_device(device)
                explanation_start = time.perf_counter()
                heatmap = rise_explainer.explain(img_tensor)
                self._synchronize_device(device)
                explanation_times.append(time.perf_counter() - explanation_start)
                heatmap_abs = np.abs(heatmap)

                # Gini Sparsity
                all_gini.append(self.gini_sparsity(heatmap_abs))
                
                # Insertion & Deletion
                all_insertion.append(self.insertion_deletion_score(self.model, img_tensor, heatmap_abs, y_model[i], steps=20, mode="insertion"))
                all_deletion.append(self.insertion_deletion_score(self.model, img_tensor, heatmap_abs, y_model[i], steps=20, mode="deletion"))

                # Local Stability
                img_noisy = (img_tensor + torch.randn_like(img_tensor) * noise_sigma).clamp(-5.0, 5.0)
                heatmap_noisy = rise_explainer.explain(img_noisy)
                stab = np.linalg.norm(vectorize_explanation(heatmap_abs) - vectorize_explanation(np.abs(heatmap_noisy)))
                all_stability.append(stab)

                # AOPC
                if compute_aopc:
                    with torch.no_grad():
                        original_output = self.model(img_tensor)
                        original_prob = torch.softmax(original_output, dim=1)[0]
                        original_class = np.argmax(original_prob.cpu().numpy())
                    try:
                        bin_mask = (heatmap_abs > np.mean(heatmap_abs)).astype(np.float32)
                        mask_t = torch.from_numpy(bin_mask).to(device).unsqueeze(0).unsqueeze(0)
                        mask_t = mask_t.expand(1, img_tensor.shape[1], bin_mask.shape[0], bin_mask.shape[1])

                    except Exception: pass
                    _, _, mean_AOPC = self._compute_single_aopc(img_tensor.squeeze(0), heatmap_abs, original_class, original_prob, block_size, percentile, False)
                    all_aopc_scores.append(mean_AOPC)

        results = {
            "RISE_Time": float(np.mean(explanation_times)) if explanation_times else 0.0,
            "RISE_Gini_Sparsity": float(np.mean(all_gini)) if all_gini else 0.0,
            "RISE_Insertion_AUC": float(np.mean(all_insertion)) if all_insertion else 0.0,
            "RISE_Deletion_AUC": float(np.mean(all_deletion)) if all_deletion else 0.0,
            "RISE_Local_Stability_L2": float(np.mean(all_stability)) if all_stability else 0.0,
        }
        if compute_aopc and len(all_aopc_scores) > 0: results["RISE_AOPC_Mean"] = float(np.mean(all_aopc_scores))
        if len(rise_orig_probs) > 0:
            try:
                avg_drop, inc_rate = self.average_drop_increase(rise_orig_probs, rise_masked_probs)
                results["RISE_AvgDrop"] = float(avg_drop)
                results["RISE_IncreaseRate"] = float(inc_rate)
            except Exception: pass
        return results
    
