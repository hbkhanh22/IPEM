import numpy as np
import torch
from pathlib import Path
from typing import List, Union
from lime import lime_image
import shap
from pebex_explainer import PEBEXExplainer  # đã viết ở trên
from skimage.segmentation import mark_boundaries
from sklearn.metrics import accuracy_score
import time

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
    def local_accuracy(y_model, y_local):
        return accuracy_score(y_model, y_local)

    @staticmethod
    def stability(expl_vecs):
        diffs = []
        for i in range(len(expl_vecs)-1):
            diffs.append(np.linalg.norm(expl_vecs[i]-expl_vecs[i+1]))
        return -np.mean(diffs)  # càng nhỏ càng ổn định

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
    def compute_fidelity_shap(self, sample_imgs, shap_values, expected_value):
        """
        Fidelity cho SHAP (dựa trên accuracy của predicted labels).
        """
        self.model.eval()
        device = next(self.model.parameters()).device

        with torch.no_grad():
            y_model = self.model(sample_imgs.to(device))  # (N, num_classes)
            y_model = torch.softmax(y_model, dim=1).cpu().numpy()
        y_pred = np.argmax(y_model, axis=1)  # nhãn mô hình gốc

        # tái tạo dự đoán từ shap
        N = sample_imgs.shape[0]
        y_local = np.zeros_like(y_model)

        for c in range(len(shap_values)):  # lặp qua class
            contrib = shap_values[c].reshape(N, -1).sum(axis=1)  # (N,)
            y_local[:, c] = expected_value[c] + contrib

        # softmax để chuyển thành phân phối
        y_local = np.exp(y_local) / np.exp(y_local).sum(axis=1, keepdims=True)
        y_pred_e = np.argmax(y_local, axis=1)  # nhãn từ surrogate

        # Fidelity = accuracy giữa nhãn mô hình gốc và surrogate
        fidelity = accuracy_score(y_pred, y_pred_e)
        return fidelity
    
    # --------- Vector hóa heatmap để tính sim ----------
    def vectorize_explanation(self, heatmap, k=2000):
        flat = heatmap.flatten()
        idx = np.argsort(-np.abs(flat))[:k]
        vec = np.zeros_like(flat)
        vec[idx] = flat[idx]
        return vec
    
    def predict_proba_fn(self, imgs: np.ndarray):
        """
        Hàm cho LIME: nhận batch ảnh numpy (N,H,W,C) -> trả về xác suất (N,num_classes).
        """
        device = next(self.model.parameters()).device

        # chuyển numpy -> torch tensor
        if imgs.ndim == 3:   # (H,W,C)
            imgs = np.expand_dims(imgs, axis=0)  # -> (1,H,W,C)

        # scale về [0,1] nếu chưa
        if imgs.max() > 1.0:
            imgs = imgs / 255.0

        # (N,H,W,C) -> (N,C,H,W)
        imgs_t = torch.from_numpy(imgs).permute(0,3,1,2).float().to(device)

        with torch.no_grad():
            logits = self.model(imgs_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        return probs
    
    # --------- LIME ----------
    def evaluate_with_lime(self, samples):
        explainer = lime_image.LimeImageExplainer()
        results = {}

        y_model, y_local = [], []
        expl_vecs, used_feats = [], []
        time_start = time.time()
        for idx, (img_t, label) in enumerate(samples):
            # img_t shape: (C,H,W)
            if img_t.dim() == 2:  # grayscale
                img_t = img_t.unsqueeze(0)  # thêm channel

            img_np = img_t.permute(1,2,0).cpu().numpy()  # (H,W,C)

            # chạy lime
            explanation = explainer.explain_instance(
                img_np,
                classifier_fn=lambda ims: self.predict_proba_fn(ims),
                top_labels=len(self.class_names),
                hide_color=0,
                num_samples=1000,
            )

            top_label = np.argmax(self.predict_proba_fn(np.array([img_np]))[0])

            # lấy mask để vector hóa
            temp, mask = explanation.get_image_and_mask(
                label=top_label, positive_only=True, num_features=8, hide_rest=False
            )

            expl_vecs.append(self.vectorize_explanation(mask))
            used_feats.append(np.count_nonzero(mask))

            y_model.append(label)
            y_local.append(top_label)

        mu_comp, _ = self.comprehensibility(used_feats)
        mu_cons, _ = self.consistency(expl_vecs)
        time_end = time.time()
        exp_time = time_end - time_start

        results = {
            "LIME_Fidelity": self.fidelity(y_model, y_local),
            "LIME_Comprehensibility": mu_comp,
            "LIME_Consistency": mu_cons,
            "LIME_LocalAcc": self.local_accuracy(y_model, y_local),
            "LIME_Stability": self.stability(expl_vecs),
            "LIME_Similarity": mu_cons,
            "LIME_Robustness": self.robustness(expl_vecs),
            "LIME_Time": exp_time,
        }
        results.update(self.fcc_score(
            results["LIME_Fidelity"],
            results["LIME_Comprehensibility"],
            results["LIME_Consistency"]
        ))
        return results


    # --------- SHAP ----------
    def evaluate_with_shap(self, loader, batch_size: int = 8, noise_sigma: float = 0.02, large_mask_prob: float = 0.2):
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
        # # Random 50% chỉ số
        # indices = torch.randperm(N)[: N // 16]

        # # Chọn 50% ảnh
        # imgs = imgs[indices]
        # N = N // 16
        # 2) Chọn background cho SHAP (dùng 2 ảnh đầu trong batch nếu có)
        bg = imgs[:2].to(device)
        print(next(self.model.parameters()).device)   # xem model đang ở đâu
        print(imgs.device, bg.device)                 # xem input ở đâu

        # 3) Tạo GradientExplainer và tính shap_values cho batch gốc
        time_start = time.time()
        print(f"Starting SHAP ...")
        explainer = shap.GradientExplainer(self.model, batch_size=batch_size, data=bg)
       
        raw_shap = explainer.shap_values(imgs)  # có thể là list / tensor / numpy
        print(f"SHAP completed in {time.time() - time_start} seconds")
        # Helper: chuẩn hoá raw_shap -> shap_per_class dạng numpy (num_classes, N, C, H, W)
        def to_shap_per_class_array(sv):
            # sv có thể là: 
            # - list of (N,C,H,W) length = num_classes
            # - numpy array 5D (N,C,H,W,num_classes)
            # - numpy array 4D (N,C,H,W)
            import numpy as _np
            if isinstance(sv, list):
                # mỗi phần tử (N,C,H,W)
                arrs = []
                for part in sv:
                    if isinstance(part, torch.Tensor):
                        part = part.detach().cpu().numpy()
                    arrs.append(part)
                stacked = _np.stack(arrs, axis=0)  # (num_classes, N, C, H, W)
                return stacked
            else:
                # tensor/ndarray
                if isinstance(sv, torch.Tensor):
                    sv = sv.detach().cpu().numpy()
                sv = _np.asarray(sv)
                if sv.ndim == 5:
                    # (N, C, H, W, num_classes) -> transpose to (num_classes, N, C, H, W)
                    return _np.transpose(sv, (4, 0, 1, 2, 3))
                elif sv.ndim == 4:
                    # (N,C,H,W) -> single class
                    return sv[np.newaxis, ...]  # (1, N, C, H, W)
                else:
                    raise ValueError(f"Unsupported shap_values ndim: {sv.ndim}, shape={sv.shape}")

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
        expl_vecs = [self.vectorize_explanation(h) for h in heatmaps]
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
        vecs_noisy = [self.vectorize_explanation(h) for h in heatmaps_noisy]

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
        # mask_np = mask.cpu().numpy()
        # shap_largep_per_class = shap_per_class * (1.0 - mask_np)

        with torch.no_grad():
            y_logits_sub = self.model(imgs_largep[subset])
            y_probs_sub = torch.softmax(y_logits_sub, dim=1).cpu().numpy()
            y_model_sub = np.argmax(y_probs_sub, axis=1)   # (M,)

        M = len(subset)
        # chọn shap theo predicted class cho từng ảnh trong subset
        sel = shap_largep_per_class[y_model_sub, np.arange(M)]   # (M, C, H, W)
        heatmaps_largep = np.abs(sel).sum(axis=1)                # (M, H, W)

        vecs_largep = [self.vectorize_explanation(h) for h in heatmaps_largep]

        expl_vecs_sub = [expl_vecs[i] for i in subset.cpu().numpy()]
        robustness_val = float(np.mean([np.linalg.norm(a - b) for a, b in zip(expl_vecs_sub, vecs_largep)]))
        print("Done robustness_val")
        # 9) Similarity (mean pairwise cosine between normalized vectors)
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            M = np.stack([v / (np.linalg.norm(v) + 1e-8) for v in expl_vecs], axis=0)
            S = cosine_similarity(M)
            n = S.shape[0]
            if n > 1:
                sim_vals = S[~np.eye(n, dtype=bool)]
                similarity_val = float(sim_vals.mean())
            else:
                similarity_val = 1.0
        except Exception:
            similarity_val = float(np.nan)
        time_end = time.time()
        exp_time = time_end - time_start
        print("Done similarity_val")
        # 10) Compose results
        results = {
            "SHAP_Fidelity": float(self.fidelity(y_model, y_local)),        # accuracy
            "SHAP_Comprehensibility": float(mu_comp),
            "SHAP_Consistency": float(mu_cons),
            "SHAP_LocalAcc": float(self.local_accuracy(y_model, y_local)),
            "SHAP_Stability": float(stability_val),
            "SHAP_Similarity": float(similarity_val),
            "SHAP_Robustness": float(robustness_val),
            "SHAP_Time": exp_time,
        }

        # FCC (sửa: truyền float, không truyền list)
        fcc = self.fcc_score(
            results["SHAP_Fidelity"],
            results["SHAP_Comprehensibility"],
            results["SHAP_Consistency"]
        )
        results.update(fcc)

        return results

    # --------- PEBEX ----------
    def evaluate_with_pebex(self, loader, mode="black"):
        pebex = PEBEXExplainer(self.model, self.class_names)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        all_expl_vecs, all_used_feats = [], []
        all_y_model, all_y_local = [], []
        time_start = time.time()

        for batch_imgs, batch_labels in loader:
            imgs = batch_imgs.to(device)
            y_model = self.model(imgs).argmax(1).cpu().numpy()
            y_local = []

            for i in range(len(imgs)):
                # explain_one bây giờ đã trả về pred_label được tính từ importance scores
                heat, pred_from_importance = pebex.explain_one(imgs[i].to(device), mode)
                
                all_expl_vecs.append(self.vectorize_explanation(np.abs(heat)))
                all_used_feats.append(np.count_nonzero(np.abs(heat) > 1e-6))
                y_local.append(pred_from_importance)
            
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
            "PEBEX_LocalAcc": self.local_accuracy(all_y_model, all_y_local),
            "PEBEX_Stability": self.stability(all_expl_vecs),
            "PEBEX_Similarity": mu_cons,
            "PEBEX_Robustness": self.robustness(all_expl_vecs),
            "PEBEX_Time": exp_time,
        }
        results.update(self.fcc_score(results["PEBEX_Fidelity"], results["PEBEX_Comprehensibility"], results["PEBEX_Consistency"]))
        return results
