from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import numpy as np
import torch
import cv2
from typing import Tuple, List


class PEBEXExplainer:
    def __init__(self, model: torch.nn.Module, class_names: List[str], grid_size: Tuple[int,int]=(5,5), perturb_modes: List[str]=["black", "blur", "mean", "noise"], device=None):
        """
        model: mô hình deep learning (đã load weight, eval)
        class_names: tên các lớp (VD: ['COVID', 'NON-COVID'])
        grid_size: số ô chia (H, W)
        n_perturb: số lần tạo nhiễu ngẫu nhiên
        device: CPU/GPU
        """
        self.model = model.eval()
        self.class_names = class_names
        self.grid_size = grid_size
        self.perturb_modes = perturb_modes
        self.device = device if device else next(model.parameters()).device

    def perturb_patch(self, img_np, y1, y2, x1, x2, mode):
        """
        Sinh perturbation cho 1 patch
        """
        perturbed = img_np.copy()
        patch = perturbed[y1:y2, x1:x2]
        if mode == "black":
            perturbed[y1:y2, x1:x2, :] = 0
        elif mode == "blur":
            blur_patch = cv2.GaussianBlur(patch, (5, 5), 0)
            perturbed[y1:y2, x1:x2, :] = blur_patch
        elif mode == "mean":
            mean_val = patch.mean(axis=(0, 1), keepdims=True)
            perturbed[y1:y2, x1:x2, :] = mean_val
        elif mode == "noise":
            noise = np.random.normal(0, 0.2, size=patch.shape)
            patch += noise
            noisy_patch = np.clip(patch, 0, 1)
            perturbed[y1:y2, x1:x2, :] = noisy_patch

        return perturbed

    def explain_one(self, img_tensor: torch.Tensor, mode="black") -> Tuple[np.ndarray, int]:
        """
        Sinh heatmap giải thích cho 1 ảnh
        img_tensor: tensor (C,H,W) đã normalize
        return: (heatmap numpy HxW, predicted_label_from_importance)
        """
        self.model.eval()
        C, H, W = img_tensor.shape
        img_np = img_tensor.permute(1,2,0).cpu().numpy()  # HWC

        # dự đoán gốc (baseline)
        with torch.no_grad():
            pred = self.model(img_tensor.unsqueeze(0).to(self.device))
            probs = torch.softmax(pred, dim=1).cpu().numpy()[0]
            baseline_label = np.argmax(probs)
            baseline_logits = pred.cpu().numpy()[0]

        # chia lưới patch
        gh, gw = self.grid_size
        ph, pw = H // gh, W // gw

        importance_map = np.zeros((gh, gw))

        batch_inputs = []
        patch_indices = []

        for i in range(gh):
            for j in range(gw):
                y1, y2 = i*ph, (i+1)*ph
                x1, x2 = j*pw, (j+1)*pw
                #for mode in self.perturb_modes:
                perturbed = self.perturb_patch(img_np, y1, y2, x1, x2, mode)
                pt = torch.tensor(perturbed.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
                batch_inputs.append(pt)
                patch_indices.append((i, j))

        # gom batch
        batch_inputs = torch.cat(batch_inputs, dim=0).to(self.device)

        with torch.no_grad():
            preds = self.model(batch_inputs)
            probs_all = torch.softmax(preds, dim=1).cpu().numpy()[:, baseline_label]
        
        # gom theo patch
        # k = len(self.perturb_modes)
        for idx, (i,j) in enumerate(patch_indices):
            if importance_map[i, j] == 0:
                importance_map[i, j] = abs(probs_all[idx] - probs[baseline_label])

        # upscale importance_map về HxW
        heat = cv2.resize(importance_map, (W, H), interpolation=cv2.INTER_CUBIC)
        
        # Tính predicted_label từ importance scores
        pred_label_from_importance = self._predict_from_importance(
            baseline_logits, importance_map, gh, gw, ph, pw
        )
        
        return heat, pred_label_from_importance

    def _predict_from_importance(self, baseline_logits, importance_map, gh, gw, ph, pw):
        """
        Tính predicted_label từ importance scores.
        
        Args:
            baseline_logits: logits gốc từ model (num_classes,)
            importance_map: importance map cho từng patch (gh, gw)
            gh, gw: grid height, width
            ph, pw: patch height, width
            
        Returns:
            predicted_label: nhãn dự đoán từ importance scores
        """
        # Tính importance scores cho từng patch
        patch_importance = importance_map  # (gh, gw)
        
        # Lấy baseline label từ baseline_logits
        baseline_label = np.argmax(baseline_logits)
        
        # Tính dự đoán từ importance scores
        # PEBEX importance phản ánh sự thay đổi confidence khi perturb patch
        # Importance cao = patch quan trọng = giữ nguyên confidence
        # Importance thấp = patch không quan trọng = có thể giảm confidence
        
        # Tính weighted confidence dựa trên importance scores
        total_importance = np.sum(patch_importance)
        if total_importance > 0:
            # Normalize importance scores
            normalized_importance = patch_importance / total_importance
            
            # Tính weighted importance factor
            # Các patch có importance cao sẽ giữ nguyên confidence
            # Các patch có importance thấp sẽ giảm confidence
            importance_factor = np.sum(normalized_importance * patch_importance)
            
            # Điều chỉnh logits dựa trên importance factor
            adjusted_logits = baseline_logits.copy()
            
            # Điều chỉnh logit của class được dự đoán
            # importance_factor cao -> giữ nguyên hoặc tăng confidence
            # importance_factor thấp -> giảm confidence
            confidence_adjustment = (importance_factor - 0.5) * 4.0  # scale factor
            adjusted_logits[baseline_label] += confidence_adjustment
            
            # Cũng điều chỉnh các class khác ngược lại
            for c in range(len(adjusted_logits)):
                if c != baseline_label:
                    adjusted_logits[c] -= confidence_adjustment * 0.1  # nhẹ hơn
        else:
            # Nếu không có importance, giữ nguyên baseline
            adjusted_logits = baseline_logits.copy()
        
        # Softmax để có phân phối xác suất
        adjusted_probs = np.exp(adjusted_logits) / np.sum(np.exp(adjusted_logits))
        predicted_label = np.argmax(adjusted_probs)
        
        return predicted_label
    