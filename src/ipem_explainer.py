from typing import List, Tuple
import numpy as np
import torch
import cv2
from skimage.segmentation import slic
from skimage.util import img_as_float

class IPEMExplainer:
    def __init__(self, model: torch.nn.Module, class_names: List[str], grid_size: Tuple[int,int]=(4,4), perturb_modes: List[str]=["black", "blur", "mean", "noise"], device=None):
        """ 
        model: mô hình deep learning (đã load weight, eval)
        class_names: tên các lớp 
        grid_size: số ô chia (H, W)
        n_perturb: số lần tạo nhiễu ngẫu nhiên
        device: CPU/GPU
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.eval()
        self.class_names = class_names
        self.grid_size = grid_size
        self.perturb_modes = perturb_modes
        self.device = device

    def _predict_from_importance(self, baseline_logits, importance_map):
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
        # IPEM importance phản ánh sự thay đổi confidence khi perturb patch
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
    
    def explain(
        self,
        img_tensor: torch.Tensor,
        n_samples: int = 400,
        mask_prob: float = 0.5,
        sigma_smooth: float = 0.5,
        grid_sizes: list = [(8, 8)]
    ):
        self.model.eval()
        C, H, W = img_tensor.shape
        img = img_tensor.to(self.device)

        with torch.no_grad():
            baseline_logits = self.model(img.unsqueeze(0))
            baseline_label = baseline_logits.argmax(dim=1).item()

        heatmaps = []

        for gh, gw in grid_sizes:
            # ----------------------------------
            # 1. Monte Carlo masks (vectorized)
            # ----------------------------------
            # masks = torch.rand(n_samples, gh, gw, device=self.device) < mask_prob
            masks = torch.rand(n_samples, gh, gw, device=self.device) < 0.5
            masks = masks.float()

            # Upsample mask to image size
            masks_up = torch.nn.functional.interpolate(
                masks.unsqueeze(1),
                size=(H, W),
                mode="nearest"
            )  # (n_mc,1,H,W)

            # ----------------------------------
            # 2. Perturb images (vectorized)
            # ----------------------------------
            perturbed_imgs = img.unsqueeze(0) * masks_up

            # ----------------------------------
            # 3. Model inference (batch)
            # ----------------------------------
            with torch.no_grad():
                preds = self.model(perturbed_imgs)
                probs = torch.softmax(preds, dim=1)[:, baseline_label]
                # probs = preds[:, baseline_label]

            # ----------------------------------
            # 4. Vectorized Monte Carlo stats
            # ----------------------------------
            probs = probs.view(-1, 1, 1)

            on_mask = masks
            off_mask = 1 - masks

            cnt_on = on_mask.sum(dim=0) + 1e-8
            cnt_off = off_mask.sum(dim=0) + 1e-8

            mean_on = (probs * on_mask).sum(dim=0) / cnt_on
            mean_off = (probs * off_mask).sum(dim=0) / cnt_off

            var_on = ((probs - mean_on)**2 * on_mask).sum(dim=0) / cnt_on
            var_off = ((probs - mean_off)**2 * off_mask).sum(dim=0) / cnt_off

            importance = (mean_on - mean_off) / (torch.sqrt(var_on + var_off) + 1e-8)

            # normalize
            importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)

            if sigma_smooth > 0:
                importance = cv2.GaussianBlur(
                    importance.cpu().numpy(),
                    (0, 0),
                    sigmaX=sigma_smooth
                )

            heat = cv2.resize(importance, (W, H), interpolation=cv2.INTER_CUBIC)
            heatmaps.append(heat)

        final_heat = np.mean(heatmaps, axis=0)
        final_heat = (final_heat - final_heat.min()) / (final_heat.max() - final_heat.min() + 1e-8)

        return final_heat, baseline_label

    def explain_by_slic(
        self, 
        img_tensor: torch.Tensor,
        n_samples: int = 400,
        mask_prob: float = 0.5,
        sigma_smooth: float = 6.0,
        n_segments_list: list=[80, 120],
        compactness: float = 10.0,
        slic_sigma: float = 1.0,
        batch_size: int = 64
    ):
        self.model.eval()
        C, H, W = img_tensor.shape
        img = img_tensor.to(self.device)

        # ----------------------------------
        # 1. Baseline prediction
        # ----------------------------------
        with torch.no_grad():
            baseline_logits = self.model(img.unsqueeze(0))
            baseline_label = baseline_logits.argmax(dim=1).item()
        
        # ----------------------------------
        # 2. Preprocess image for SLIC
        # ----------------------------------
        img_np = img_tensor.detach().cpu().float().numpy().transpose(1,2,0)

        img_min = img_np.min()
        img_max = img_np.max()

        img_segment = (img_np - img_min) / (img_max - img_min + 1e-8)

        heatmaps = []

        # ----------------------------------
        # 3. Iterate over different numbers of segments
        # ----------------------------------
        for n_segments in n_segments_list:
            segments = slic(
                img_segment,
                n_segments=n_segments,
                compactness=compactness,
                sigma=slic_sigma,
                start_label=0,
                channel_axis=-1
            )

            segments = segments.astype(np.int64)
            K = int(segments.max()) + 1

            segments_t = torch.from_numpy(segments).to(self.device)

            masks_sp = (torch.rand(n_samples, K, device=self.device) < mask_prob).float()

            # Ensure not-all-zero masks
            empty_rows = masks_sp.sum(dim=1) == 0
            if empty_rows.any():
                rand_idx = torch.randint(0, K, (empty_rows.sum().item(),), device=self.device)
                masks_sp[empty_rows, rand_idx] = 1.0

            # --------------------------------------------------
            # 4. Expand superpixel masks -> pixel masks
            #    pixel_masks: (n_samples, H, W)
            # --------------------------------------------------
            pixel_masks = masks_sp[:, segments_t]  # advanced indexing -> (n_samples, H, W)
            pixel_masks = pixel_masks.unsqueeze(1)  # (n_samples, 1, H, W)

            # Broadcast to channels
            perturbed_imgs = img.unsqueeze(0) * pixel_masks  # (n_samples, C, H, W)

            # --------------------------------------------------
            # 5. Model inference in mini-batches
            # --------------------------------------------------
            probs_all = []
            with torch.no_grad():
                for start in range(0, n_samples, batch_size):
                    end = min(start + batch_size, n_samples)
                    preds = self.model(perturbed_imgs[start:end])
                    probs = torch.softmax(preds, dim=1)[:, baseline_label]
                    probs_all.append(probs)

            probs = torch.cat(probs_all, dim=0)  # (n_samples,)
            probs = probs.view(-1, 1)  # (n_samples, 1)

            # --------------------------------------------------
            # 6. Vectorized Monte Carlo stats on superpixels
            # --------------------------------------------------
            on_mask = masks_sp                  # (n_samples, K)
            off_mask = 1.0 - masks_sp          # (n_samples, K)

            cnt_on = on_mask.sum(dim=0) + 1e-8     # (K,)
            cnt_off = off_mask.sum(dim=0) + 1e-8   # (K,)

            mean_on = (probs * on_mask).sum(dim=0) / cnt_on     # (K,)
            mean_off = (probs * off_mask).sum(dim=0) / cnt_off  # (K,)

            var_on = (((probs - mean_on.unsqueeze(0)) ** 2) * on_mask).sum(dim=0) / cnt_on
            var_off = (((probs - mean_off.unsqueeze(0)) ** 2) * off_mask).sum(dim=0) / cnt_off

            importance_sp = (mean_on - mean_off) / (torch.sqrt(var_on + var_off) + 1e-8)  # (K,)

            # Normalize superpixel importance
            importance_sp = (importance_sp - importance_sp.min()) / (
                importance_sp.max() - importance_sp.min() + 1e-8
            )

            # --------------------------------------------------
            # 7. Project superpixel importance back to pixel space
            # --------------------------------------------------
            heat = importance_sp[segments_t].detach().cpu().numpy()  # (H, W)

            if sigma_smooth > 0:
                heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=sigma_smooth)
            
            heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
            heatmaps.append(heat)

        # --------------------------------------------------
        # 8. Multi-scale fusion
        # --------------------------------------------------
        final_heat = np.mean(heatmaps, axis=0)
        final_heat = (final_heat - final_heat.min()) / (
            final_heat.max() - final_heat.min() + 1e-8
        )

        return final_heat, baseline_label
