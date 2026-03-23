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

    # def perturb_patch(self, img_np, y1, y2, x1, x2, mode="black_blur"):
    #     """
    #     Perturb 1 patch bằng cách:
    #     - Tắt pixel trong vùng patch
    #     - Dùng blur để smooth vùng bị mask
    #     img_np: H x W x 3, float [0,1]
    #     """
    #     perturbed = img_np.copy()

    #     # 1. Tắt hoàn toàn patch
    #     perturbed[y1:y2, x1:x2, :] = 0.0

    #     # 2. Blur toàn ảnh (nhẹ) để smooth biên
    #     if mode == "black_blur":
    #         blurred = cv2.GaussianBlur(perturbed, (11, 11), 0)

    #         # 3. Chỉ lấy vùng patch đã blur
    #         perturbed[y1:y2, x1:x2, :] = blurred[y1:y2, x1:x2, :]

    #     return perturbed

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

    # def _generate_masks(self, n_superpixels, segments, n_samples, H, W, mask_prob=0.5):
    #     """
    #     Tạo các tổ hợp mask ngẫu nhiên
    #     """
    #     masks = []
    #     H_up = int(H * 1.5)
    #     W_up = int(W * 1.5)
        
    #     segments_up = cv2.resize(segments.astype(np.float32), (W_up, H_up), 
    #                         interpolation=cv2.INTER_NEAREST).astype(np.int32)

    #     for _ in range(n_samples):
    #         fragment_mask = np.random.binomial(1, mask_prob, n_superpixels)
    #         mask_up = np.zeros((H_up, W_up), dtype=np.float32)

    #         for seg_id in range(n_superpixels):
    #             if fragment_mask[seg_id] == 1:
    #                 mask_up[segments_up == seg_id] = 1.0

    #         y_start = np.random.randint(0, H_up - H + 1)
    #         x_start = np.random.randint(0, W_up - W + 1)
    #         mask = mask_up[y_start:y_start+H, x_start:x_start+W]

    #         # mask = cv2.GaussianBlur(mask, (11, 11), 5)
    #         masks.append(mask)

    #     return masks
    
    def explain(
        self,
        img_tensor: torch.Tensor,
        n_samples: int = 400,
        mask_prob: float = 0.5,
        sigma_smooth: float = 0.5,
        grid_sizes: list = [(8, 8), (4, 4)]
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
            masks = torch.rand(n_samples, gh, gw, device=self.device) < mask_prob
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
        n_segments: int = 50,
        compactness: float = 10.0,
        sigma: float = 1.0,
        sigma_smooth: float = 6.0,
    ):
        """
        Parameters
        ----------
        img_tensor   : torch.Tensor (C, H, W), float, giá trị [0,1]
        n_samples    : số mẫu Monte Carlo
        mask_prob    : xác suất một superpixel được GIỮ LẠI (=1)
        n_segments   : số superpixels mong muốn cho SLIC
        compactness  : tham số SLIC – cân bằng màu sắc vs khoảng cách không gian
        sigma        : Gaussian smoothing trước SLIC
        sigma_smooth : Gaussian blur cho heatmap cuối
        """
        self.model.eval()
        C, H, W = img_tensor.shape
        img = img_tensor.to(self.device)

        # ------------------------------------------------------------------
        # 1. Baseline prediction
        # ------------------------------------------------------------------
        with torch.no_grad():
            baseline_logits = self.model(img.unsqueeze(0))
            baseline_label = baseline_logits.argmax(dim=1).item()

        # ------------------------------------------------------------------
        # 2. SLIC segmentation
        # ------------------------------------------------------------------
        # SLIC cần ảnh HWC float [0,1]
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)

        segments = slic(
            img_np,
            n_segments=n_segments,
            compactness=compactness,
            sigma=sigma,
            start_label=0,
            channel_axis=-1,
        )
        # segments: (H, W), giá trị nguyên 0..K-1
        segment_ids = np.unique(segments)
        K = len(segment_ids)            # số superpixels thực tế

        # Remap về 0..K-1 liên tục (phòng trường hợp SLIC skip id)
        remap = {old: new for new, old in enumerate(segment_ids)}
        segments_remap = np.vectorize(remap.get)(segments)  # (H, W)

        # Tensor segment map để dùng sau
        seg_t = torch.from_numpy(segments_remap).long().to(self.device)  # (H, W)

        # ------------------------------------------------------------------
        # 3. Monte Carlo binary masks trên không gian superpixel
        # ------------------------------------------------------------------
        # masks_sp: (n_samples, K)  – 1 = superpixel được giữ, 0 = bị xoá
        masks_sp = (torch.rand(n_samples, K, device=self.device) < mask_prob).float()

        # ------------------------------------------------------------------
        # 4. Upsample mask từ superpixel → pixel (vectorized)
        # ------------------------------------------------------------------
        # seg_t (H*W,) → dùng làm index chọn cột trong masks_sp
        seg_flat = seg_t.view(-1)                           # (H*W,)
        masks_px = masks_sp[:, seg_flat]                    # (n_samples, H*W)
        masks_up = masks_px.view(n_samples, 1, H, W)       # (n_samples, 1, H, W)

        # ------------------------------------------------------------------
        # 5. Perturb & batch inference
        # ------------------------------------------------------------------
        perturbed_imgs = img.unsqueeze(0) * masks_up       # (n_samples, C, H, W)

        with torch.no_grad():
            preds = self.model(perturbed_imgs)
            probs = torch.softmax(preds, dim=1)[:, baseline_label]  # (n_samples,)

        # ------------------------------------------------------------------
        # 6. Monte Carlo importance per superpixel (vectorized)
        # ------------------------------------------------------------------
        # probs: (n_samples,) → (n_samples, 1)
        probs_col = probs.unsqueeze(1)                       # (n_samples, 1)

        on_mask  = masks_sp                                  # (n_samples, K)
        off_mask = 1.0 - masks_sp

        cnt_on  = on_mask.sum(dim=0)  + 1e-8                # (K,)
        cnt_off = off_mask.sum(dim=0) + 1e-8

        mean_on  = (probs_col * on_mask ).sum(dim=0) / cnt_on
        mean_off = (probs_col * off_mask).sum(dim=0) / cnt_off

        var_on  = ((probs_col - mean_on )**2 * on_mask ).sum(dim=0) / cnt_on
        var_off = ((probs_col - mean_off)**2 * off_mask).sum(dim=0) / cnt_off

        importance_sp = (mean_on - mean_off) / (torch.sqrt(var_on + var_off) + 1e-8)

        # Normalize về [0, 1]
        imp_min, imp_max = importance_sp.min(), importance_sp.max()
        importance_sp = (importance_sp - imp_min) / (imp_max - imp_min + 1e-8)

        # ------------------------------------------------------------------
        # 7. Map importance về không gian ảnh
        # ------------------------------------------------------------------
        # Mỗi pixel lấy giá trị importance của superpixel nó thuộc về
        heatmap_flat = importance_sp[seg_flat]              # (H*W,)
        heatmap = heatmap_flat.view(H, W).cpu().numpy()    # (H, W)

        # ------------------------------------------------------------------
        # 8. Smooth & normalize lần cuối
        # ------------------------------------------------------------------
        if sigma_smooth > 0:
            heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=sigma_smooth)

        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        return heatmap, baseline_label
