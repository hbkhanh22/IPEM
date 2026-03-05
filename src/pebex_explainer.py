from typing import List, Tuple
import numpy as np
import torch
import cv2
from skimage.segmentation import slic

class PEBEXExplainer:
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

    def perturb_patch(self, img_np, y1, y2, x1, x2, mode="black_blur"):
        """
        Perturb 1 patch bằng cách:
        - Tắt pixel trong vùng patch
        - Dùng blur để smooth vùng bị mask
        img_np: H x W x 3, float [0,1]
        """
        perturbed = img_np.copy()

        # 1. Tắt hoàn toàn patch
        perturbed[y1:y2, x1:x2, :] = 0.0

        # 2. Blur toàn ảnh (nhẹ) để smooth biên
        if mode == "black_blur":
            blurred = cv2.GaussianBlur(perturbed, (11, 11), 0)

            # 3. Chỉ lấy vùng patch đã blur
            perturbed[y1:y2, x1:x2, :] = blurred[y1:y2, x1:x2, :]

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
            baseline_logits, importance_map
        )
        
        return heat, pred_label_from_importance

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

    def explain_slic(self, img_tensor, n_segments_list=[20, 30], compactness=10.0, n_samples_per_scale=200, mode="black", mask_prob=0.5):
        """
        Sinh heatmap giải thích cho 1 ảnh sử dụng SLIC superpixels
        
        Args:
            img_tensor: tensor (C,H,W) đã normalize
            n_segments: số lượng superpixels
            compactness: độ compact của superpixels
            n_samples: số lượng mask combinations
            mode: cách che ["black", "blur", "mean", "noise"]
        Returns:
            heatmap: importance map (H, W)
            predicted_label: nhãn dự đoán từ importance
        """
        # torch.backends.cudnn.enabled = False
        self.model.eval()
        C, H, W = img_tensor.shape
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()

        with torch.no_grad():
            pred = self.model(img_tensor.unsqueeze(0).to(self.device))
            probs = torch.softmax(pred, dim=1).cpu().numpy()[0]
            baseline_label = np.argmax(probs)
            baseline_probs = probs[baseline_label]
            baseline_logits = pred.cpu().numpy()[0]
        
        importance_scores = np.zeros((H, W))
        total_mask = 0
        
        for n_segments in n_segments_list:
            segments = slic(img_np, n_segments=n_segments, compactness=compactness, start_label=0, sigma=1.0)
            n_superpixels = segments.max() + 1

            masks = self._generate_masks(n_superpixels, segments, n_samples_per_scale, H, W, mask_prob)

            batch_inputs = []

            for mask in masks:
                perturbed = img_np * mask[:, :, np.newaxis]
                pt = torch.tensor(perturbed.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
                batch_inputs.append(pt)

            batch_inputs = torch.cat(batch_inputs, dim=0).to(self.device)

            with torch.no_grad():
                preds = self.model(batch_inputs)
                probs_all = torch.softmax(preds, dim=1).cpu().numpy()[:, baseline_label]
            
            for i, (mask, score) in enumerate(zip(masks, probs_all)):
                importance_scores += score * mask
            
            total_mask += len(masks)
        
        # Normalize importance map
        expected_mask_value = mask_prob
        final_heatmap = importance_scores / (expected_mask_value * total_mask)

        if final_heatmap.max() > final_heatmap.min():
            final_heatmap = (final_heatmap - final_heatmap.min()) / (final_heatmap.max() - final_heatmap.min())


        predicted_label = self._predict_from_importance(baseline_logits, final_heatmap)

        return final_heatmap, predicted_label

    def _generate_masks(self, n_superpixels, segments, n_samples, H, W, mask_prob=0.5):
        """
        Tạo các tổ hợp mask ngẫu nhiên
        """
        masks = []
        H_up = int(H * 1.5)
        W_up = int(W * 1.5)
        
        segments_up = cv2.resize(segments.astype(np.float32), (W_up, H_up), 
                            interpolation=cv2.INTER_NEAREST).astype(np.int32)

        for _ in range(n_samples):
            fragment_mask = np.random.binomial(1, mask_prob, n_superpixels)
            mask_up = np.zeros((H_up, W_up), dtype=np.float32)

            for seg_id in range(n_superpixels):
                if fragment_mask[seg_id] == 1:
                    mask_up[segments_up == seg_id] = 1.0

            y_start = np.random.randint(0, H_up - H + 1)
            x_start = np.random.randint(0, W_up - W + 1)
            mask = mask_up[y_start:y_start+H, x_start:x_start+W]

            # mask = cv2.GaussianBlur(mask, (11, 11), 5)
            masks.append(mask)

        return masks
    
    def explain_one_mc(
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
