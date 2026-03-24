import torch
import numpy as np
import cv2
from skimage.segmentation import slic
from scipy.stats import qmc
from typing import List, Tuple

class IPEMImprovedExplainer:
    def __init__(self, model: torch.nn.Module, class_names: List[str], device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.eval()
        self.class_names = class_names
        self.device = device

    def _generate_qmc_masks(self, n_samples: int, shape: Tuple[int, ...], mask_prob: float = 0.5):
        """
        [CẢI TIẾN 1]: Khởi tạo Quasi-Monte Carlo masks sử dụng Sobol sequence.
        Quasi-Monte Carlo phủ đều không gian mẫu tĩnh, giúp hội tụ phương sai
        nhanh hơn rất nhiều so với Random thông thường.
        """
        dim = int(np.prod(shape))
        
        # Tạo bộ sinh Sobol (Chỉ hỗ trợ d <= 21201)
        sampler = qmc.Sobol(d=dim, scramble=True)
        
        # Lấy mẫu
        sobol_samples = sampler.random(n=n_samples) # (n_samples, dim)
        
        # Ngưỡng (Thresholding) để tạo boolean mask -> float mask
        masks = (sobol_samples < mask_prob).astype(np.float32)
        
        masks = masks.reshape((n_samples, *shape))
        return torch.from_numpy(masks).to(self.device)

    def explain_by_slic(
        self, 
        img_tensor: torch.Tensor,
        max_samples: int = 400,
        mask_prob: float = 0.5,
        sigma_smooth: float = 6.0,
        n_segments_list: list = (40, 80, 120, 180),
        compactness: float = 10.0,
        slic_sigma: float = 1.0,
        batch_size: int = 64,
        tolerance: float = 5e-3  # Ngưỡng hội tụ Early Stopping
    ):
        self.model.eval()
        C, H, W = img_tensor.shape
        img = img_tensor.to(self.device)

        with torch.no_grad():
            baseline_logits = self.model(img.unsqueeze(0))
            baseline_label = baseline_logits.argmax(dim=1).item()
        
        # ----------------------------------------------------------------------------------
        # [CẢI TIẾN 2]: Tạo ảnh nền mờ (Blurred baseline) thay vì nền đen để tránh Out-Of-Distribution (OOD)
        # Sử dụng blur pool hoặc gaussian filter để tạo thông tin nền trung tính
        # ----------------------------------------------------------------------------------
        img_blurred = torch.nn.functional.avg_pool2d(
            img.unsqueeze(0), kernel_size=11, stride=1, padding=5
        )

        img_np = img_tensor.detach().cpu().float().numpy().transpose(1, 2, 0)
        img_min, img_max = img_np.min(), img_np.max()
        img_segment = (img_np - img_min) / (img_max - img_min + 1e-8)

        heatmaps = []

        for n_segments in n_segments_list:
            # Sinh SLIC siêu điểm ảnh (Superpixels)
            segments = slic(
                img_segment, n_segments=n_segments, compactness=compactness,
                sigma=slic_sigma, start_label=0, channel_axis=-1
            )
            segments = segments.astype(np.int64)
            K = int(segments.max()) + 1
            segments_t = torch.from_numpy(segments).to(self.device)

            # Khởi tạo Mask bằng Sobol Quasi-Monte Carlo
            masks_sp = self._generate_qmc_masks(max_samples, (K,), mask_prob)

            # Đảm bảo mask không rỗng
            empty_rows = masks_sp.sum(dim=1) == 0
            if empty_rows.any():
                rand_idx = torch.randint(0, K, (empty_rows.sum().item(),), device=self.device)
                masks_sp[empty_rows, rand_idx] = 1.0

            importance_sp = torch.zeros(K, device=self.device)
            prev_importance_sp = torch.zeros(K, device=self.device)
            
            probs_all = []
            completed_samples = 0
            
            # ----------------------------------------------------------------------------------
            # [CẢI TIẾN 3]: Batched processing với Early Stopping (Dừng sớm)
            # ----------------------------------------------------------------------------------
            for start in range(0, max_samples, batch_size):
                end = min(start + batch_size, max_samples)
                current_batch_size = end - start
                
                batch_masks_sp = masks_sp[start:end]
                pixel_masks = batch_masks_sp[:, segments_t].unsqueeze(1) # (B, 1, H, W)
                
                # Trộn ảnh gốc và ảnh mờ
                # Vùng nào mask = 1 -> Lấy ảnh gốc
                # Vùng nào mask = 0 -> Lấy ảnh bị làm mờ (Xóa bỏ OOD)
                perturbed_imgs = (img.unsqueeze(0) * pixel_masks) + (img_blurred * (1.0 - pixel_masks))
                
                with torch.no_grad():
                    preds = self.model(perturbed_imgs)
                    probs = torch.softmax(preds, dim=1)[:, baseline_label]
                    probs_all.append(probs)
                
                completed_samples += current_batch_size
                
                # Cần chạy vài chục sample để bắt đầu tính phương sai ổn định
                if completed_samples >= min(32, max_samples//2):
                    current_probs = torch.cat(probs_all, dim=0).view(-1, 1) # (n_curr, 1)
                    current_masks = masks_sp[:completed_samples]            # (n_curr, K)
                    
                    on_mask = current_masks
                    off_mask = 1.0 - current_masks

                    cnt_on = on_mask.sum(dim=0) + 1e-8
                    cnt_off = off_mask.sum(dim=0) + 1e-8

                    mean_on = (current_probs * on_mask).sum(dim=0) / cnt_on
                    mean_off = (current_probs * off_mask).sum(dim=0) / cnt_off

                    var_on = (((current_probs - mean_on.unsqueeze(0)) ** 2) * on_mask).sum(dim=0) / cnt_on
                    var_off = (((current_probs - mean_off.unsqueeze(0)) ** 2) * off_mask).sum(dim=0) / cnt_off

                    # ----------------------------------------------------------------------------------
                    # [CẢI TIẾN 4]: Chỉnh sửa công thức thống kê theo chuẩn Standard Error (Welch's t-test)
                    # ----------------------------------------------------------------------------------
                    se = torch.sqrt((var_on / cnt_on) + (var_off / cnt_off) + 1e-8)
                    importance_sp = (mean_on - mean_off) / se
                    
                    # Cập nhật và kiểm tra Early Stopping
                    if completed_samples > batch_size:
                        max_diff = torch.max(torch.abs(importance_sp - prev_importance_sp))
                        if max_diff < tolerance:
                            # Phương sai hội tụ -> Thuật toán Dừng Sớm -> Tối ưu thời gian!
                            break
                            
                    prev_importance_sp = importance_sp.clone()

            # Lấy Map ở cấp độ Pixel
            heat = importance_sp[segments_t].detach().cpu().numpy()

            if sigma_smooth > 0:
                heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=sigma_smooth)
            
            # [CẢI TIẾN 5]: Chỉ lưu vào mảng, không Min-Max Normalize sớm làm hỏng Tỷ lệ phương sai
            heatmaps.append(heat)

        # ----------------------------------------------------------------------------------
        # [CẢI TIẾN 5]: Chuẩn hóa Min-Max duy nhất 1 lần ở bước Cuối Cùng
        # ----------------------------------------------------------------------------------
        final_heat = np.mean(heatmaps, axis=0)
        final_heat = (final_heat - np.min(final_heat)) / (np.max(final_heat) - np.min(final_heat) + 1e-8)

        return final_heat, baseline_label
