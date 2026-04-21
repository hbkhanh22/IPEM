from typing import List, Tuple
import numpy as np
import torch
import cv2
from skimage.segmentation import watershed
from skimage.filters import sobel
from skimage.util import img_as_float

class IPEMExplainer:
    def __init__(self, model: torch.nn.Module, class_names: List[str], grid_size: Tuple[int,int]=(4,4), perturb_modes: List[str]=["black", "blur", "mean", "noise"], device=None):
        """Initialize the IPEM explainer with model metadata and perturbation settings.
        
        Args:
            model: Trained deep learning model used for explanation.
            class_names: Ordered list of class labels.
            grid_size: Spatial grid size used by grid-based variants.
            perturb_modes: Supported perturbation strategies.
            device: Target device for inference and explanation.
        
        Return:
            None.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.eval()
        self.class_names = class_names
        self.grid_size = grid_size
        self.perturb_modes = perturb_modes
        self.device = device

    def _get_baseline_label(self, img: torch.Tensor) -> int:
        """Predict the baseline class label for a single input image tensor.
        
        Args:
            img: Input image tensor in CHW format.
        
        Return:
            int: Predicted class index for the original image.
        """
        with torch.no_grad():
            baseline_logits = self.model(img.unsqueeze(0))
            return baseline_logits.argmax(dim=1).item()

    def _prepare_image_for_segmentation(self, img_tensor: torch.Tensor) -> np.ndarray:
        """Convert a tensor image into a normalized NumPy array suitable for segmentation.
        
        Args:
            img_tensor: Input image tensor in CHW format.
        
        Return:
            np.ndarray: Float image array normalized for segmentation.
        """
        img_np = img_tensor.detach().cpu().float().numpy().transpose(1, 2, 0)
        img_min = img_np.min()
        img_max = img_np.max()
        img_np = (img_np - img_min) / (img_max - img_min + 1e-8)
        return img_as_float(img_np)

    def _build_watershed_segments(
        self,
        img_segment: np.ndarray,
        n_markers: int = 100,
        compactness: float = 0.001,
        sigma: float = 1.0
    ) -> np.ndarray:
        """Generate watershed-based superpixel segments from an input image.
        
        Args:
            img_segment: Image used to compute watershed regions.
            n_markers: Approximate number of watershed markers.
            compactness: Compactness parameter passed to watershed.
            sigma: Gaussian smoothing strength applied before edge detection.
        
        Return:
            np.ndarray: Integer segment map with zero-based segment ids.
        """
        H, W = img_segment.shape[:2]

        if img_segment.ndim == 3 and img_segment.shape[2] >= 3:
            gray = cv2.cvtColor((img_segment * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            gray = gray.astype(np.float32) / 255.0
        elif img_segment.ndim == 3:
            gray = img_segment[..., 0]
        else:
            gray = img_segment

        if sigma > 0:
            gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)

        gradient = sobel(gray)

        n_markers = max(4, int(n_markers))
        marker_rows = max(2, int(np.sqrt(n_markers * H / max(W, 1))))
        marker_cols = max(2, int(np.ceil(n_markers / marker_rows)))

        ys = np.linspace(0, H - 1, marker_rows, dtype=np.int32)
        xs = np.linspace(0, W - 1, marker_cols, dtype=np.int32)

        markers = np.zeros((H, W), dtype=np.int32)
        label = 1
        for y in ys:
            for x in xs:
                markers[y, x] = label
                label += 1

        segments = watershed(
            gradient,
            markers=markers,
            compactness=compactness,
        )
        return segments.astype(np.int64) - 1

    def _explain_with_segments(
        self,
        img: torch.Tensor,
        baseline_label: int,
        segments: np.ndarray,
        n_samples: int,
        mask_prob: float,
        sigma_smooth: float,
        batch_size: int
    ) -> np.ndarray:
        """Estimate segment importance scores by perturbing superpixels and comparing model confidence.
        
        Args:
            img: Input image tensor in CHW format.
            baseline_label: Predicted class index used as the explanation target.
            segments: Segment map defining superpixel regions.
            n_samples: Number of random perturbation samples.
            mask_prob: Probability that a segment remains visible.
            sigma_smooth: Gaussian smoothing strength for the final heatmap.
            batch_size: Number of perturbed images processed per batch.
        
        Return:
            np.ndarray: Normalized heatmap derived from segment importance scores.
        """
        segments = segments.astype(np.int64)
        K = int(segments.max()) + 1
        if K <= 0:
            raise ValueError("Segmentation must contain at least one segment.")

        segments_t = torch.from_numpy(segments).to(self.device)
        masks_sp = (torch.rand(n_samples, K, device=self.device) < mask_prob).float()

        empty_rows = masks_sp.sum(dim=1) == 0
        if empty_rows.any():
            rand_idx = torch.randint(0, K, (empty_rows.sum().item(),), device=self.device)
            masks_sp[empty_rows, rand_idx] = 1.0

        pixel_masks = masks_sp[:, segments_t].unsqueeze(1)
        perturbed_imgs = img.unsqueeze(0) * pixel_masks

        probs_all = []
        with torch.no_grad():
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                preds = self.model(perturbed_imgs[start:end])
                probs_all.append(torch.softmax(preds, dim=1)[:, baseline_label])

        probs = torch.cat(probs_all, dim=0).view(-1, 1)

        on_mask = masks_sp
        off_mask = 1.0 - masks_sp

        cnt_on = on_mask.sum(dim=0) + 1e-8
        cnt_off = off_mask.sum(dim=0) + 1e-8

        mean_on = (probs * on_mask).sum(dim=0) / cnt_on
        mean_off = (probs * off_mask).sum(dim=0) / cnt_off

        var_on = (((probs - mean_on.unsqueeze(0)) ** 2) * on_mask).sum(dim=0) / cnt_on
        var_off = (((probs - mean_off.unsqueeze(0)) ** 2) * off_mask).sum(dim=0) / cnt_off

        importance_sp = (mean_on - mean_off) / (torch.sqrt(var_on + var_off) + 1e-8)
        importance_sp = (importance_sp - importance_sp.min()) / (
            importance_sp.max() - importance_sp.min() + 1e-8
        )

        heat = importance_sp[segments_t].detach().cpu().numpy()
        if sigma_smooth > 0:
            heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=sigma_smooth)

        return (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

    def explain_by_watershed(
        self,
        img_tensor: torch.Tensor,
        n_samples: int = 500,
        mask_prob: float = 0.5,
        sigma_smooth: float = 5.0,
        n_segments_list: list = [80, 120],
        watershed_sigma: float = 1.0,
        compactness: float = 0.001,
        batch_size: int = 64
    ):
        """Generate a watershed-based explanation for the input image.
        
        Args:
            img_tensor: Input image tensor in CHW format.
            n_samples: Number of random perturbation samples.
            mask_prob: Probability that a segment remains visible.
            sigma_smooth: Gaussian smoothing strength for the final heatmap.
            n_segments_list: List of number of segments to use.
            watershed_sigma: Gaussian smoothing strength for the watershed segments.
            compactness: Compactness parameter passed to watershed.
            batch_size: Number of perturbed images processed per batch.
        
        Return:
            np.ndarray: Normalized heatmap derived from segment importance scores.
        """
        self.model.eval()
        img = img_tensor.to(self.device)
        baseline_label = self._get_baseline_label(img)
        img_segment = self._prepare_image_for_segmentation(img_tensor)

        heatmaps = []
        for n_segments in n_segments_list:
            segments = self._build_watershed_segments(
                img_segment,
                n_markers=n_segments,
                compactness=compactness,
                sigma=watershed_sigma
            )
            heat = self._explain_with_segments(
                img=img,
                baseline_label=baseline_label,
                segments=segments,
                n_samples=n_samples,
                mask_prob=mask_prob,
                sigma_smooth=sigma_smooth,
                batch_size=batch_size
            )
            heatmaps.append(heat)

        final_heat = np.mean(heatmaps, axis=0)
        final_heat = (final_heat - final_heat.min()) / (
            final_heat.max() - final_heat.min() + 1e-8
        )

        return final_heat