import numpy as np
import torch
from typing import List
from lime import lime_image
from ipem_explainer import IPEMExplainer  
from rise_explainer import RISE
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
        """Initialize the XAI evaluator with a trained model and its class labels.
        
        Args:
            model: Trained PyTorch model used for explanation and scoring.
            class_names: Ordered list of class labels.
        
        Return:
            None.
        """
        self.model = model.eval()   
        self.class_names = class_names

    @staticmethod
    def _synchronize_device(device: torch.device):
        """Synchronize the active accelerator device so runtime measurements remain accurate.
        
        Args:
            device: Torch device to synchronize.
        
        Return:
            None.
        """
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps" and hasattr(torch, "mps"):
            try:
                torch.mps.synchronize()
            except Exception:
                pass

    # =====================================================================
    # --------- Shared explanation evaluation metrics ---------
    # =====================================================================

    @staticmethod
    def gini_sparsity(heatmap):
        """Measure explanation sparsity with the Gini coefficient.
        
        Args:
            heatmap: Explanation map whose sparsity is being measured.
        
        Return:
            float: Gini sparsity score for the provided heatmap.
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
        """Measure explanation faithfulness with insertion or deletion perturbation curves.
        
        Args:
            model: Trained model used to score perturbed inputs.
            img_tensor: Original image tensor in CHW or NCHW format.
            heatmap: Explanation heatmap associated with the image.
            target_class: Class index whose confidence is tracked.
            steps: Number of perturbation steps used to build the curve.
            mode: Either insertion or deletion.
        
        Return:
            float: Area under the insertion or deletion curve.
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
        indices = np.argsort(-heatmap_flat)  # Sort in descending order

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
        """Compute Average Drop and Increase Rate from original and masked prediction scores.
        
        Args:
            original_probs: Confidence values predicted for the original inputs.
            masked_probs: Confidence values predicted for the masked inputs.
        
        Return:
            tuple[float, float]: Average drop percentage and increase rate percentage.
        """
        original_probs = np.array(original_probs)
        masked_probs = np.array(masked_probs)

        drops = np.maximum(0, original_probs - masked_probs)
        avg_drop = np.mean(drops / (original_probs + 1e-8)) * 100  

        increases = (masked_probs > original_probs).astype(np.float32)
        increase_rate = np.mean(increases) * 100  

        return avg_drop, increase_rate
    
    # --------- AOPC MoRF ----------
    def _compute_single_aopc(self, img_tensor, explanation, original_class, original_prob, block_size=8, percentile=None, verbose=False):
        """Compute the AOPC MoRF score for a single explanation map and image.
        
        Args:
            img_tensor: Original image tensor or image array.
            explanation: Explanation heatmap used to rank image regions.
            original_class: Predicted class index before perturbation.
            original_prob: Class probability vector for the original image.
            block_size: Side length of each perturbation block.
            percentile: Optional cumulative importance threshold used to stop early.
            verbose: Whether to print intermediate progress information.
        
        Return:
            tuple: Original class index, AOPC curve values, and mean AOPC score.
        """
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
    # --------- Apply the metrics to the four XAI methods ---------
    # =====================================================================

    # --------- LIME ----------
    def evaluate_with_lime(self, samples, num_samples=1000, compute_aopc=True, block_size=8, percentile=None, noise_sigma=0.02):
        """Evaluate LIME explanations across the provided samples and aggregate multiple faithfulness metrics.
        
        Args:
            samples: Iterable of image tensors and labels used for evaluation.
            num_samples: Number of perturbation samples requested from LIME.
            compute_aopc: Whether to compute AOPC-based metrics.
            block_size: Block size used for AOPC perturbations.
            percentile: Optional cumulative importance threshold used in AOPC.
            noise_sigma: Noise level used for local stability estimation.
        
        Return:
            dict: Aggregated LIME evaluation metrics.
        """
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
        
        # Local Stability: pick one sample to avoid repeating the full LIME cost for many iterations
        stability_val = 0.0
        if len(samples) > 0:
            img_t, _ = samples[0] # take one sample
            if img_t.dim() == 2: img_t = img_t.unsqueeze(0)
            # Create a lightly perturbed version of the sample
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

    # --------- IPEM ----------
    def evaluate_with_ipem(self, loader, compute_aopc=True, block_size=8, percentile=None, noise_sigma=0.02):
        """Evaluate IPEM explanations on a dataloader and aggregate multiple explanation quality metrics.
        
        Args:
            loader: DataLoader that supplies evaluation batches.
            compute_aopc: Whether to compute AOPC-based metrics.
            block_size: Block size used for AOPC perturbations.
            percentile: Optional cumulative importance threshold used in AOPC.
            noise_sigma: Noise level used for local stability estimation.
        
        Return:
            dict: Aggregated IPEM evaluation metrics.
        """
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
                heat = ipem.explain_by_watershed(img_t.to(device))
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
        """Evaluate GradCAM explanations on a dataloader and aggregate explanation quality metrics.
        
        Args:
            loader: DataLoader that supplies evaluation batches.
            compute_aopc: Whether to compute AOPC-based metrics.
            block_size: Block size used for AOPC perturbations.
            percentile: Optional cumulative importance threshold used in AOPC.
            noise_sigma: Noise level used for local stability estimation.
        
        Return:
            dict: Aggregated GradCAM evaluation metrics.
        """
        def get_last_conv_layer(model):
            """Locate the final convolutional layer required to run GradCAM.
            
            Args:
                model: Model whose modules are inspected.
            
            Return:
                nn.Module: Last convolutional layer found in the model.
            """
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
        """Evaluate RISE explanations on a dataloader and aggregate explanation quality metrics.
        
        Args:
            loader: DataLoader that supplies evaluation batches.
            compute_aopc: Whether to compute AOPC-based metrics.
            block_size: Block size used for AOPC perturbations.
            percentile: Optional cumulative importance threshold used in AOPC.
            noise_sigma: Noise level used for local stability estimation.
        
        Return:
            dict: Aggregated RISE evaluation metrics.
        """
        rise_explainer = RISE(
            model=self.model,
            n_masks=500,
            p=0.5,
            initial_mask_size=(7, 7),
        )
        device = next(self.model.parameters()).device
        all_gini, all_insertion, all_deletion = [], [], []
        all_stability = []
        all_aopc_scores = []
        rise_orig_probs, rise_masked_probs = [], []
        explanation_times = []
        for b_idx, (batch_imgs, batch_labels) in enumerate(loader):
            print(f"Batch {b_idx} of {len(loader)}")
            imgs = batch_imgs.to(device)
            y_model = self.model(imgs).argmax(1).cpu().numpy()

            for i in range(len(imgs)):
                img_tensor = imgs[i].unsqueeze(0)
                self._synchronize_device(device)
                explanation_start = time.perf_counter()
                heatmap = rise_explainer.explain(img_tensor)
                self._synchronize_device(device)
                explanation_times.append(time.perf_counter() - explanation_start)
                # RISE returns (n_classes, H, W); select the channel that matches the predicted class
                if torch.is_tensor(heatmap):
                    heatmap_abs = np.abs(heatmap[int(y_model[i])].detach().cpu().numpy())
                else:
                    heatmap_abs = np.abs(np.asarray(heatmap)[int(y_model[i])])
                _, _, h_img, w_img = img_tensor.shape
                if heatmap_abs.shape != (h_img, w_img):
                    heatmap_abs = cv2.resize(
                        heatmap_abs.astype(np.float32),
                        (int(w_img), int(h_img)),
                        interpolation=cv2.INTER_LINEAR,
                    )

                # Gini Sparsity
                all_gini.append(self.gini_sparsity(heatmap_abs))
                
                # Insertion & Deletion
                all_insertion.append(self.insertion_deletion_score(self.model, img_tensor, heatmap_abs, y_model[i], steps=20, mode="insertion"))
                all_deletion.append(self.insertion_deletion_score(self.model, img_tensor, heatmap_abs, y_model[i], steps=20, mode="deletion"))

                # Local Stability
                img_noisy = (img_tensor + torch.randn_like(img_tensor) * noise_sigma).clamp(-5.0, 5.0)
                heatmap_noisy = rise_explainer.explain(img_noisy)
                if torch.is_tensor(heatmap_noisy):
                    hm_noisy_2d = np.abs(heatmap_noisy[int(y_model[i])].detach().cpu().numpy())
                else:
                    hm_noisy_2d = np.abs(np.asarray(heatmap_noisy)[int(y_model[i])])
                if hm_noisy_2d.shape != (h_img, w_img):
                    hm_noisy_2d = cv2.resize(
                        hm_noisy_2d.astype(np.float32),
                        (int(w_img), int(h_img)),
                        interpolation=cv2.INTER_LINEAR,
                    )
                stab = np.linalg.norm(vectorize_explanation(heatmap_abs) - vectorize_explanation(hm_noisy_2d))
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

                        with torch.no_grad():
                            masked_out = self.model(img_tensor * mask_t)
                            masked_prob = torch.softmax(masked_out, dim=1)[0, original_class].item()
                        rise_orig_probs.append(float(original_prob[original_class].item()))
                        rise_masked_probs.append(float(masked_prob))
                        
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
    
