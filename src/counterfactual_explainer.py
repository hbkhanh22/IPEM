"""
Counterfactual Visual Explainer

Phương pháp giải thích dựa trên counterfactual:
1. Mô hình dự đoán ảnh thuộc class c
2. Tìm 1 ảnh tương tự trong cùng class c
3. Mapping các vùng quan trọng giữa 2 ảnh để giải thích quyết định
"""

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image
from torchvision import transforms
from skimage.segmentation import slic
from skimage.measure import regionprops
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity
import torchvision.models as models


class CounterfactualExplainer:
    def __init__(self, model: torch.nn.Module, class_names: List[str], 
                 data_dir: str, device=None, img_size: int = 224):
        """
        Counterfactual Visual Explainer
        
        Args:
            model: Mô hình đã train
            class_names: Danh sách tên các class
            data_dir: Đường dẫn đến dataset
            device: CPU/GPU
            img_size: Kích thước ảnh
        """
        self.model = model.eval()
        self.class_names = class_names
        self.data_dir = Path(data_dir)
        self.device = device if device else next(model.parameters()).device
        self.img_size = img_size
        
        # Transform cho ảnh
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Load dataset để tìm ảnh tương tự
        self._load_dataset()
    
    def _denormalize_uint8(self, img_tensor: torch.Tensor) -> np.ndarray:
        """Denormalize tensor (C,H,W) về ảnh uint8 HxWxC."""
        img_np = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_uint8 = np.clip((img_np * std + mean) * 255, 0, 255).astype(np.uint8)
        return img_uint8
        
    def _load_dataset(self):
        """Load dataset để tìm ảnh tương tự"""
        self.dataset_images = {}
        
        # Load ảnh từ dataset animals
        animals_dir = self.data_dir / "animals"
        if animals_dir.exists():
            for class_name in self.class_names:
                class_dir = animals_dir / class_name
                if class_dir.exists():
                    image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
                    self.dataset_images[class_name] = image_files[:100]  # Giới hạn 100 ảnh mỗi class
        
        # Load ảnh từ dataset COVID nếu có
        covid_dir = self.data_dir / "SARS-COV2-CT"
        if covid_dir.exists():
            for class_name in self.class_names:
                class_dir = covid_dir / class_name
                if class_dir.exists():
                    image_files = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
                    if class_name not in self.dataset_images:
                        self.dataset_images[class_name] = []
                    self.dataset_images[class_name].extend(image_files[:50])  # Thêm 50 ảnh mỗi class
    
    def _extract_features(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Trích xuất features từ ảnh sử dụng model"""
        with torch.no_grad():
            # Lấy features từ layer trước classifier
            if hasattr(self.model, 'classifier') and hasattr(self.model, 'features'):
                # EfficientNet, MobileNet
                features = self.model.features(img_tensor)
                features = F.adaptive_avg_pool2d(features, (1, 1))
                features = features.view(features.size(0), -1)
            elif hasattr(self.model, 'fc') and hasattr(self.model, 'avgpool'):
                # ResNet
                x = self.model.conv1(img_tensor)
                x = self.model.bn1(x)
                x = self.model.relu(x)
                x = self.model.maxpool(x)
                x = self.model.layer1(x)
                x = self.model.layer2(x)
                x = self.model.layer3(x)
                x = self.model.layer4(x)
                features = self.model.avgpool(x)
                features = features.view(features.size(0), -1)
            elif hasattr(self.model, 'features') and hasattr(self.model, 'avgpool'):
                # VGG
                features = self.model.features(img_tensor)
                features = self.model.avgpool(features)
                features = features.view(features.size(0), -1)
            else:
                # Fallback: sử dụng toàn bộ model và lấy features từ layer cuối
                features = self.model(img_tensor)
                # Nếu là logits, lấy features từ layer trước đó
                if features.dim() == 2 and features.size(1) == len(self.class_names):
                    # Tạo một tensor features giả từ logits
                    features = F.softmax(features, dim=1)
            
            return features
    
    def _find_similar_image(
        self,
        img_tensor: torch.Tensor,
        predicted_class: int,
        exclude_image_uint8: Optional[np.ndarray] = None,
        exclude_path: Optional[str] = None,
    ) -> Tuple[Image.Image, str, float]:
        """
        Tìm ảnh tương tự nhất trong cùng class
        
        Args:
            img_tensor: Ảnh input (C, H, W)
            predicted_class: Class được dự đoán
            
        Returns:
            (similar_image, image_path, similarity_score)
        """
        class_name = self.class_names[predicted_class]
        
        if class_name not in self.dataset_images or not self.dataset_images[class_name]:
            raise ValueError(f"Không tìm thấy ảnh nào trong class {class_name}")
        
        # Đảm bảo img_tensor ở đúng device
        img_tensor = img_tensor.to(self.device)
        
        # Trích xuất features của ảnh input
        input_features = self._extract_features(img_tensor.unsqueeze(0))
        
        best_similarity = -1
        best_image = None
        best_path = None

        # Tạo hash cho ảnh cần loại trừ (nếu có)
        exclude_hash = None
        if exclude_image_uint8 is not None:
            try:
                exclude_hash = hash(exclude_image_uint8.tobytes())
            except Exception:
                exclude_hash = None
        
        # Tìm ảnh tương tự nhất
        for img_path in self.dataset_images[class_name]:
            try:
                # Loại trừ theo đường dẫn nếu trùng
                if exclude_path is not None and str(img_path) == str(exclude_path):
                    continue
                # Load và preprocess ảnh
                img = Image.open(img_path).convert('RGB')
                img_tensor_candidate = self.transform(img).unsqueeze(0).to(self.device)
                
                # Trích xuất features
                candidate_features = self._extract_features(img_tensor_candidate)
                
                # Tính cosine similarity
                similarity = F.cosine_similarity(input_features, candidate_features).item()
                
                # Nếu giống hệt nội dung thì bỏ qua (tránh trùng ảnh dự đoán)
                if exclude_hash is not None:
                    try:
                        cand_np = (img_tensor_candidate.squeeze(0).permute(1, 2, 0).detach().cpu().numpy())
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        cand_uint8 = np.clip((cand_np * std + mean) * 255, 0, 255).astype(np.uint8)
                        cand_hash = hash(cand_uint8.tobytes())
                        if cand_hash == exclude_hash:
                            continue
                    except Exception:
                        pass

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_image = img
                    best_path = str(img_path)
                    
            except Exception as e:
                print(f"Lỗi khi xử lý ảnh {img_path}: {e}")
                continue
        
        return best_image, best_path, best_similarity
    
    
