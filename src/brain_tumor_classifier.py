import argparse
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.metrics import specificity_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from xai_metrics import XAIEvaluator
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BrainTumorClassifier:
    def __init__(self, data_dir: str, args_model: str = "efficientnet_b3", output_dir: str = "outputs", img_size: int = 224,
                 batch_size: int = 32, epochs: int = 10, lr: float = 1e-4, weight_decay: float = 1e-4,
                 val_split: float = 0.15, test_split: float = 0.15, patience: int = 10, num_workers: int = 4):
        self.data_dir = data_dir
        self.output_dir = Path(output_dir) / "brain-tumor"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.val_split = val_split
        self.test_split = test_split
        self.patience = patience
        self.num_workers = num_workers
        self.class_names = []
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.args_model_name = args_model.lower()

    def _build_dataloaders(self):
        train_tf = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        eval_tf = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        train_dir = os.path.join(self.data_dir, "brain-tumor", "Training")
        test_dir = os.path.join(self.data_dir, "brain-tumor", "Testing")

        full_train_ds = ImageFolder(train_dir, transform=train_tf)
        test_ds = ImageFolder(test_dir, transform=eval_tf)

        self.class_names = full_train_ds.classes

        n_total = len(full_train_ds)
        n_val = int(n_total * self.val_split)
        n_train = n_total - n_val
        train_ds, val_ds = random_split(
            full_train_ds,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(SEED)
        )

        val_ds.dataset.transform = eval_tf

        workers = 0 if os.name == 'nt' else self.num_workers
        pin = (device.type == 'cuda')
        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=workers, pin_memory=pin)
        self.val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=workers, pin_memory=pin)
        self.test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False, num_workers=workers, pin_memory=pin)

    def _build_model(self):
        if self.args_model_name == "efficientnet_b3":
            model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
            in_feats = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_feats, len(self.class_names))

        elif self.args_model_name == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            in_feats = model.fc.in_features
            model.fc = nn.Linear(in_feats, len(self.class_names))

        elif self.args_model_name == "densenet121":
            model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            in_feats = model.classifier.in_features
            model.classifier = nn.Linear(in_feats, len(self.class_names))

        elif self.args_model_name == "mobilenet_v3":
            model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
            in_feats = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(in_feats, len(self.class_names))
        elif self.args_model_name == "vgg19":
            model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
            in_feats = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(in_feats, len(self.class_names))
        elif self.args_model_name == "transformer":
            # Vision Transformer (ViT) support (torchvision >= ~0.14). Handle multiple torchvision APIs.
            try:
                ViTWeights = getattr(models, "ViT_B_16_Weights", None)
                if ViTWeights is not None:
                    model = models.vit_b_16(weights=ViTWeights.IMAGENET1K_V1)
                else:
                    # older torchvision might accept pretrained=True
                    model = models.vit_b_16(pretrained=True)
            except Exception as e:
                raise ValueError(
                    "ViT (vit_b_16) is not available in the installed torchvision. "
                    "Install torchvision with ViT support (e.g. >=0.14.0) or choose another model."
                ) from e

            # Replace classifier/head to match number of classes (handle variants)
            if hasattr(model, 'heads'):
                # torchvision ViT usually has model.heads.head (nn.Linear)
                if hasattr(model.heads, 'head') and isinstance(model.heads.head, nn.Linear):
                    in_feats = model.heads.head.in_features
                    model.heads.head = nn.Linear(in_feats, len(self.class_names))
                elif isinstance(model.heads, nn.Linear):
                    in_feats = model.heads.in_features
                    model.heads = nn.Linear(in_feats, len(self.class_names))
            elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
                in_feats = model.classifier.in_features
                model.classifier = nn.Linear(in_feats, len(self.class_names))
        else:
            raise ValueError(f"Model {self.args_model_name} is not supported!")
        
        # model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        # for p in model.parameters():
        #     p.requires_grad = True
        # in_feats = model.classifier[1].in_features
        # model.classifier[1] = nn.Linear(in_feats, len(self.class_names))
        # self.model = model.to(device)

        # fine-tune tất cả tham số
        for p in model.parameters():
            p.requires_grad = True

        self.model = model.to(device)

    def _compute_class_weights(self):
        labels = []
        for _, targets in self.train_loader:
            labels.extend(targets.cpu().numpy())
        weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(self.class_names)), y=labels)
        return torch.tensor(weights, dtype=torch.float32).to(device)

    def train(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._build_dataloaders()
        self._build_model()

        class_weights = self._compute_class_weights()
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

        best_val_acc, patience_left = 0.0, self.patience
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            running_loss, correct, total = 0.0, 0, 0
            for imgs, targets in tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False):
                imgs, targets = imgs.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    logits = self.model(imgs)
                    loss = criterion(logits, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
            tr_loss, tr_acc = running_loss / total, correct / total

            va_loss, va_acc = self.evaluate(self.val_loader, criterion)
            print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}")

            if va_acc > best_val_acc:
                best_val_acc, patience_left = va_acc, self.patience
                torch.save(self.model.state_dict(), self.output_dir / f'{self.args_model_name}_best_model.pt')
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print("Early stopping triggered.")
                    break

    def evaluate(self, loader, criterion):
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, targets in loader:
                imgs, targets = imgs.to(device), targets.to(device)
                logits = self.model(imgs)
                loss = criterion(logits, targets)
                running_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        return running_loss / total, correct / total
    
    def load_trained_model(self):
        if not self.class_names:
        # Đảm bảo đã có class_names từ dataloader
            self._build_dataloaders()
        if self.model is None:
            self._build_model()
        model_path = self.output_dir / f'{self.args_model_name}_best_model.pt'
        if not model_path.exists():
            raise FileNotFoundError(f"Không tìm thấy {model_path}, cần train trước khi test.")

        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

    def test(self):
        self._build_dataloaders()
        self.load_trained_model()  # đảm bảo model đã được khởi tạo + load checkpoint
        all_preds, all_targets = [], []
        with torch.no_grad():
            for imgs, targets in self.test_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                preds = self.model(imgs).argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        print(classification_report(all_targets, all_preds, target_names=self.class_names, digits=4))
        cm = confusion_matrix(all_targets, all_preds)
        model_metrics = self.evaluate_model_metrics(all_targets, all_preds)

        print(f"Accuracy: {model_metrics['accuracy']:.4f}")
        print(f"Precision: {model_metrics['precision']:.4f}")
        print(f"Recall: {model_metrics['recall']:.4f}")
        print(f"F1 Score: {model_metrics['f1_score']:.4f}")
        print(f"Specificity: {model_metrics['specificity']:.4f}")
        print(f"AUC Score: {model_metrics['auc']:.4f}")

        self._plot_confusion_matrix(cm)

    def _plot_confusion_matrix(self, cm):
        fig, ax = plt.subplots(figsize=(20, 15))
        im = ax.imshow(cm)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=200)
        plt.close()

    def evaluate_model_metrics(self, targets, predicts, average='macro'):
        acc = accuracy_score(targets, predicts)
        prec = precision_score(targets, predicts, average=average, zero_division=0)
        rec = recall_score(targets, predicts, average=average, zero_division=0)
        f1 = f1_score(targets, predicts, average=average, zero_division=0)
        spec = specificity_score(targets, predicts, average=average)

        # Lấy xác suất dự đoán cho toàn bộ test set
        all_probs = []
        for imgs, _ in self.test_loader:
            imgs = imgs.to(device)
            with torch.no_grad():
                batch_probs = torch.softmax(self.model(imgs), dim=1).cpu().numpy()
            all_probs.append(batch_probs)
        probs = np.vstack(all_probs)

        # Nếu là multiclass
        auc = roc_auc_score(targets, probs, multi_class='ovr', average=average)

        metrics = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "specificity": spec,
            "auc": auc,
        }
        return metrics

    # ----------- Helper Methods -----------
    def tensor_to_np_image(self, t: torch.Tensor) -> np.ndarray:
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        x = t.detach().cpu().numpy()
        x = (x * std + mean)
        x = np.clip(x, 0, 1)
        x = np.transpose(x, (1, 2, 0))
        return x

    def predict_proba_fn(self, imgs_np: np.ndarray) -> np.ndarray:
        self.model.eval()
        imgs_t = []
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        with torch.no_grad():
            for img in imgs_np:
                x = torch.from_numpy(np.transpose(img, (2, 0, 1))).float().unsqueeze(0)
                x = (x - mean) / std
                imgs_t.append(x)
            batch = torch.cat(imgs_t, dim=0).to(device)
            preds = torch.softmax(self.model(batch), dim=1).cpu().numpy()
        return preds
    
    def run_lime_metrics(self):
        """Tính metrics cho LIME"""
        evaluator = XAIEvaluator(self.model, self.class_names)
        # lấy samples giống run_lime
        samples = [
            (img, label.item())
            for imgs, labels in self.test_loader
            for img, label in zip(imgs, labels)
        ]

        results = evaluator.evaluate_with_lime(samples)
        print("📊 LIME metrics:", results)
        return results

    def run_shap_metrics(self, batch_size=8):
        """Tính metrics cho SHAP"""
        evaluator = XAIEvaluator(self.model, self.class_names)
        results = evaluator.evaluate_with_shap(self.test_loader, batch_size=batch_size)

        print("📊 SHAP metrics:", results)
        return results

    def run_pebex_metrics(self):
        """Tính metrics cho PEBEX"""
        evaluator = XAIEvaluator(self.model, self.class_names)
        
        results = evaluator.evaluate_with_pebex(self.test_loader)
        print("📊 PEBEX metrics:", results)
        return results

    def run_gradcam_metrics(self):
        """ Tính metrics cho GradCAM """
        evaluator = XAIEvaluator(self.model, self.class_names)
        results = evaluator.evaluate_with_GradCAM(self.test_loader)
        print("📊 GradCAM metrics:", results)
        return results