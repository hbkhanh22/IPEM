import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Reference: https://github.com/yiskw713/RISE/blob/master/rise.py

class RISE(nn.Module):
    def __init__(self, model, n_masks=1000, p=0.5, input_size=(224, 224), initial_mask_size=(7,7), n_batch=64, mask_path=None):
        super().__init__()
        self.model = model
        self.n_masks = n_masks
        self.p = p
        self.input_size = input_size
        self.initial_mask_size = initial_mask_size
        self.n_batch = n_batch
        if mask_path is None:
            self.masks = self._generate_masks()
        else:
            self.masks = self.load_masks(mask_path)

    def _generate_masks(self):
        # Cell size in the unsampled mask
        Ch = np.ceil(self.input_size[0] / self.initial_mask_size[0])
        Cw = np.ceil(self.input_size[1] / self.initial_mask_size[1])

        resize_h = int((self.initial_mask_size[0] + 1) * Ch)
        resize_w = int((self.initial_mask_size[1] + 1) * Cw)

        masks = []
        for _ in range(self.n_masks):
            # Generate a random binary mask
            binary_mask = torch.randn(1, 1, self.initial_mask_size[0], self.initial_mask_size[1])
            binary_mask = (binary_mask < self.p).float()            

            # Unsample mask (Sử dụng bilinear thay vì nearest để tránh bị pixelated ô vuông)
            mask = F.interpolate(binary_mask, (resize_h, resize_w), mode="bilinear", align_corners=False)

            # Random cropping
            i = np.random.randint(0, Ch)
            j = np.random.randint(0, Cw)

            mask = mask[:, :, i:i+self.input_size[0], j:j+self.input_size[1]]

            masks.append(mask)

        return torch.cat(masks, dim=0)

    def load_masks(self, filepath):
        masks = torch.load(filepath)
        return masks

    def save_masks(self, filepath):
        torch.save(self.masks, filepath)

    @torch.no_grad()
    def explain(self, input_img):
        # input_img size: (1, 3, H, W)
        device = next(self.model.parameters()).device
        self.masks = self.masks.to(device)
        input_img = input_img.to(device)

        masked_x = torch.mul(self.masks, input_img.data)

        # Chạy qua model theo từng batch (n_batch=64) để tránh tràn VRAM sang Shared RAM
        probs_list = []
        for i in range(0, self.n_masks, self.n_batch):
            input_batch = masked_x[i:i+self.n_batch]
            out = self.model(input_batch)
            probs_list.append(torch.softmax(out, dim=1).detach())
            
        probs = torch.cat(probs_list)
        
        n_classes = probs.shape[1]

        saliency_map = torch.matmul(probs.transpose(0,1), self.masks.view(self.n_masks, -1))
        saliency_map = saliency_map.view(n_classes, self.input_size[0], self.input_size[1])
        saliency_map /= (self.n_masks * self.p)

        # Normalize
        m, _ = torch.min(saliency_map.view(n_classes, -1), dim=1)
        saliency_map -= m.view(n_classes, 1, 1)
        M, _ = torch.max(saliency_map.view(n_classes, -1), dim=1)
        saliency_map /= M.view(n_classes, 1, 1)

        return saliency_map.cpu()