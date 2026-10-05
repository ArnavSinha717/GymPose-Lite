"""Dataset loader for knowledge distillation from teacher heatmaps.

Loads precomputed teacher heatmaps (from generate_teacher.py) along with
the cropped person images, optionally with classical CV preprocessing channels.
"""

import os
import glob
import math
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


class TeacherDataset(Dataset):

    def __init__(self, data_dir, use_cv_channels=True, augment=True, rotation=30,
                 scale_range=(0.75, 1.25)):
        self.data_dir = data_dir
        self.use_cv_channels = use_cv_channels
        self.augment = augment
        self.rotation = rotation
        self.scale_range = scale_range

        self.files = sorted(glob.glob(os.path.join(data_dir, "sample_*.npz")))
        print(f"TeacherDataset: {len(self.files)} samples from {data_dir}")

        self.flip_pairs = [
            (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)
        ]
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.files)

    def _compute_cv_channels(self, img_rgb):
        """Compute classical CV preprocessing channels.

        Returns 3 extra channels:
          - Canny edge map (normalized 0-1)
          - Skin mask (HSV-based)
          - Gradient magnitude (Sobel)
        """
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0

        # Skin detection via HSV thresholding
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin1 = cv2.inRange(hsv, lower_skin, upper_skin)
        lower_skin2 = np.array([170, 20, 70], dtype=np.uint8)
        upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
        skin2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        skin = ((skin1 | skin2) / 255.0).astype(np.float32)

        # Gradient magnitude (Sobel)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx ** 2 + gy ** 2)
        grad = grad / (grad.max() + 1e-8)

        return np.stack([edges, skin, grad], axis=-1)  # (H, W, 3)

    def _augment(self, img, heatmaps, coords, visibility):
        h, w = img.shape[:2]
        hm_h, hm_w = heatmaps.shape[1], heatmaps.shape[2]

        # Random horizontal flip
        if np.random.random() < 0.5:
            img = cv2.flip(img, 1)
            heatmaps = heatmaps[:, :, ::-1].copy()
            coords[:, 0] = 1.0 - coords[:, 0]
            for l, r in self.flip_pairs:
                heatmaps[[l, r]] = heatmaps[[r, l]]
                coords[[l, r]] = coords[[r, l]]
                visibility[[l, r]] = visibility[[r, l]]

        # Random rotation
        angle = np.random.uniform(-self.rotation, self.rotation)
        if abs(angle) > 1:
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            if img.ndim == 3:
                img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
            # Rotate heatmaps
            hm_center = (hm_w / 2, hm_h / 2)
            M_hm = cv2.getRotationMatrix2D(hm_center, angle, 1.0)
            for k in range(17):
                heatmaps[k] = cv2.warpAffine(heatmaps[k], M_hm, (hm_w, hm_h),
                                              flags=cv2.INTER_LINEAR)
            # Rotate coords
            cos_a = math.cos(math.radians(-angle))
            sin_a = math.sin(math.radians(-angle))
            for k in range(17):
                if visibility[k] > 0:
                    cx, cy = coords[k, 0] - 0.5, coords[k, 1] - 0.5
                    coords[k, 0] = cos_a * cx - sin_a * cy + 0.5
                    coords[k, 1] = sin_a * cx + cos_a * cy + 0.5

        return img, heatmaps, coords, visibility

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        img = data["image"]            # (256, 192, 3) RGB uint8
        heatmaps = data["heatmaps"]    # (17, 64, 48) float32
        coords = data["coords"]        # (17, 2) normalized [0,1]
        visibility = data["visibility"]  # (17,)

        # Augment (on RGB image + heatmaps together)
        if self.augment:
            img, heatmaps, coords, visibility = self._augment(
                img, heatmaps, coords, visibility)

        # Classical CV channels
        if self.use_cv_channels:
            cv_channels = self._compute_cv_channels(img)  # (256, 192, 3)

        # Normalize RGB
        img_norm = img.astype(np.float32) / 255.0
        img_norm = (img_norm - self.mean) / self.std  # (256, 192, 3)

        if self.use_cv_channels:
            # Stack: RGB(3) + CV(3) = 6 channels
            combined = np.concatenate([img_norm, cv_channels], axis=-1)  # (256, 192, 6)
            tensor = torch.from_numpy(combined.transpose(2, 0, 1)).float()  # (6, 256, 192)
        else:
            tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()  # (3, 256, 192)

        return {
            "image": tensor,
            "heatmaps": torch.from_numpy(heatmaps),
            "coords": torch.from_numpy(coords),
            "visibility": torch.from_numpy(visibility),
        }
