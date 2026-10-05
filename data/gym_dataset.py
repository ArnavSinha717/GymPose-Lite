"""YOLO Pose format dataset loader for the Kaggle Gym dataset.

The gym dataset uses YOLO pose format:
  class cx cy w h kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v ... (17 keypoints)
All coordinates are normalized [0, 1] relative to image size.

Directory structure:
  JIM_DATA29/
    Deadlifts/sumo data 29/{images,labels}/{train,val}/
    Squats/squats data29/{images,labels}/{train,val}/
    ...
"""

import os
import glob
import math
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


class GymPoseDataset(Dataset):

    def __init__(self, root, input_size=(256, 192), heatmap_size=(64, 48),
                 sigma=2, augment=True, rotation=30, scale_range=(0.75, 1.25),
                 split="train"):
        self.input_h, self.input_w = input_size
        self.heatmap_h, self.heatmap_w = heatmap_size
        self.sigma = sigma
        self.augment = augment
        self.rotation = rotation
        self.scale_range = scale_range
        self.num_keypoints = 17

        self.flip_pairs = [
            (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)
        ]
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Find all image/label pairs recursively
        self.samples = []
        for img_path in sorted(glob.glob(os.path.join(root, "**", "images", split, "*.jpg"),
                                         recursive=True)):
            # Corresponding label: .../images/split/name.jpg -> .../labels/split/name.txt
            label_path = img_path.replace("/images/", "/labels/").replace(".jpg", ".txt")
            if os.path.exists(label_path):
                self.samples.append((img_path, label_path))

        print(f"GymPoseDataset [{split}]: {len(self.samples)} samples from {root}")

    def __len__(self):
        return len(self.samples)

    def _parse_label(self, label_path, img_h, img_w):
        """Parse YOLO pose label file.

        Format: class cx cy w h kp1_x kp1_y kp1_v ... kp17_x kp17_y kp17_v
        Returns: bbox (x,y,w,h in pixels), keypoints (17,2), visibility (17,)
        """
        with open(label_path) as f:
            line = f.readline().strip()

        parts = list(map(float, line.split()))
        # parts[0] = class, parts[1:5] = cx cy w h (normalized)
        cx, cy, w, h = parts[1], parts[2], parts[3], parts[4]

        # Convert to pixel coordinates
        bbox_x = (cx - w / 2) * img_w
        bbox_y = (cy - h / 2) * img_h
        bbox_w = w * img_w
        bbox_h = h * img_h
        bbox = [bbox_x, bbox_y, bbox_w, bbox_h]

        # Parse 17 keypoints (x, y, visibility)
        keypoints = np.zeros((17, 2), dtype=np.float32)
        visibility = np.zeros(17, dtype=np.float32)

        kp_data = parts[5:]  # Should be 17 * 3 = 51 values
        for k in range(min(17, len(kp_data) // 3)):
            kx = kp_data[k * 3] * img_w
            ky = kp_data[k * 3 + 1] * img_h
            kv = kp_data[k * 3 + 2]
            keypoints[k] = [kx, ky]
            visibility[k] = kv

        return bbox, keypoints, visibility

    def _crop_person(self, img, bbox, keypoints):
        """Crop person bounding box with padding."""
        x, y, w, h = bbox
        pad_w, pad_h = w * 0.2, h * 0.2
        x1 = max(0, int(x - pad_w))
        y1 = max(0, int(y - pad_h))
        x2 = min(img.shape[1], int(x + w + pad_w))
        y2 = min(img.shape[0], int(y + h + pad_h))

        crop = img[y1:y2, x1:x2]
        kps = keypoints.copy()
        kps[:, 0] -= x1
        kps[:, 1] -= y1

        return crop, kps

    def _augment(self, img, keypoints, visibility):
        h, w = img.shape[:2]

        if np.random.random() < 0.5:
            img = cv2.flip(img, 1)
            keypoints[:, 0] = w - 1 - keypoints[:, 0]
            for l, r in self.flip_pairs:
                keypoints[[l, r]] = keypoints[[r, l]]
                visibility[[l, r]] = visibility[[r, l]]

        angle = np.random.uniform(-self.rotation, self.rotation)
        if abs(angle) > 1:
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
            for i in range(self.num_keypoints):
                if visibility[i] > 0:
                    x, y = keypoints[i]
                    cos_a = math.cos(math.radians(-angle))
                    sin_a = math.sin(math.radians(-angle))
                    nx = cos_a * (x - center[0]) - sin_a * (y - center[1]) + center[0]
                    ny = sin_a * (x - center[0]) + cos_a * (y - center[1]) + center[1]
                    keypoints[i] = [nx, ny]

        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        if abs(scale - 1.0) > 0.05:
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            keypoints *= scale

        return img, keypoints, visibility

    def _resize_to_input(self, img, keypoints):
        h, w = img.shape[:2]
        scale_x = self.input_w / max(w, 1)
        scale_y = self.input_h / max(h, 1)

        img = cv2.resize(img, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        keypoints[:, 0] *= scale_x
        keypoints[:, 1] *= scale_y

        return img, keypoints

    def _generate_heatmaps(self, keypoints, visibility):
        heatmaps = np.zeros((self.num_keypoints, self.heatmap_h, self.heatmap_w),
                            dtype=np.float32)

        scale_x = self.heatmap_w / self.input_w
        scale_y = self.heatmap_h / self.input_h

        for k in range(self.num_keypoints):
            if visibility[k] == 0:
                continue
            cx = keypoints[k, 0] * scale_x
            cy = keypoints[k, 1] * scale_y

            x = np.arange(0, self.heatmap_w, 1, np.float32)
            y = np.arange(0, self.heatmap_h, 1, np.float32)
            yy, xx = np.meshgrid(y, x, indexing='ij')

            heatmaps[k] = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * self.sigma ** 2))

        return heatmaps

    def _normalize(self, img):
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        return img.transpose(2, 0, 1)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not load {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_h, img_w = img.shape[:2]
        bbox, keypoints, visibility = self._parse_label(label_path, img_h, img_w)

        img, keypoints = self._crop_person(img, bbox, keypoints)

        if self.augment:
            img, keypoints, visibility = self._augment(img, keypoints, visibility)

        img, keypoints = self._resize_to_input(img, keypoints)
        heatmaps = self._generate_heatmaps(keypoints, visibility)
        img = self._normalize(img)

        coords_normalized = keypoints.copy()
        coords_normalized[:, 0] /= self.input_w
        coords_normalized[:, 1] /= self.input_h

        return {
            "image": torch.from_numpy(img),
            "heatmaps": torch.from_numpy(heatmaps),
            "coords": torch.from_numpy(coords_normalized),
            "visibility": torch.from_numpy(visibility),
            "keypoints_px": torch.from_numpy(keypoints),
        }
