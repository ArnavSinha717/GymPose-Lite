"""COCO Keypoints dataset loader.

Works for both COCO 2017 and Kaggle Gym dataset (same annotation format).
Each sample: crop person bounding box -> resize to 256x192 -> augment ->
generate 17 Gaussian heatmaps at 64x48 as supervision targets.
"""

import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import cv2


class COCOKeypointsDataset(Dataset):

    def __init__(self, root, ann_file, input_size=(256, 192), heatmap_size=(64, 48),
                 sigma=2, augment=True, rotation=30, scale_range=(0.75, 1.25)):
        """
        Args:
            root: Path to image directory (e.g., train2017/)
            ann_file: Path to annotation JSON
            input_size: (H, W) model input resolution
            heatmap_size: (H, W) heatmap resolution (1/4 of input)
            sigma: Gaussian kernel sigma for heatmap generation
            augment: Whether to apply geometric augmentations
            rotation: Max rotation in degrees (+/-)
            scale_range: (min_scale, max_scale) for scale jitter
        """
        self.root = root
        self.input_h, self.input_w = input_size
        self.heatmap_h, self.heatmap_w = heatmap_size
        self.sigma = sigma
        self.augment = augment
        self.rotation = rotation
        self.scale_range = scale_range
        self.num_keypoints = 17

        # COCO flip pairs for horizontal augmentation
        self.flip_pairs = [
            (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)
        ]

        # ImageNet normalization (since backbone is pretrained on ImageNet)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Load COCO annotations
        self.coco = COCO(ann_file)

        # Filter: keep only images with person keypoints annotated
        self.samples = []
        for ann_id in self.coco.getAnnIds():
            ann = self.coco.anns[ann_id]
            # Must have keypoints and a valid bbox
            if ann.get("num_keypoints", 0) > 0 and ann.get("bbox"):
                self.samples.append(ann)

    def __len__(self):
        return len(self.samples)

    def _load_image(self, img_id):
        img_info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.root, img_info["file_name"])
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not load {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _crop_person(self, img, bbox, keypoints):
        """Crop person bounding box with padding, adjust keypoints."""
        x, y, w, h = bbox
        # Add 20% padding
        pad_w, pad_h = w * 0.2, h * 0.2
        x1 = max(0, int(x - pad_w))
        y1 = max(0, int(y - pad_h))
        x2 = min(img.shape[1], int(x + w + pad_w))
        y2 = min(img.shape[0], int(y + h + pad_h))

        crop = img[y1:y2, x1:x2]

        # Adjust keypoints relative to crop
        kps = keypoints.copy()
        kps[:, 0] -= x1
        kps[:, 1] -= y1

        return crop, kps, (x1, y1, x2 - x1, y2 - y1)

    def _augment(self, img, keypoints, visibility):
        """Apply geometric image transforms: flip, rotation, scale."""
        h, w = img.shape[:2]

        # Random horizontal flip
        if np.random.random() < 0.5:
            img = cv2.flip(img, 1)
            keypoints[:, 0] = w - 1 - keypoints[:, 0]
            # Swap left/right keypoints
            for l, r in self.flip_pairs:
                keypoints[[l, r]] = keypoints[[r, l]]
                visibility[[l, r]] = visibility[[r, l]]

        # Random rotation (affine transform with bilinear interpolation)
        angle = np.random.uniform(-self.rotation, self.rotation)
        if abs(angle) > 1:
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
            # Rotate keypoints
            for i in range(self.num_keypoints):
                if visibility[i] > 0:
                    x, y = keypoints[i]
                    cos_a = math.cos(math.radians(-angle))
                    sin_a = math.sin(math.radians(-angle))
                    nx = cos_a * (x - center[0]) - sin_a * (y - center[1]) + center[0]
                    ny = sin_a * (x - center[0]) + cos_a * (y - center[1]) + center[1]
                    keypoints[i] = [nx, ny]

        # Random scale jitter
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        if abs(scale - 1.0) > 0.05:
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            keypoints *= scale

        return img, keypoints, visibility

    def _resize_to_input(self, img, keypoints):
        """Resize to model input size, adjust keypoints."""
        h, w = img.shape[:2]
        scale_x = self.input_w / w
        scale_y = self.input_h / h

        img = cv2.resize(img, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        keypoints[:, 0] *= scale_x
        keypoints[:, 1] *= scale_y

        return img, keypoints

    def _generate_heatmaps(self, keypoints, visibility):
        """Generate 17 target heatmaps with 2D Gaussian kernels.

        For each visible keypoint, place a Gaussian blob (sigma=2)
        centered at the keypoint location on a 64x48 map.
        This is a core image processing operation.
        """
        heatmaps = np.zeros((self.num_keypoints, self.heatmap_h, self.heatmap_w),
                            dtype=np.float32)

        # Scale keypoints from input space to heatmap space
        scale_x = self.heatmap_w / self.input_w
        scale_y = self.heatmap_h / self.input_h

        for k in range(self.num_keypoints):
            if visibility[k] == 0:
                continue

            cx = keypoints[k, 0] * scale_x
            cy = keypoints[k, 1] * scale_y

            # Generate 2D Gaussian kernel centered at (cx, cy)
            size = 6 * self.sigma + 1
            x = np.arange(0, self.heatmap_w, 1, np.float32)
            y = np.arange(0, self.heatmap_h, 1, np.float32)
            yy, xx = np.meshgrid(y, x, indexing='ij')

            heatmaps[k] = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * self.sigma ** 2))

        return heatmaps

    def _normalize(self, img):
        """ImageNet normalization."""
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        # HWC -> CHW
        img = img.transpose(2, 0, 1)
        return img

    def __getitem__(self, idx):
        ann = self.samples[idx]

        # Load image
        img = self._load_image(ann["image_id"])

        # Parse keypoints: [x1,y1,v1, x2,y2,v2, ...] -> (17,2) and (17,)
        kps_raw = np.array(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
        keypoints = kps_raw[:, :2].copy()       # (17, 2) x,y
        visibility = kps_raw[:, 2].copy()        # (17,) 0=missing, 1=occluded, 2=visible

        # Crop person bounding box
        img, keypoints, _ = self._crop_person(img, ann["bbox"], keypoints)

        # Augmentations (geometric image transforms)
        if self.augment:
            img, keypoints, visibility = self._augment(img, keypoints, visibility)

        # Resize to model input
        img, keypoints = self._resize_to_input(img, keypoints)

        # Generate target heatmaps (2D Gaussian kernels)
        heatmaps = self._generate_heatmaps(keypoints, visibility)

        # Normalize image (ImageNet stats)
        img = self._normalize(img)

        # Normalize keypoint coords to [0, 1] for coordinate loss
        coords_normalized = keypoints.copy()
        coords_normalized[:, 0] /= self.input_w
        coords_normalized[:, 1] /= self.input_h

        return {
            "image": torch.from_numpy(img),
            "heatmaps": torch.from_numpy(heatmaps),
            "coords": torch.from_numpy(coords_normalized),
            "visibility": torch.from_numpy(visibility),
            "keypoints_px": torch.from_numpy(keypoints),  # In pixel space for evaluation
        }
