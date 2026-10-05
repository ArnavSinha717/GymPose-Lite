"""Generate teacher heatmaps using pretrained KeypointRCNN.

Runs the heavy pretrained model on COCO images and saves the predicted
heatmaps as training targets for knowledge distillation.
"""

import os
import argparse
import numpy as np
import cv2
import torch
import torchvision
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
from tqdm import tqdm
from pycocotools.coco import COCO


def generate_teacher_heatmaps(coco_root, output_dir, max_samples=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading teacher model (KeypointRCNN ResNet50)...")
    teacher = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1).to(device)
    teacher.eval()

    ann_file = os.path.join(coco_root, "annotations", "person_keypoints_train2017.json")
    img_root = os.path.join(coco_root, "train2017")
    coco = COCO(ann_file)

    samples = []
    for ann_id in coco.getAnnIds():
        ann = coco.anns[ann_id]
        if ann.get("num_keypoints", 0) > 0 and ann.get("bbox"):
            samples.append(ann)

    if max_samples > 0:
        samples = samples[:max_samples]

    print(f"Processing {len(samples)} samples...")
    os.makedirs(output_dir, exist_ok=True)

    successful = 0
    for idx, ann in enumerate(tqdm(samples, desc="Generating teacher heatmaps")):
        img_info = coco.loadImgs(ann["image_id"])[0]
        img_path = os.path.join(img_root, img_info["file_name"])

        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Crop person with padding
        x, y, w, h = ann["bbox"]
        pad_w, pad_h = w * 0.2, h * 0.2
        x1 = max(0, int(x - pad_w))
        y1 = max(0, int(y - pad_h))
        x2 = min(img.shape[1], int(x + w + pad_w))
        y2 = min(img.shape[0], int(y + h + pad_h))

        crop = img_rgb[y1:y2, x1:x2]
        if crop.shape[0] < 10 or crop.shape[1] < 10:
            continue

        resized = cv2.resize(crop, (192, 256), interpolation=cv2.INTER_LINEAR)

        # Run teacher
        img_tensor = torchvision.transforms.functional.to_tensor(resized).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = teacher(img_tensor)

        if len(outputs[0]["keypoints"]) == 0:
            continue

        best_idx = outputs[0]["scores"].argmax()
        teacher_keypoints = outputs[0]["keypoints"][best_idx].cpu().numpy()  # (17, 3)

        # Generate heatmaps from teacher predictions
        heatmaps = np.zeros((17, 64, 48), dtype=np.float32)
        keypoints_norm = np.zeros((17, 2), dtype=np.float32)
        visibility = np.zeros(17, dtype=np.float32)

        for k in range(17):
            kx, ky, conf = teacher_keypoints[k]
            if conf < 0.5:
                continue

            nx = kx / 192.0
            ny = ky / 256.0
            keypoints_norm[k] = [nx, ny]
            visibility[k] = 1.0

            cx = nx * 48
            cy = ny * 64
            sigma = 3.0

            x_grid = np.arange(0, 48, 1, np.float32)
            y_grid = np.arange(0, 64, 1, np.float32)
            yy, xx = np.meshgrid(y_grid, x_grid, indexing='ij')
            heatmaps[k] = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))

        if visibility.sum() < 5:
            continue

        np.savez_compressed(os.path.join(output_dir, f"sample_{idx:06d}.npz"),
                            image=resized, heatmaps=heatmaps,
                            coords=keypoints_norm, visibility=visibility)
        successful += 1

    print(f"Done. {successful} teacher samples saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-root", default="datasets/coco")
    parser.add_argument("--output", default="datasets/teacher_heatmaps")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()
    generate_teacher_heatmaps(os.path.expanduser(args.coco_root), args.output, args.max_samples)
