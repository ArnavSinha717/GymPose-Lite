"""COCO AP evaluation for GymPose-Lite.

Uses pycocotools to compute standard pose estimation metrics:
AP, AP50, AP75, AR.
"""

import os
import json
import argparse
import yaml
import numpy as np
import torch
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from data.coco_dataset import COCOKeypointsDataset
from models.gympose_lite import GymPoseLite


def load_config(path="configs/default.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate(model, dataset, coco_gt, device):
    """Run model on dataset and compute COCO AP metrics."""
    model.eval()
    results = []

    for idx in tqdm(range(len(dataset)), desc="Evaluating"):
        sample = dataset[idx]
        image = sample["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            _, coords = model(image)

        # coords are in [0,1], convert to pixel space
        coords = coords[0].cpu().numpy()  # (17, 2)
        coords[:, 0] *= dataset.input_w
        coords[:, 1] *= dataset.input_h

        # Get image info for this annotation
        ann = dataset.samples[idx]
        img_id = ann["image_id"]

        # Build COCO-format result
        kps = np.zeros(17 * 3)
        for k in range(17):
            # Map back to original image coordinates (approximate)
            bbox = ann["bbox"]
            pad_w, pad_h = bbox[2] * 0.2, bbox[3] * 0.2
            x1 = max(0, bbox[0] - pad_w)
            y1 = max(0, bbox[1] - pad_h)
            crop_w = bbox[2] + 2 * pad_w
            crop_h = bbox[3] + 2 * pad_h

            orig_x = x1 + coords[k, 0] * crop_w / dataset.input_w
            orig_y = y1 + coords[k, 1] * crop_h / dataset.input_h

            kps[k * 3] = orig_x
            kps[k * 3 + 1] = orig_y
            kps[k * 3 + 2] = 1  # visibility = visible

        results.append({
            "image_id": img_id,
            "category_id": 1,  # person
            "keypoints": kps.tolist(),
            "score": 1.0,
        })

    # Save results
    results_path = "/tmp/gympose_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f)

    # Run COCO evaluation
    coco_dt = coco_gt.loadRes(results_path)
    coco_eval = COCOeval(coco_gt, coco_dt, "keypoints")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metrics = {
        "AP": coco_eval.stats[0],
        "AP50": coco_eval.stats[1],
        "AP75": coco_eval.stats[2],
        "AR": coco_eval.stats[5],
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate GymPose-Lite")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", choices=["coco", "gym"], default="coco")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = GymPoseLite(
        num_keypoints=cfg["dataset"]["num_keypoints"],
        use_gnn=cfg["model"]["use_gnn"],
        gnn_hidden=cfg["model"]["gnn_hidden"],
        pretrained=False,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    # Load dataset
    ds_cfg = cfg["dataset"]
    if args.dataset == "coco":
        root = os.path.join(os.path.expanduser(ds_cfg["coco_root"]), "val2017")
        ann_file = os.path.join(os.path.expanduser(ds_cfg["coco_root"]),
                                "annotations", "person_keypoints_val2017.json")
    else:
        gym_root = os.path.expanduser(ds_cfg["gym_root"])
        root = os.path.join(gym_root, "valid")
        ann_file = os.path.join(gym_root, "annotations", "valid.json")

    dataset = COCOKeypointsDataset(
        root=root,
        ann_file=ann_file,
        input_size=tuple(ds_cfg["input_size"]),
        heatmap_size=tuple(ds_cfg["heatmap_size"]),
        sigma=ds_cfg["sigma"],
        augment=False,
    )

    coco_gt = COCO(ann_file)

    print(f"Evaluating on {len(dataset)} samples...")
    metrics = evaluate(model, dataset, coco_gt, device)

    print("\n--- Results ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()
