"""Quick distillation training test.

Trains our lightweight model on teacher-generated heatmaps
with optional classical CV input channels.
"""

import os
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data.teacher_dataset import TeacherDataset
from models.gympose_lite import GymPoseLite
from models.gnn_refine import bone_length_loss


def load_config(path="configs/default.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    mse = nn.MSELoss()
    l1 = nn.L1Loss()

    for batch in tqdm(loader, desc="Training"):
        images = batch["image"].to(device)
        gt_heatmaps = batch["heatmaps"].to(device)
        gt_coords = batch["coords"].to(device)
        vis = batch["visibility"].to(device)

        optimizer.zero_grad()
        pred_hm, pred_coords = model(images)

        loss_hm = mse(pred_hm, gt_heatmaps)
        vis_mask = (vis > 0).unsqueeze(-1).float()
        loss_coord = l1(pred_coords * vis_mask, gt_coords * vis_mask)

        loss = loss_hm + 5.0 * loss_coord  # Boosted coord weight
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_errors = []

    for batch in loader:
        images = batch["image"].to(device)
        gt_coords = batch["coords"].numpy()
        vis = batch["visibility"].numpy()

        _, pred_coords = model(images)
        pred_coords = pred_coords.cpu().numpy()

        for b in range(pred_coords.shape[0]):
            for k in range(17):
                if vis[b, k] > 0:
                    px = pred_coords[b, k, 0] * 192
                    py = pred_coords[b, k, 1] * 256
                    gx = gt_coords[b, k, 0] * 192
                    gy = gt_coords[b, k, 1] * 256
                    err = np.sqrt((px - gx) ** 2 + (py - gy) ** 2)
                    all_errors.append(err)

    errors = np.array(all_errors)
    return {
        "mean_px_error": errors.mean(),
        "median_px_error": np.median(errors),
        "pct_under_10px": (errors < 10).mean() * 100,
        "pct_under_20px": (errors < 20).mean() * 100,
    }


def check_heatmap_quality(model, loader, device):
    model.eval()
    peaks = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            pred_hm, _ = model(images)
            for b in range(pred_hm.shape[0]):
                peaks.append(pred_hm[b].cpu().numpy().max())
            if len(peaks) > 50:
                break

    return np.mean(peaks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/teacher_heatmaps")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--use-cv", action="store_true", help="Use classical CV channels (6ch input)")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--resume", default="", help="Resume from checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    in_channels = 6 if args.use_cv else 3

    # Dataset
    dataset = TeacherDataset(args.data, use_cv_channels=args.use_cv, augment=True)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # Disable augment for val
    val_ds_no_aug = TeacherDataset(args.data, use_cv_channels=args.use_cv, augment=False)
    val_indices = val_ds.indices
    val_ds_eval = torch.utils.data.Subset(val_ds_no_aug, val_indices)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds_eval, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = GymPoseLite(
        num_keypoints=17, use_gnn=True, gnn_hidden=64,
        pretrained=True, in_channels=in_channels,
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {total:,} | Input channels: {in_channels}")

    # Resume from checkpoint if provided
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Resumed from: {args.resume}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Check baseline (before training)
    print("\n--- Before training ---")
    metrics = evaluate(model, val_loader, device)
    peak = check_heatmap_quality(model, val_loader, device)
    print(f"  Heatmap peak: {peak:.3f}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.1f}")

    # Train
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        loss = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        print(f"  Loss: {loss:.4f}")

        metrics = evaluate(model, val_loader, device)
        peak = check_heatmap_quality(model, val_loader, device)
        print(f"  Heatmap peak: {peak:.3f}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.1f}")

    # Save
    save_path = "checkpoints/distill_test.pth"
    torch.save({"model": model.state_dict()}, save_path)
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
