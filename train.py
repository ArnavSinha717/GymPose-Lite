"""Training script for GymPose-Lite.

Two-stage training:
  Stage 1: Pretrain on COCO Keypoints (~50 epochs)
  Stage 2: Fine-tune on Kaggle Gym COCO dataset (~20 epochs, lower LR)
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data.coco_dataset import COCOKeypointsDataset
from data.gym_dataset import GymPoseDataset
from models.gympose_lite import GymPoseLite
from models.gnn_refine import bone_length_loss


def load_config(path="configs/default.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def create_dataloader(cfg, split="train"):
    ds_cfg = cfg["dataset"]
    aug_cfg = cfg["augmentation"]

    if split == "train":
        root = os.path.join(os.path.expanduser(ds_cfg["coco_root"]), "train2017")
        ann = os.path.join(os.path.expanduser(ds_cfg["coco_root"]),
                           "annotations", "person_keypoints_train2017.json")
        augment = True
    else:
        root = os.path.join(os.path.expanduser(ds_cfg["coco_root"]), "val2017")
        ann = os.path.join(os.path.expanduser(ds_cfg["coco_root"]),
                           "annotations", "person_keypoints_val2017.json")
        augment = False

    dataset = COCOKeypointsDataset(
        root=root,
        ann_file=ann,
        input_size=tuple(ds_cfg["input_size"]),
        heatmap_size=tuple(ds_cfg["heatmap_size"]),
        sigma=ds_cfg["sigma"],
        augment=augment,
        rotation=aug_cfg["rotation"],
        scale_range=(aug_cfg["scale_min"], aug_cfg["scale_max"]),
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=(split == "train"),
        num_workers=4,
        pin_memory=True,
        drop_last=(split == "train"),
    )
    return loader


def create_gym_dataloader(cfg, split="train"):
    """Create dataloader for the Kaggle Gym dataset (YOLO pose format)."""
    ds_cfg = cfg["dataset"]
    aug_cfg = cfg["augmentation"]
    gym_root = os.path.join(os.path.expanduser(ds_cfg["gym_root"]), "JIM_DATA29")

    dataset = GymPoseDataset(
        root=gym_root,
        input_size=tuple(ds_cfg["input_size"]),
        heatmap_size=tuple(ds_cfg["heatmap_size"]),
        sigma=ds_cfg["sigma"],
        augment=(split == "train"),
        rotation=aug_cfg["rotation"],
        scale_range=(aug_cfg["scale_min"], aug_cfg["scale_max"]),
        split=split,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=(split == "train"),
        num_workers=4,
        pin_memory=True,
        drop_last=(split == "train"),
    )
    return loader


def train_one_epoch(model, loader, optimizer, device, cfg):
    model.train()
    total_loss = 0
    hm_weight = cfg["train"]["heatmap_loss_weight"]
    coord_weight = cfg["train"]["coord_loss_weight"]
    bone_weight = cfg["train"]["bone_loss_weight"]

    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        images = batch["image"].to(device)
        gt_heatmaps = batch["heatmaps"].to(device)
        gt_coords = batch["coords"].to(device)
        visibility = batch["visibility"].to(device)

        optimizer.zero_grad()

        pred_heatmaps, pred_coords = model(images)

        # Heatmap loss (MSE on spatial probability maps)
        loss_hm = mse_loss(pred_heatmaps, gt_heatmaps)

        # Coordinate loss (L1, only on visible keypoints)
        vis_mask = (visibility > 0).unsqueeze(-1).float()  # (B, 17, 1)
        loss_coord = l1_loss(pred_coords * vis_mask, gt_coords * vis_mask)

        # Bone length consistency loss
        loss_bone = bone_length_loss(pred_coords, gt_coords) if cfg["model"]["use_gnn"] else 0

        loss = hm_weight * loss_hm + coord_weight * loss_coord + bone_weight * loss_bone
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", hm=f"{loss_hm.item():.4f}",
                         coord=f"{loss_coord.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0
    mse_loss = nn.MSELoss()

    for batch in tqdm(loader, desc="Validating"):
        images = batch["image"].to(device)
        gt_heatmaps = batch["heatmaps"].to(device)

        pred_heatmaps, _ = model(images)
        loss = mse_loss(pred_heatmaps, gt_heatmaps)
        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser(description="Train GymPose-Lite")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--stage", choices=["pretrain", "finetune"], default="pretrain")
    parser.add_argument("--checkpoint", default="", help="Resume from checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    model = GymPoseLite(
        num_keypoints=cfg["dataset"]["num_keypoints"],
        use_gnn=cfg["model"]["use_gnn"],
        gnn_hidden=cfg["model"]["gnn_hidden"],
        pretrained=True,
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # Setup training
    if args.stage == "pretrain":
        epochs = cfg["train"]["epochs"]
        lr = cfg["train"]["lr"]
        train_loader = create_dataloader(cfg, "train")
        val_loader = create_dataloader(cfg, "val")
        save_prefix = "coco"
    else:
        epochs = cfg["finetune"]["epochs"]
        lr = cfg["finetune"]["lr"]
        train_loader = create_gym_dataloader(cfg, "train")
        val_loader = create_gym_dataloader(cfg, "val")
        save_prefix = "gym"

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    save_dir = os.path.expanduser(cfg["train"]["save_dir"])
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float("inf")
    start_epoch = 1

    # Resume from checkpoint (loads model, optimizer, scheduler, epoch)
    resume_path = args.checkpoint
    if not resume_path:
        # Auto-resume: check if latest checkpoint exists
        auto_path = os.path.join(save_dir, f"{save_prefix}_latest.pth")
        if os.path.exists(auto_path):
            resume_path = auto_path
            print(f"Auto-resuming from: {resume_path}")

    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            if args.stage == "pretrain":
                # Only restore training state when resuming same stage
                if "optimizer" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer"])
                if "scheduler" in ckpt:
                    scheduler.load_state_dict(ckpt["scheduler"])
                if "epoch" in ckpt:
                    start_epoch = ckpt["epoch"] + 1
                if "best_val_loss" in ckpt:
                    best_val_loss = ckpt["best_val_loss"]
                print(f"Resumed from epoch {ckpt.get('epoch', '?')}, best_val_loss={best_val_loss:.4f}")
            else:
                # Finetune: only load model weights, fresh optimizer/scheduler/epoch
                print(f"Loaded pretrained weights from: {resume_path} (fresh optimizer for fine-tuning)")
        else:
            # Legacy checkpoint (model state_dict only)
            model.load_state_dict(ckpt)
            print(f"Loaded model weights from: {resume_path} (no optimizer/epoch state)")

    unfreeze_epoch = cfg["train"].get("unfreeze_epoch", 10)
    unfrozen = start_epoch > unfreeze_epoch  # Already past unfreeze point if resuming

    for epoch in range(start_epoch, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")

        # Unfreeze backbone after warmup
        if not unfrozen and epoch >= unfreeze_epoch:
            model.backbone.unfreeze_all()
            # Rebuild optimizer to include newly unfrozen params
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr * 0.1,  # Lower LR for pretrained layers
                weight_decay=cfg["train"]["weight_decay"],
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs - epoch)
            unfrozen = True
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Unfroze backbone at epoch {epoch}. Trainable params: {trainable:,}")

        train_loss = train_one_epoch(model, train_loader, optimizer, device, cfg)
        print(f"Train loss: {train_loss:.4f}")

        scheduler.step()

        if epoch % cfg["train"]["val_interval"] == 0 or epoch == epochs:
            val_loss = validate(model, val_loader, device)
            print(f"Val loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                path = os.path.join(save_dir, f"{save_prefix}_best.pth")
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                }, path)
                print(f"Saved best model: {path}")

        # Save latest (always includes full state for resuming)
        path = os.path.join(save_dir, f"{save_prefix}_latest.pth")
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
        }, path)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
