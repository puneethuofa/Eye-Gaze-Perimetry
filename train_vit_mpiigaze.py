#!/usr/bin/env python3
"""
Improved train_vit_mpiigaze.py

- ViT with img_size fix for non-224 inputs (e.g., 96x96)
- Train in radians (targets converted from degrees)
- Log per-axis MAE (deg) and 2D angular error (deg)
- ImageNet normalization (matching ViT pretraining)
- Mild data augmentation (no flips to avoid sign issues)
- Cosine LR schedule + gradient clipping
- Separate LR for backbone vs head (backbone_lr_mult)
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import timm

DEG2RAD = 3.141592653589793 / 180.0
RAD2DEG = 180.0 / 3.141592653589793


# ---------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="ViT gaze regression on MPIIGaze-style data (img_size fix + metrics)."
    )
    parser.add_argument("--manifest", type=str, required=True,
                        help="CSV with img/yaw_deg/pitch_deg or similar columns.")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Output directory for logs/checkpoints.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument("--model", type=str, default="vit_tiny_patch16_224")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--device", type=str, default=None,
                        help="cuda, cpu, or leave empty to auto-detect.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Gradient clipping max-norm (0 to disable).")
    parser.add_argument(
        "--backbone-lr-mult",
        type=float,
        default=0.1,
        help="Backbone LR is lr * backbone_lr_mult (head uses lr).",
    )

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[warn] Ignoring unknown CLI args: {unknown}")

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    args.run_dir = str(Path(args.run_dir).expanduser().resolve())
    return args


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
class GazeDataset(Dataset):
    """
    CSV-based gaze dataset for MPIIGaze-style manifests.

    Expected columns (flexible):
      - img / path / image / ...     : image path
      - yaw_deg / yaw / gaze_yaw ... : horizontal gaze (deg)
      - pitch_deg / pitch / ...      : vertical gaze (deg)
    """

    def __init__(self, manifest_path: str, transform=None):
        self.manifest_path = Path(manifest_path)
        self.df = pd.read_csv(self.manifest_path)

        if len(self.df) == 0:
            raise RuntimeError(f"Manifest {manifest_path} is empty.")

        colmap = {c.lower(): c for c in self.df.columns}

        def resolve(options):
            for opt in options:
                if opt in colmap:
                    return colmap[opt]
            raise KeyError(f"None of {options} found in manifest columns {list(self.df.columns)}")

        self.path_col = resolve(["img", "path", "img_path", "image", "fname", "file"])
        self.yaw_col = resolve(["yaw_deg", "yaw", "gaze_yaw", "gaze_y"])
        self.pitch_col = resolve(["pitch_deg", "pitch", "gaze_pitch", "gaze_x"])

        self.transform = transform
        self.root = self.manifest_path.parent

        print(f"[dataset] Using columns: path={self.path_col}, yaw={self.yaw_col}, pitch={self.pitch_col}")
        print(f"[dataset] Num samples: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(str(row[self.path_col]))
        if not img_path.is_absolute():
            img_path = self.root / img_path

        from PIL import Image
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        yaw_deg = float(row[self.yaw_col])
        pitch_deg = float(row[self.pitch_col])
        target_deg = torch.tensor([yaw_deg, pitch_deg], dtype=torch.float32)

        return img, target_deg


# ---------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------
def build_transforms(args):
    # ImageNet normalization (what ViT was pretrained with)
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    train_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        # Mild augmentations that don't change semantic gaze direction
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.RandomAffine(
            degrees=5, translate=(0.02, 0.02), scale=(0.95, 1.05)
        ),
        transforms.ToTensor(),
        normalize,
    ])

    val_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        normalize,
    ])

    return train_tf, val_tf


# ---------------------------------------------------------------------
# Model (with ViT img_size fix)
# ---------------------------------------------------------------------
def build_model(args):
    extra_kwargs = {}
    # Fix: pass img_size for ViT models so 96x96 etc. work
    if "vit" in args.model.lower():
        extra_kwargs["img_size"] = args.img_size

    model = timm.create_model(
        args.model,
        pretrained=True,
        num_classes=2,  # (yaw, pitch)
        **extra_kwargs,
    )

    # Safety: make sure final head has 2 outputs
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        if model.head.out_features != 2:
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, 2)

    return model


# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------
def yaw_pitch_to_vec(yaw_pitch_rad: torch.Tensor) -> torch.Tensor:
    """
    Convert (yaw, pitch) in radians to 3D unit vector.
    yaw_pitch_rad: [N, 2]
    """
    yaw = yaw_pitch_rad[:, 0]
    pitch = yaw_pitch_rad[:, 1]

    x = torch.cos(pitch) * torch.sin(yaw)
    y = torch.sin(pitch)
    z = torch.cos(pitch) * torch.cos(yaw)

    vec = torch.stack([x, y, z], dim=1)
    vec = vec / (vec.norm(dim=1, keepdim=True) + 1e-8)  # normalize
    return vec


# ---------------------------------------------------------------------
# Optimizer + scheduler (backbone vs head LR)
# ---------------------------------------------------------------------
def build_optimizer_and_scheduler(model, args):
    """
    Use different LR for backbone vs head:
    - head: lr = args.lr
    - backbone: lr = args.lr * args.backbone_lr_mult
    """
    head_params = []
    backbone_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # crude but effective: anything with "head" in its name -> head group
        if "head" in name:
            head_params.append(p)
        else:
            backbone_params.append(p)

    param_groups = [
        {"params": backbone_params, "lr": args.lr * args.backbone_lr_mult},
        {"params": head_params, "lr": args.lr},
    ]

    optim = torch.optim.AdamW(
        param_groups,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs
    )

    return optim, scheduler


# ---------------------------------------------------------------------
# Training / eval loops
# ---------------------------------------------------------------------
def train_one_epoch(model, loader, crit, optim, device, max_grad_norm=0.0):
    model.train()
    running_loss = 0.0
    count = 0

    for xb, yb_deg in loader:
        xb = xb.to(device, non_blocking=True)
        yb_deg = yb_deg.to(device, non_blocking=True)
        yb_rad = yb_deg * DEG2RAD

        optim.zero_grad(set_to_none=True)
        out_rad = model(xb)
        loss = crit(out_rad, yb_rad)
        loss.backward()

        if max_grad_norm and max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optim.step()

        bs = xb.size(0)
        running_loss += loss.item() * bs
        count += bs

    return running_loss / max(count, 1)


@torch.no_grad()
def eval_one_epoch(model, loader, crit, device):
    model.eval()
    sum_loss = 0.0
    sum_yaw_mae = 0.0
    sum_pitch_mae = 0.0
    sum_ang_deg = 0.0
    count = 0

    for xb, yb_deg in loader:
        xb = xb.to(device, non_blocking=True)
        yb_deg = yb_deg.to(device, non_blocking=True)
        yb_rad = yb_deg * DEG2RAD

        out_rad = model(xb)
        loss = crit(out_rad, yb_rad)

        # Convert predictions to degrees for MAE
        out_deg = out_rad * RAD2DEG
        err_deg = torch.abs(out_deg - yb_deg)
        sum_yaw_mae += err_deg[:, 0].sum().item()
        sum_pitch_mae += err_deg[:, 1].sum().item()

        # 2D angular error via 3D unit vectors
        pred_vec = yaw_pitch_to_vec(out_rad)
        gt_vec = yaw_pitch_to_vec(yb_rad)
        cos_sim = torch.clamp(torch.sum(pred_vec * gt_vec, dim=1), -1.0, 1.0)
        ang_deg = torch.acos(cos_sim) * RAD2DEG
        sum_ang_deg += ang_deg.sum().item()

        bs = xb.size(0)
        sum_loss += loss.item() * bs
        count += bs

    if count == 0:
        return {
            "loss": 0.0,
            "yaw_mae_deg": 0.0,
            "pitch_mae_deg": 0.0,
            "ang_err_deg": 0.0,
        }

    return {
        "loss": sum_loss / count,
        "yaw_mae_deg": sum_yaw_mae / count,
        "pitch_mae_deg": sum_pitch_mae / count,
        "ang_err_deg": sum_ang_deg / count,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = parse_args()

    os.makedirs(args.run_dir, exist_ok=True)
    with open(Path(args.run_dir) / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"[info] Using device     : {args.device}")
    print(f"[info] Manifest         : {args.manifest}")
    print(f"[info] Run dir          : {args.run_dir}")
    print(f"[info] Model            : {args.model}")
    print(f"[info] Img size         : {args.img_size}")
    print(f"[info] Epochs           : {args.epochs}")
    print(f"[info] max_grad_norm    : {args.max_grad_norm}")
    print(f"[info] backbone_lr_mult : {args.backbone_lr_mult}")

    train_tf, val_tf = build_transforms(args)
    full_ds = GazeDataset(args.manifest, transform=train_tf)

    val_len = int(len(full_ds) * args.val_split)
    train_len = len(full_ds) - val_len
    train_ds, val_ds = random_split(full_ds, [train_len, val_len])

    # Validation uses val transforms (no aug)
    val_ds.dataset.transform = val_tf

    print(f"[split] train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device(args.device)
    model = build_model(args).to(device)
    crit = nn.MSELoss()
    optim, scheduler = build_optimizer_and_scheduler(model, args)

    best_val_loss = float("inf")
    ckpt_path = Path(args.run_dir) / "best.ckpt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, crit, optim, device,
            max_grad_norm=args.max_grad_norm,
        )
        val_stats = eval_one_epoch(model, val_loader, crit, device)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_loss:.6f}  "
            f"val_loss={val_stats['loss']:.6f}  "
            f"val_yaw_mae={val_stats['yaw_mae_deg']:.3f}°  "
            f"val_pitch_mae={val_stats['pitch_mae_deg']:.3f}°  "
            f"val_ang_err={val_stats['ang_err_deg']:.3f}°  "
            f"lr={current_lr:.2e}"
        )

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optim_state": optim.state_dict(),
                    "epoch": epoch,
                    "val_stats": val_stats,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"[info] New best val_loss={best_val_loss:.6f}, saved to {ckpt_path}")

    print("[done] Training complete.")


if __name__ == "__main__":
    main()
