"""Self-supervised scene classification

Usage:
    python train.py --data_dir data
"""
from __future__ import annotations

import os
import argparse
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
import torchvision.transforms.v2 as v2
from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image

from models import resnet50, SimCLR, Classifier

SEED = 42
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def set_random_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_transforms(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([
            transforms.GaussianBlur(23, (0.1, 2.0))
        ], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ])


def test_transforms(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(img_size / 0.875)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ])


class SimCLRDataset(Dataset):
    """Two independently augmented views per image (labels ignored)."""

    def __init__(self, root: str, class_to_idx: dict, img_size: int = 224):
        self.samples: list[str] = []
        self.tf = train_transforms(img_size)
        for cls_name in class_to_idx:
            class_dir = os.path.join(root, cls_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname == ".DS_Store":
                    continue
                if not fname.lower().endswith(
                        (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")):
                    continue
                self.samples.append(os.path.join(class_dir, fname))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img = Image.open(self.samples[idx]).convert("RGB")
        return self.tf(img), self.tf(img)


def build_splits(root: str, test_root: str, img_size: int,
                 val_fraction: float, seed: int):
    classes = sorted(
        c for c in os.listdir(root)
        if os.path.isdir(os.path.join(root, c)) and c != ".DS_Store"
    )
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)

    full_train = datasets.ImageFolder(root, transform=train_transforms(img_size))
    val_template = datasets.ImageFolder(root, transform=test_transforms(img_size))
    test_ds = datasets.ImageFolder(test_root, transform=test_transforms(img_size))

    targets = np.array(full_train.targets)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction,
                                 random_state=seed)
    train_idx, val_idx = next(sss.split(np.zeros_like(targets), targets))

    train_ds = Subset(full_train, list(train_idx))
    val_ds = Subset(val_template, list(val_idx))
    ssl_ds = SimCLRDataset(root, class_to_idx, img_size)

    return classes, class_to_idx, num_classes, train_ds, val_ds, ssl_ds, test_ds


def nt_xent(z1: torch.Tensor, z2: torch.Tensor,
            temperature: float = 0.5) -> torch.Tensor:
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature

    positives = torch.cat(
        [torch.diag(sim, batch_size), torch.diag(sim, -batch_size)]
    )
    sim = sim.masked_fill(
        torch.eye(2 * batch_size, device=z.device).bool(), -1e4)
    denominator = sim.exp().sum(dim=1)
    return -(positives - torch.log(denominator + 1e-12)).mean()


def train_simclr(model: nn.Module, loader: DataLoader, device: torch.device,
                 epochs: int, lr: float, ckpt_path: str,
                 log_every: int = 20, accum_steps: int = 1,
                 amp: bool = True) -> None:
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    use_amp = amp and device.type == "cuda"

    model.train()
    for epoch in range(1, epochs + 1):
        running = 0.0
        n_batches = 0
        optimizer.zero_grad(set_to_none=True)
        for i, (view1, view2) in enumerate(loader):
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                _, p1 = model(view1)
                _, p2 = model(view2)
                loss_full = nt_xent(p1, p2)
                loss = loss_full / accum_steps
            scaler.scale(loss).backward()
            last_batch = (i + 1) == len(loader)
            if (i + 1) % accum_steps == 0 or last_batch:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running += loss_full.item()
            n_batches += 1
        scheduler.step()
        if epoch % log_every == 0 or epoch == 1:
            print(f"  [SimCLR] {epoch:3d}/{epochs}  loss={running / max(n_batches, 1):.4f}")

    torch.save(model.backbone.state_dict(), ckpt_path)
    print(f"  [SimCLR] backbone saved -> {ckpt_path}")


def train_supervised(model: nn.Module, train_loader: DataLoader,
                     val_loader: DataLoader, device: torch.device,
                     epochs: int, lr: float, weight_decay: float,
                     label_smoothing: float, class_weights: torch.Tensor | None,
                     mixup_alpha: float, cutmix_alpha: float,
                     backbone_lr_multiplier: float, ckpt_path: str,
                     frozen: bool = False, log_every: int = 20,
                     num_classes: int = 16, amp: bool = True,
                     patience: int | None = None) -> float:
    model = model.to(device)
    for p in model.backbone.parameters():
        p.requires_grad = not frozen
    model.backbone.train(not frozen)

    criterion = nn.CrossEntropyLoss(weight=class_weights,
                                    label_smoothing=label_smoothing)

    aug = v2.RandomChoice([
        v2.MixUp(alpha=mixup_alpha, num_classes=num_classes),
        v2.CutMix(alpha=cutmix_alpha, num_classes=num_classes),
    ])

    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and not n.startswith("head")]
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and n.startswith("head")]
    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params,
                             "lr": lr * backbone_lr_multiplier})
    if head_params:
        param_groups.append({"params": head_params, "lr": lr})

    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")

    best_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    start = time.time()

    model.train()
    for epoch in range(1, epochs + 1):
        running = 0.0
        n_batches = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x_m, y_m = aug(x, y)

            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda" and amp:
                with torch.amp.autocast("cuda"):
                    logits = model(x_m)
                    loss = criterion(logits, y_m)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x_m)
                loss = criterion(logits, y_m)
                loss.backward()
                optimizer.step()

            running += loss.item()
            n_batches += 1

        scheduler.step()

        val_acc, val_loss = _validate(model, val_loader, device)
        if epoch % log_every == 0 or epoch == 1:
            elapsed = time.time() - start
            print(f"  [sup] {epoch:3d}/{epochs} "
                  f"lr {optimizer.param_groups[0]['lr']:.2e} "
                  f"train_loss={running / max(n_batches, 1):.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                  f"({elapsed:.1f}s)")

        improved = val_acc > best_acc
        if improved:
            best_acc = val_acc
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_acc": best_acc,
            }, ckpt_path)
            print(f"  New best val acc: {best_acc:.4f} -> {ckpt_path}")
        else:
            patience_counter += 1
            if patience is not None and patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}. Best val acc: {best_acc:.4f}")
                break

    model.load_state_dict(best_state)
    return best_acc


def _validate(model: nn.Module, loader: DataLoader,
              device: torch.device) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss_sum += criterion(logits, y).item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return correct / total, loss_sum / total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SimCLR pretraining -> supervised fine-tuning (ResNet-50, from scratch)")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--ssl_epochs", type=int, default=300)
    p.add_argument("--ssl_batch", type=int, default=128,
                   help="per-step SSL batch (two views are processed per step)")
    p.add_argument("--ssl_accum_steps", type=int, default=2,
                   help="gradient accumulation steps (effective batch = ssl_batch * accum_steps)")
    p.add_argument("--ssl_lr", type=float, default=1e-3)
    p.add_argument("--ft_epochs", type=int, default=150)
    p.add_argument("--ft_batch", type=int, default=128)
    p.add_argument("--ft_lr", type=float, default=3e-4)
    p.add_argument("--val_fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--ssl_ckpt", default="simclr_backbone.pt")
    p.add_argument("--ssl_model", default="best_ssl.pt")
    p.add_argument("--patience", type=int, default=None,
                   help="early stopping patience (epochs)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"

    print(f"PyTorch {torch.__version__} | device: {device} | AMP: {amp}")

    train_dir = os.path.join(args.data_dir, "train")
    test_dir = os.path.join(args.data_dir, "test")

    (classes, class_to_idx, num_classes,
     train_ds, val_ds, ssl_ds, test_ds) = build_splits(
        train_dir, test_dir, args.img_size, args.val_fraction, args.seed)

    print(f"Classes ({num_classes}): {classes}")
    print(f"Labeled train: {len(train_ds)} | Val: {len(val_ds)} | "
          f"SSL images: {len(ssl_ds)} | Test: {len(test_ds)} "
          f"(final eval in evaluate.py)")

    targets = np.array([train_ds.dataset.targets[i] for i in train_ds.indices])
    counts = np.bincount(targets, minlength=num_classes).astype(np.float64)
    weights = np.ones(num_classes, dtype=np.float64)
    present = counts > 0
    weights[present] = 1.0 / counts[present]
    weights = weights / weights[present].mean()
    class_weights = torch.tensor(weights, device=device, dtype=torch.float32)
    print(f"Class weights: {np.round(weights, 3)}")

    train_loader = DataLoader(
        train_ds, args.ft_batch,
        sampler=WeightedRandomSampler(weights[targets], len(targets),
                                      replacement=True),
        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, args.ft_batch, shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=device.type == "cuda")
    ssl_loader = DataLoader(ssl_ds, args.ssl_batch, shuffle=True,
                            num_workers=args.num_workers,
                            pin_memory=device.type == "cuda",
                            drop_last=True)

    # ================= Stage 1: SimCLR pretraining =================
    print("\n=== Stage 1: SimCLR self-supervised pretraining ===")
    simclr = SimCLR(resnet50(num_classes=num_classes), proj_dim=128)
    print(simclr)
    print(f"Trainable parameters: {sum(p.numel() for p in simclr.parameters() if p.requires_grad):,}")

    train_simclr(simclr, ssl_loader, device,
                 epochs=args.ssl_epochs, lr=args.ssl_lr,
                 ckpt_path=args.ssl_ckpt,
                 accum_steps=args.ssl_accum_steps, amp=amp)

    # ================= Stage 2a: Linear probe =================
    print("\n=== Stage 2a: Linear probe (head only) ===")
    ft_backbone = resnet50(num_classes=num_classes)
    ft_backbone.load_state_dict(torch.load(args.ssl_ckpt, map_location="cpu",
                                           weights_only=True))
    classifier = Classifier(ft_backbone, num_classes)
    print(classifier)

    train_supervised(classifier, train_loader, val_loader, device,
                     epochs=25, lr=1e-3, weight_decay=1e-4,
                     label_smoothing=0.1, class_weights=class_weights,
                     mixup_alpha=0.2, cutmix_alpha=1.0,
                     backbone_lr_multiplier=1.0, ckpt_path="probe.pt",
                     frozen=True, num_classes=num_classes, amp=amp,
                     patience=args.patience)

    # ================= Stage 2b: Full fine-tuning =================
    print("\n=== Stage 2b: Full fine-tuning (unfrozen backbone) ===")
    train_supervised(classifier, train_loader, val_loader, device,
                     epochs=args.ft_epochs, lr=args.ft_lr,
                     weight_decay=1e-4, label_smoothing=0.1,
                     class_weights=class_weights, mixup_alpha=0.2,
                     cutmix_alpha=1.0, backbone_lr_multiplier=0.1,
                     ckpt_path=args.ssl_model, num_classes=num_classes,
                     amp=amp, patience=args.patience)

    print(f"\nDone. Best SSL classifier -> {args.ssl_model}")
    print("Run evaluate.py on the test split to report final accuracy.")


if __name__ == "__main__":
    main()
