import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, models, transforms

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    roc_auc_score,
)


# ----------------------------
# Dataset wrappers
# ----------------------------
class CompactSubset(torch.utils.data.Dataset):
    """
    Wrap a subset and remap its labels using `mapping` so labels become contiguous [0..K-1].
    """

    def __init__(self, subset, mapping: dict):
        self.subset = subset
        self.mapping = mapping

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        x, y = self.subset[i]
        return x, self.mapping[int(y)]


class DummyLabelSubset(torch.utils.data.Dataset):
    """
    Wrap a subset but ignore its label (useful for unknown samples where we only need images).
    """

    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        x, _ = self.subset[i]
        return x, 0


# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------
# Splitting utilities
# ----------------------------
def stratified_split(
    indices: np.ndarray,
    labels: np.ndarray,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Stratified split over the provided indices/labels.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    rng = np.random.default_rng(seed)

    train_idx, val_idx, test_idx = [], [], []
    for cls in np.unique(labels):
        cls_mask = labels == cls
        cls_indices = indices[cls_mask].copy()
        rng.shuffle(cls_indices)

        n = len(cls_indices)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)

        train_idx.extend(cls_indices[:n_train].tolist())
        val_idx.extend(cls_indices[n_train : n_train + n_val].tolist())
        test_idx.extend(cls_indices[n_train + n_val :].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def make_weighted_sampler(targets: List[int]) -> WeightedRandomSampler:
    targets_np = np.asarray(targets, dtype=int)
    class_counts = np.bincount(targets_np)
    class_counts = np.maximum(class_counts, 1)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[targets_np]
    sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )


# ----------------------------
# OSR scoring methods
# ----------------------------
def confidence_score(logits: torch.Tensor) -> torch.Tensor:
    """
    Max softmax probability. Higher = more confident = more likely known.
    """
    return F.softmax(logits, dim=1).max(dim=1).values


def energy_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Energy-based score: E(x) = -T * log(sum(exp(logits/T))).

    Lower (more negative) energy = more likely in-distribution (known).
    Higher (less negative) energy = more likely out-of-distribution (unknown).
    """
    return -temperature * torch.logsumexp(logits / temperature, dim=1)


def get_osr_scores(
    logits: torch.Tensor, method: str, temperature: float = 1.0
) -> torch.Tensor:
    """
    Returns scores where HIGHER = more likely KNOWN.

    - confidence: higher max-softmax => more known
    - energy: lower energy => more known, so we negate energy to keep the interface consistent
    """
    if method == "confidence":
        return confidence_score(logits)
    if method == "energy":
        return -energy_score(logits, temperature)
    raise ValueError(f"Unknown OSR method: {method}")


@torch.no_grad()
def collect_logits_labels(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_list, y_list = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x).cpu().numpy()
        logits_list.append(logits)
        y_list.append(y.numpy())
    return np.concatenate(logits_list), np.concatenate(y_list)


@torch.no_grad()
def collect_scores(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    method: str,
    temperature: float = 1.0,
) -> np.ndarray:
    model.eval()
    scores = []
    for x, _ in loader:
        x = x.to(device)
        logits = model(x)
        scores.append(get_osr_scores(logits, method, temperature).cpu().numpy())
    return np.concatenate(scores)


def choose_threshold(known_scores: np.ndarray, unknown_scores: np.ndarray) -> float:
    """
    Choose threshold that maximizes Youden's J = TPR - FPR.
    Samples are classified as unknown if score < threshold.
    """
    all_scores = np.unique(np.concatenate([known_scores, unknown_scores]))
    best_thr, best_J = None, -1e9

    for thr in all_scores:
        tpr = (known_scores >= thr).mean()    # known correctly accepted
        fpr = (unknown_scores >= thr).mean()  # unknown wrongly accepted as known
        J = tpr - fpr
        if J > best_J:
            best_J, best_thr = J, float(thr)

    if best_thr is None:
        raise RuntimeError("Threshold selection failed (no scores available).")
    return best_thr


# ----------------------------
# Model
# ----------------------------
def build_model(num_classes: int, backbone: str, pretrained: bool) -> nn.Module:
    if backbone == "resnet18":
        model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if backbone == "resnet50":
        model = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError("backbone must be resnet18 or resnet50")


@dataclass
class EpochStats:
    loss: float
    acc: float


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> EpochStats:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    preds_all, y_all = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)

        preds_all.append(preds.detach().cpu().numpy())
        y_all.append(y.detach().cpu().numpy())

    preds_all = np.concatenate(preds_all)
    y_all = np.concatenate(y_all)
    return EpochStats(
        loss=total_loss / len(loader.dataset),
        acc=accuracy_score(y_all, preds_all),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Folder containing aircraft/drone/bird/unknown subfolders.",
    )
    ap.add_argument("--out_dir", type=str, default="outputs")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--backbone", choices=["resnet18", "resnet50"], default="resnet18")
    ap.add_argument("--pretrained", action="store_true")

    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)

    # OSR scoring method
    ap.add_argument(
        "--osr_method",
        choices=["confidence", "energy"],
        default="confidence",
        help="OSR scoring method used for threshold-based rejection.",
    )
    ap.add_argument(
        "--energy_temperature",
        type=float,
        default=1.0,
        help="Temperature for energy-based scoring (only used if osr_method='energy').",
    )

    args = ap.parse_args()
    set_seed(args.seed)

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Device selection: CUDA > MPS (Apple Silicon) > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Transforms
    train_tfms = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Load dataset twice: train uses augmentation; eval is deterministic
    ds_all_train = datasets.ImageFolder(str(data_root), transform=train_tfms)
    ds_all_eval = datasets.ImageFolder(str(data_root), transform=eval_tfms)

    class_to_idx = ds_all_train.class_to_idx  # alphabetical by folder name
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    required = {"aircraft", "drone", "bird", "unknown"}
    if set(class_to_idx.keys()) != required:
        raise ValueError(
            f"Expected exactly folders {sorted(required)} under {data_root}, "
            f"got {sorted(class_to_idx.keys())}"
        )

    unknown_idx = class_to_idx["unknown"]

    targets = np.array(ds_all_train.targets, dtype=int)
    all_indices = np.arange(len(targets))

    # ----------------------------
    # IMPORTANT: unknown is NEVER used for training
    # ----------------------------
    known_mask = targets != unknown_idx
    unk_mask = targets == unknown_idx

    known_indices = all_indices[known_mask]
    known_labels = targets[known_mask]

    unk_indices = all_indices[unk_mask]

    # Split KNOWN into train/val/test (stratified)
    known_train_idx, known_val_idx, known_test_idx = stratified_split(
        known_indices,
        known_labels,
        args.train_frac,
        args.val_frac,
        args.test_frac,
        args.seed,
    )

    # Split UNKNOWN into val/test only (no training)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unk_indices)
    cut = len(unk_indices) // 2
    unk_val_idx = unk_indices[:cut].tolist()
    unk_test_idx = unk_indices[cut:].tolist()

    # Build subsets
    train_subset = Subset(ds_all_train, known_train_idx)  # known only
    val_subset_known = Subset(ds_all_eval, known_val_idx)
    test_subset_known = Subset(ds_all_eval, known_test_idx)

    val_subset_unk = Subset(ds_all_eval, unk_val_idx)
    test_subset_unk = Subset(ds_all_eval, unk_test_idx)

    # Remap known labels to [0..2] in a stable way
    known_class_ids = sorted(
        [class_to_idx["aircraft"], class_to_idx["bird"], class_to_idx["drone"]]
    )
    compact_map = {old: new for new, old in enumerate(known_class_ids)}
    compact_idx_to_name = {new: idx_to_class[old] for old, new in compact_map.items()}

    train_subset = CompactSubset(train_subset, compact_map)
    val_subset_known = CompactSubset(val_subset_known, compact_map)
    test_subset_known = CompactSubset(test_subset_known, compact_map)

    # Unknown subsets: labels are irrelevant (only used to compute scores)
    val_subset_unk = DummyLabelSubset(val_subset_unk)
    test_subset_unk = DummyLabelSubset(test_subset_unk)

    # Balanced sampling for training (known only)
    train_targets_compact = [compact_map[int(targets[i])] for i in known_train_idx]
    sampler = make_weighted_sampler(train_targets_compact)

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
    )
    val_loader_known = DataLoader(
        val_subset_known,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader_known = DataLoader(
        test_subset_known,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    val_loader_unk = DataLoader(
        val_subset_unk,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader_unk = DataLoader(
        test_subset_unk,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    num_classes = 3
    unknown_label = 3  # emitted by OSR post-processing only

    # Train
    model = build_model(
        num_classes=num_classes, backbone=args.backbone, pretrained=args.pretrained
    ).to(device)
    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val = -1.0
    best_path = out_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, device, optim)
        va = run_epoch(model, val_loader_known, device, optimizer=None)
        val_acc = va.acc

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {tr.loss:.4f} acc {tr.acc:.4f} | "
            f"val(known) acc {va.acc:.4f}"
        )

        if val_acc > best_val:
            best_val = val_acc
            torch.save({"model_state": model.state_dict(), "args": vars(args)}, best_path)

    print(f"Saved best model: {best_path} (best val(known)={best_val:.4f})")
    model.load_state_dict(torch.load(best_path, map_location=device)["model_state"])

    # ----------------------------
    # Evaluation
    # ----------------------------

    # (A) Closed-set baseline on KNOWN ONLY (3-way)
    known_logits, known_y = collect_logits_labels(model, test_loader_known, device)
    known_preds = known_logits.argmax(axis=1)

    print("\nClosed-set KNOWN-only test report (3-way):")
    print(
        classification_report(
            known_y,
            known_preds,
            target_names=[compact_idx_to_name[i] for i in range(num_classes)],
        )
    )

    # (B) OSR evaluation: threshold-based rejection using UNKNOWN (not training)
    if len(unk_val_idx) == 0 or len(unk_test_idx) == 0:
        print(
            "Warning: unknown set is empty (or too small to split). "
            "Skipping OSR threshold calibration and open-set evaluation."
        )
        cfg = {
            "mode": "known_only_closed_set",
            "known_classes": compact_idx_to_name,  # 0..2 -> name
            "unknown_label": unknown_label,
        }
        (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        return

    osr_method = args.osr_method
    temperature = args.energy_temperature

    # Threshold calibration (val known + val unknown)
    val_known_scores = collect_scores(model, val_loader_known, device, osr_method, temperature)
    val_unk_scores = collect_scores(model, val_loader_unk, device, osr_method, temperature)
    thr = choose_threshold(val_known_scores, val_unk_scores)

    method_desc = f"{osr_method}" + (f" (T={temperature})" if osr_method == "energy" else "")
    print(f"\nOSR method: {method_desc}")
    print(f"Chosen threshold: {thr:.6f} (unknown if score < threshold)")

    # Test unknown logits
    unk_logits, _ = collect_logits_labels(model, test_loader_unk, device)

    # (Optional but useful) Open-set baseline WITHOUT rejection (force argmax among known)
    all_logits = np.concatenate([known_logits, unk_logits], axis=0)
    t = torch.from_numpy(all_logits)

    base_preds = t.argmax(dim=1).numpy()
    true = np.concatenate(
        [known_y, np.full(len(unk_logits), unknown_label, dtype=int)], axis=0
    )

    # With rejection
    scores = get_osr_scores(t, osr_method, temperature).numpy()
    pred = base_preds.copy()
    pred[scores < thr] = unknown_label

    open_acc = accuracy_score(true, pred)
    true_is_unknown = (true == unknown_label).astype(int)
    pred_is_unknown = (pred == unknown_label).astype(int)
    f1_unk = f1_score(true_is_unknown, pred_is_unknown)
    auroc = roc_auc_score(true_is_unknown, -scores)  # lower score => more unknown

    print("\nOpen-set results (3 known classes + unknown rejection):")
    print(f"Open-set accuracy: {open_acc:.4f}")
    print(f"F1 unknown-detection: {f1_unk:.4f}")
    print(f"AUROC unknown-vs-known: {auroc:.4f}")

    # Save config for inference
    cfg = {
        "mode": "osr_threshold",
        "osr_method": osr_method,
        "threshold": thr,
        "known_classes": compact_idx_to_name,  # 0..2 -> name
        "unknown_label": unknown_label,
    }
    if osr_method == "energy":
        cfg["energy_temperature"] = temperature

    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"\nSaved inference config: {out_dir / 'config.json'}")


if __name__ == "__main__":
    main()