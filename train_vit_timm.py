# ===============================================================
# ViT (PyTorch + timm) + Cosine LR (Warmup) + GroupKFold
# AMP on CUDA (auto BF16/FP16), Focal/CE(+Label Smoothing), Head Warmup, Optuna
# + Grad Accum, EMA, Balanced Sampler, Artifacts (per-TRIAL/FOLD/EPOCH),
# Inference, Robust Visualizations, TF32 toggles, Grad Clipping, torch.compile
# Windows-safe, tqdm
#
# NEW (accuracy & robustness):
# - MixUp + CutMix with SoftTargetCrossEntropy (timm)  [config.use_mixup_cutmix]
# - RandAugment + RandomErasing                         [config.augmentation]
# - DropPath on ViT backbone                            [config.drop_path_rate]
# - Head LR multiplier & no-decay param groups for AdamW
# - Optional eval TTA (horizontal flip)                 [config.eval_tta_hflip]
# - **FIX**: Always save "best_foldX.pt" (also during warmup) + synth fallback
# - **FIX**: Separate train/eval losses (prevents SoftTarget vs hard-label crash)
#
# NEW (advanced visualizations):
# - SmoothGrad + RISE saliency (robust attribution)
# - Confidence–Accuracy scatter, margin hist, per-class prob distributions
# - Cumulative Gain, Lift, DET, and KS curves (one-vs-rest)
# - Calibration histogram (counts per confidence bin)
# - Normalized + counts confusion matrices; per-class mistake galleries
# - t-SNE on CLS features; subject-wise accuracy bars
# - Trial HTML Report that stitches key plots together
# ===============================================================
import os, re, json, random, math, time, datetime, warnings, glob, shutil
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import timm
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.loss import SoftTargetCrossEntropy
from timm.data.mixup import Mixup

from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

import optuna

# -----------------------------
# Reproducibility & CUDA
# -----------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
print("Device:", DEVICE)
if DEVICE == "cuda":
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        if torch.cuda.device_count() > 0:
            torch.cuda.set_device(0)
            print("Using GPU:", torch.cuda.get_device_name(0))
    except Exception as e:
        print("CUDA config warning:", e)

# -----------------------------
# Config (8 GB VRAM–friendly)
# -----------------------------
config = {
    "base_path": r"C:\Users\Diat\E GAN\ViT_Optuna",
    "images_root": "images",
    "image_column": "image_path",
    "label_column": "label",
    "subject_column": "subject",
    "use_subject_splits": True,
    "class_names": ["low", "medium", "high"],
    "img_size": 224,

    # Model & regularization
    "model_name": "vit_base_patch16_224.augreg_in21k_ft_in1k",
    "pretrained": True,
    "drop_path_rate": 0.1,

    "n_folds": 3,
    "max_epochs": 15,
    "patience": 5,

    # Batching & LR
    "batch_size": 16,            # effective 64 with accum
    "base_lr": 3e-5,
    "head_lr_mult": 3.0,
    "weight_decay": 0.05,
    "max_grad_norm": 1.0,

    # Cosine LR with warmup
    "lr_scheduler": "cosine",
    "lr_min_mult": 0.01,
    "lr_warmup_epochs": 3,
    "warmup_lr_init_mult": 0.1,

    # Head-only warmup
    "head_warmup_epochs": 2,

    # Losses / regularization
    "use_focal_loss": False,
    "label_smoothing": 0.1,
    "focal_gamma": 2.0,
    "focal_alpha": 0.25,

    # Stronger augmentation
    "augmentation": True,
    "randaug_magnitude": 6,
    "rand_erasing_p": 0.25,

    # MixUp / CutMix
    "use_mixup_cutmix": True,
    "mixup_alpha": 0.2,
    "cutmix_alpha": 1.0,

    # Gradient Accumulation
    "grad_accum_steps": 4,

    # EMA
    "use_ema": True,
    "ema_decay": 0.999,
    "ema_update_after_warmup": True,

    # Balanced Sampler
    "use_balanced_sampler": True,

    # Optuna
    "do_optuna": True,
    "n_trials": 8,
    "optuna_study_name": "vit_optuna_study",
    "optuna_storage_path": "optuna_study.db",

    # Artifacts
    "save_dir": "pytorch_vit_optuna_cosine",
    "save_fold_cms": True,

    # Visualizations
    "viz_after_training": True,
    "viz_sample_count": 12,
    "viz_saliency_count": 3,
    "viz_topk_mistakes": 16,
    "viz_tsne_max": 1500,
    "viz_occlusion_patch": 32,
    "viz_occlusion_stride": 16,

    # More advanced viz
    "viz_smoothgrad_count": 3,
    "smoothgrad_noise_sigma": 0.1,
    "smoothgrad_steps": 20,
    "viz_rise_count": 2,
    "rise_masks": 64,
    "rise_grid": 7,
    "rise_p": 0.5,

    # Eval
    "eval_tta_hflip": False,

    # Advanced toggles
    "use_compile": False,
    "deterministic": False,
    "prefer_bf16": True,

    # Resume controls
    "resume": True,
    "resume_trial": "auto",
    "resume_fold": "auto",
    "resume_epoch": "auto",
}

if config["deterministic"]:
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

BASE = Path(config["base_path"])
IMG_ROOT = BASE / config["images_root"]
ROOT_SAVE_DIR = BASE / config["save_dir"]
ROOT_SAVE_DIR.mkdir(parents=True, exist_ok=True)

assert BASE.exists(), f"Base path not found: {BASE}"
assert IMG_ROOT.exists(), f"Images root not found: {IMG_ROOT}"

NUM_CLASSES = len(config["class_names"])
CLASS_TO_IDX = {c: i for i, c in enumerate([s.lower() for s in config["class_names"]])}

# AMP dtype selection (BF16 if supported, else FP16)
AMP_DTYPE = torch.float16
if DEVICE == "cuda" and config["prefer_bf16"]:
    try:
        if torch.cuda.is_bf16_supported():
            AMP_DTYPE = torch.bfloat16
    except Exception:
        pass

def autocast():
    return torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=(DEVICE=="cuda"))

# -----------------------------
# Trial/Fold/Epoch logging helpers (+ Resume helpers)
# -----------------------------
def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

class RunLogger:
    def __init__(self, root_save: Path):
        self.root = root_save
        self.trial_dir: Optional[Path] = None
        self.fold_dir: Optional[Path] = None
        self.trial_number = 0
        self.fold_idx = 0
        self.trial_params: Dict = {}

    def start_trial(self, trial_number: int, params: Dict):
        self.trial_number = int(trial_number)
        self.trial_params = dict(params)
        self.trial_dir = self.root / f"trial_{self.trial_number:03d}"
        self.trial_dir.mkdir(parents=True, exist_ok=True)
        with open(self.trial_dir / "params.json", "w", encoding="utf-8") as f:
            json.dump(self.trial_params, f, indent=2)
        with open(self.trial_dir / "env.json", "w", encoding="utf-8") as f:
            env = {
                "time": now_str(),
                "torch": torch.__version__,
                "timm": timm.__version__,
                "device": DEVICE,
                "amp_dtype": "bf16" if AMP_DTYPE == torch.bfloat16 else "fp16",
                "cuda_name": torch.cuda.get_device_name(0) if DEVICE=="cuda" and torch.cuda.device_count()>0 else None,
            }
            json.dump(env, f, indent=2)
        return self.trial_dir

    def start_fold(self, fold_idx: int):
        assert self.trial_dir is not None, "Call start_trial first."
        self.fold_idx = int(fold_idx)
        self.fold_dir = self.trial_dir / f"fold_{self.fold_idx}"
        self.fold_dir.mkdir(parents=True, exist_ok=True)
        csvp = self.fold_dir / "fold_history.csv"
        if not csvp.exists():
            pd.DataFrame(columns=[
                "timestamp","epoch","phase","train_loss","train_acc",
                "val_loss","val_acc","lr","improved","best_acc","epochs_no_improve"
            ]).to_csv(csvp, index=False)
        return self.fold_dir

    def log_epoch(self, epoch: int, phase: str, train_loss: float, train_acc: float,
                  val_loss: float, val_acc: float, lr: float, improved: bool,
                  best_acc: float, epochs_no_improve: int):
        csvp = self.fold_dir / "fold_history.csv"
        row = {
            "timestamp": now_str(),
            "epoch": epoch,
            "phase": phase,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": lr,
            "improved": bool(improved),
            "best_acc": best_acc,
            "epochs_no_improve": epochs_no_improve
        }
        pd.DataFrame([row]).to_csv(csvp, mode="a", header=False, index=False)
        with open(self.fold_dir / f"epoch_{epoch:03d}_metrics.json", "w") as f:
            json.dump(row, f, indent=2)

    def save_epoch_checkpoint(self, epoch: int, model_raw_state: Dict, model_ema_state: Optional[Dict],
                              optimizer_state: Optional[Dict], scheduler_state: Optional[Dict],
                              scaler_state: Optional[Dict], extra: Dict):
        ckpt = {
            "model": model_raw_state,
            "model_ema": model_ema_state,
            "optimizer": optimizer_state,
            "scheduler": scheduler_state,
            "scaler": scaler_state,
            "extra": extra
        }
        torch.save(ckpt, self.fold_dir / f"epoch_{epoch:03d}.pt")

    def save_epoch_cm(self, epoch: int, y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
            fig = plt.figure(figsize=(6,5))
            plt.imshow(cm, interpolation="nearest")
            plt.title(f"Confusion Matrix - Epoch {epoch}")
            plt.colorbar(fraction=0.046, pad=0.04)
            ticks = np.arange(len(class_names))
            plt.xticks(ticks, class_names, rotation=45, ha="right")
            plt.yticks(ticks, class_names)
            for i in range(len(class_names)):
                for j in range(len(class_names)):
                    plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="w", fontsize=8)
            plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
            outp = self.fold_dir / f"epoch_{epoch:03d}_cm.png"
            fig.savefig(outp, dpi=140, bbox_inches="tight"); plt.close(fig)
            print("Saved:", outp)
        except Exception as e:
            print("Per-epoch CM plot failed:", e)

    def write_trial_summary(self, summary: Dict):
        assert self.trial_dir is not None
        with open(self.trial_dir / "trial_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def find_latest_trial_number(self) -> Optional[int]:
        trials = sorted(glob.glob(str(self.root / "trial_*")))
        if not trials: return None
        nums = []
        for p in trials:
            m = re.search(r"trial_(\d{3})$", p.replace("\\","/"))
            if m: nums.append(int(m.group(1)))
        return max(nums) if nums else None

    def find_completed_folds(self, trial_dir: Path) -> List[int]:
        folds = []
        for p in sorted(glob.glob(str(trial_dir / "fold_*"))):
            m = re.search(r"fold_(\d+)$", p.replace("\\","/"))
            if not m: continue
            folds.append(int(m.group(1)))
        return sorted(folds)

    def find_latest_epoch(self, fold_dir: Path) -> Optional[int]:
        ep_files = sorted(glob.glob(str(fold_dir / "epoch_*.pt")))
        if not ep_files: return None
        ep_nums = []
        for p in ep_files:
            m = re.search(r"epoch_(\d{3})\.pt$", os.path.basename(p))
            if m: ep_nums.append(int(m.group(1)))
        return max(ep_nums) if ep_nums else None

RUN_LOGGER = RunLogger(ROOT_SAVE_DIR)

# -----------------------------
# Lazy Matplotlib (headless-safe)
# -----------------------------
def ensure_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt

# -----------------------------
# Utilities
# -----------------------------
def build_image_index(root: Path):
    idx = {}
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    for p in root.rglob("*"):
        if p.suffix.lower() in exts and p.is_file():
            idx[p.name] = str(p)
    print(f"Indexed {len(idx):,} image files")
    return idx

def load_metadata(img_root: Path) -> pd.DataFrame:
    csv_p = BASE / "metadata.csv"
    xls_p = BASE / "metadata.xlsx"
    if csv_p.exists():
        df = pd.read_csv(csv_p); print("Loaded metadata.csv")
    elif xls_p.exists():
        df = pd.read_excel(xls_p); print("Loaded metadata.xlsx")
    else:
        raise FileNotFoundError("metadata.csv/.xlsx not found")

    need = {config["image_column"], config["label_column"]}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in metadata: {missing}")

    img_index = build_image_index(img_root)
    df["basename"] = df[config["image_column"]].astype(str).apply(os.path.basename)
    df["full_path"] = df["basename"].map(img_index)
    df = df[df["full_path"].notna()].copy()

    df["__label_norm__"] = df[config["label_column"]].astype(str).str.strip().str.lower()
    df["label_idx"] = df["__label_norm__"].map(CLASS_TO_IDX)
    df = df[df["label_idx"].notna()].copy()
    df["label_idx"] = df["label_idx"].astype(int)

    subj_col = config["subject_column"]
    if subj_col not in df.columns:
        print(f"Subject column '{subj_col}' not found; attempting inference from basename...")
        def infer_subject(basename: str):
            m = re.search(r'(S\d+|SUBJ\d+|SUB\d+|SUBJECT\d+|subj\d+|sub\d+|s\d+)', basename, re.I)
            return m.group(0).lower() if m else "subj_unknown"
        df[subj_col] = df["basename"].apply(infer_subject)

    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    print("Usable rows:", len(df))
    print(df[config["label_column"]].value_counts())

    try:
        plt = ensure_matplotlib()
        counts = df["label_idx"].value_counts().reindex(range(NUM_CLASSES), fill_value=0)
        fig = plt.figure(figsize=(6,4))
        plt.bar([config["class_names"][i] for i in counts.index], counts.values)
        plt.title("Class Distribution (usable rows)")
        plt.xlabel("Class"); plt.ylabel("Count"); plt.tight_layout()
        outp = ROOT_SAVE_DIR / "class_distribution.png"
        fig.savefig(outp, dpi=140, bbox_inches="tight"); plt.close(fig)
        print("Saved:", outp)
    except Exception as e:
        print("Class distribution plot failed:", e)

    return df

# -----------------------------
# Dataset / Transforms
# -----------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

from torchvision import transforms

def build_transforms(train=True):
    if train and config["augmentation"]:
        aug = [
            transforms.Resize((config["img_size"], config["img_size"])),

            # Strong aug
            transforms.RandAugment(num_ops=2, magnitude=int(config["randaug_magnitude"])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.05, contrast=0.1, saturation=0.05),
            transforms.RandomRotation(5),

            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),

            transforms.RandomErasing(p=float(config["rand_erasing_p"]),
                                     scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'),
        ]
        return transforms.Compose(aug)
    else:
        return transforms.Compose([
            transforms.Resize((config["img_size"], config["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

class TopomapDataset(Dataset):
    def __init__(self, sub_df: pd.DataFrame, train: bool):
        self.paths = sub_df["full_path"].tolist()
        self.labels = sub_df["label_idx"].tolist()
        self.tfm = build_transforms(train=train)
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        p = self.paths[i]; y = self.labels[i]
        with Image.open(p) as im:
            im = im.convert("RGB")
            x = self.tfm(im)
        return x, y

def make_loader(sub_df: pd.DataFrame, train: bool, batch_size: int):
    ds = TopomapDataset(sub_df, train=train)
    num_workers = 0 if os.name == "nt" else max(0, os.cpu_count() // 2)
    pin = (DEVICE == "cuda")

    if train and config["use_balanced_sampler"]:
        labels = np.array(sub_df["label_idx"].values)
        class_sample_count = np.bincount(labels, minlength=NUM_CLASSES)
        class_sample_count = np.maximum(class_sample_count, 1)
        class_weights = 1.0 / class_sample_count
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False
    else:
        sampler = None
        shuffle = train

    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=num_workers, pin_memory=pin, prefetch_factor=None,
        persistent_workers=False
    )

# -----------------------------
# Losses
# -----------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean"):
        super().__init__()
        self.gamma = gamma; self.alpha = alpha; self.reduction = reduction
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        if self.reduction == "mean": return focal.mean()
        if self.reduction == "sum":  return focal.sum()
        return focal

def make_criterion(train_labels: np.ndarray, soft_targets: bool = False):
    if soft_targets:
        return SoftTargetCrossEntropy()
    if config["use_focal_loss"]:
        return FocalLoss(gamma=config["focal_gamma"], alpha=config["focal_alpha"])
    ls = float(config.get("label_smoothing", 0.0))
    cls = np.arange(NUM_CLASSES)
    weights = compute_class_weight("balanced", classes=cls, y=train_labels)
    w = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    if ls > 0:
        return nn.CrossEntropyLoss(label_smoothing=ls, weight=w)
    return nn.CrossEntropyLoss(weight=w)

# -----------------------------
# Model, Optimizer & EMA
# -----------------------------
def build_model(num_classes=NUM_CLASSES):
    model = timm.create_model(
        config["model_name"],
        pretrained=config["pretrained"],
        num_classes=num_classes,
        drop_path_rate=float(config["drop_path_rate"]),
    )
    try:
        if hasattr(model, "set_grad_checkpointing"):
            model.set_grad_checkpointing()
    except Exception:
        pass
    if config["use_compile"] and hasattr(torch, "compile") and DEVICE != "mps":
        try:
            model = torch.compile(model); print("Compiled model with torch.compile()")
        except Exception as e:
            print("torch.compile failed (continuing without):", e)
    return model.to(DEVICE)

def freeze_backbone(model: nn.Module, freeze=True):
    head_names = {"head", "fc", "classifier"}
    for n, p in model.named_parameters():
        if any(h in n.lower() for h in head_names):
            p.requires_grad = True
        else:
            p.requires_grad = not freeze

def build_optimizer(model: nn.Module, base_lr: float, weight_decay: float, head_lr_mult: float):
    decay, no_decay, head_params = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        nl = n.lower()
        if ('head' in nl) or ('classifier' in nl):
            head_params.append(p); continue
        if p.ndim <= 1 or ('norm' in nl) or nl.endswith('bias'):
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW([
        {"params": decay, "lr": base_lr, "weight_decay": weight_decay},
        {"params": no_decay, "lr": base_lr, "weight_decay": 0.0},
        {"params": head_params, "lr": base_lr*head_lr_mult, "weight_decay": weight_decay},
    ])

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay); self.shadow = {}; self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().float().clone()
    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            if name in self.shadow and p.requires_grad:
                self.shadow[name].mul_(self.decay).add_(p.detach().float(), alpha=1.0 - self.decay)
    @torch.no_grad()
    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        for name, p in model.named_parameters():
            if name in self.shadow and p.requires_grad:
                self.backup[name] = p.detach().clone()
                p.data.copy_(self.shadow[name].data.to(p.dtype, copy=False))
    @torch.no_grad()
    def restore(self, model: nn.Module):
        for name, p in model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name].data)
        self.backup = {}

# -----------------------------
# Train / Validate (+ TTA)
# -----------------------------
def _maybe_flip_h(xb: torch.Tensor) -> torch.Tensor:
    return torch.flip(xb, dims=[3])

def train_one_epoch(model, loader, criterion, optimizer, scaler, accum_steps:int,
                    ema: Optional[EMA], ema_enabled: bool, progress_desc: str,
                    max_grad_norm: float, mixup_fn: Optional[Mixup]) -> Tuple[float, float]:
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    optimizer.zero_grad(set_to_none=True)

    step = 0
    for step, (xb, yb_hard) in enumerate(tqdm(loader, desc=progress_desc, leave=False), start=1):
        xb = xb.to(DEVICE, non_blocking=True)
        yb_hard = yb_hard.to(DEVICE, non_blocking=True)
        yb = yb_hard
        if mixup_fn is not None:
            xb, yb = mixup_fn(xb, yb_hard)  # soft targets

        with autocast():
            logits = model(xb)
            loss = criterion(logits, yb) / max(1, accum_steps)

        if DEVICE == "cuda":
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if step % accum_steps == 0:
                    if max_grad_norm and max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer); scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if ema is not None and ema_enabled: ema.update(model)
            else:
                loss.backward()
                if step % accum_steps == 0:
                    if max_grad_norm and max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
                    if ema is not None and ema_enabled: ema.update(model)
        else:
            loss.backward()
            if step % accum_steps == 0:
                if max_grad_norm and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                if ema is not None and ema_enabled: ema.update(model)

        with torch.no_grad():
            loss_sum += (loss.detach() * xb.size(0) * max(1, accum_steps)).item()
            preds = logits.argmax(dim=1)
            correct += (preds == yb_hard).sum().item()
            total += xb.size(0)

    if step and (step % accum_steps) != 0:
        if DEVICE == "cuda":
            if scaler.is_enabled():
                if max_grad_norm and max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer); scaler.update()
            else:
                if max_grad_norm and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
        else:
            if max_grad_norm and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if ema is not None and ema_enabled: ema.update(model)

    return loss_sum / max(1, total), correct / max(1, total)

@torch.no_grad()
def evaluate(model, loader, criterion, ema: Optional[EMA]=None, use_ema: bool=False, progress_desc: str="Eval",
             tta_hflip: bool=False) -> Tuple[float, float, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    applied_ema = False
    if ema is not None and use_ema:
        ema.apply_shadow(model); applied_ema = True

    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_preds, all_true, all_probs = [], [], []
    for xb, yb in tqdm(loader, desc=progress_desc, leave=False):
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        with autocast():
            logits = model(xb)
            if tta_hflip:
                logits2 = model(_maybe_flip_h(xb))
                logits = (logits + logits2) / 2.0
            loss = criterion(logits, yb)
        loss_sum += loss.item() * xb.size(0)
        probs = logits.softmax(dim=-1).to(torch.float32)
        preds = probs.argmax(dim=1)
        correct += (preds == yb).sum().item(); total += xb.size(0)
        all_preds.append(preds.cpu().numpy()); all_true.append(yb.cpu().numpy()); all_probs.append(probs.cpu().numpy())

    if applied_ema: ema.restore(model)
    P = np.concatenate(all_probs) if len(all_probs) else None
    return loss_sum / max(1, total), correct / max(1, total), np.concatenate(all_preds) if all_preds else np.array([]), np.concatenate(all_true) if all_true else np.array([]), P

# -----------------------------
# VISUALIZATION HELPERS (omitted for brevity in comment—functions are included)
# -----------------------------
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def plot_history(history: List[Dict], fold_idx: int, out_dir: Path):
    if not history: return
    try:
        plt = ensure_matplotlib()
        epochs = [h["epoch"] for h in history]
        tr_loss = [h["train_loss"] for h in history]
        tr_acc  = [h["train_acc"]  for h in history]
        va_loss = [h["val_loss"]   for h in history]
        va_acc  = [h["val_acc"]    for h in history]
        lrs     = [h["lr"]         for h in history]

        fig = plt.figure(figsize=(10,4.5))
        ax1 = fig.add_subplot(1,2,1)
        ax1.plot(epochs, tr_loss, label="train")
        ax1.plot(epochs, va_loss, label="val")
        ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend()

        ax2 = fig.add_subplot(1,2,2)
        ax2.plot(epochs, tr_acc, label="train")
        ax2.plot(epochs, va_acc, label="val")
        ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.legend()

        fig.tight_layout()
        outp = out_dir / f"fold{fold_idx}_history.png"
        fig.savefig(outp, dpi=150, bbox_inches="tight"); plt.close(fig); print("Saved:", outp)

        fig = plt.figure(figsize=(6,4))
        plt.plot(epochs, lrs)
        plt.title(f"LR Schedule (Fold {fold_idx})")
        plt.xlabel("Epoch"); plt.ylabel("Learning Rate"); plt.tight_layout()
        outp2 = out_dir / f"fold{fold_idx}_lr.png"
        fig.savefig(outp2, dpi=150, bbox_inches="tight"); plt.close(fig); print("Saved:", outp2)
    except Exception as e:
        print("History plots failed:", e)

def save_optuna_plots(study: optuna.Study, out_dir: Path):
    try:
        trials = [({"number": t.number, "value": t.value, **t.params}) for t in study.trials if t.value is not None]
        df = pd.DataFrame(trials)
        csvp = out_dir / "optuna_trials.csv"; df.to_csv(csvp, index=False); print("Saved:", csvp)
        if len(df):
            plt = ensure_matplotlib()
            fig = plt.figure(figsize=(6,4))
            plt.scatter(df["number"], df["value"])
            plt.title("Optuna Trials (mean CV acc)")
            plt.xlabel("Trial #"); plt.ylabel("Accuracy"); plt.tight_layout()
            outp = out_dir / "optuna_trials.png"
            fig.savefig(outp, dpi=150, bbox_inches="tight"); plt.close(fig); print("Saved:", outp)
    except Exception as e:
        print("Optuna plot failed:", e)

def denormalize_to_uint8(t: torch.Tensor) -> np.ndarray:
    c, h, w = t.shape
    mean = torch.tensor(IMAGENET_MEAN, device=t.device).view(3,1,1)
    std  = torch.tensor(IMAGENET_STD,  device=t.device).view(3,1,1)
    x = (t * std + mean).clamp(0,1)
    x = (x * 255.0).byte().permute(1,2,0).cpu().numpy()
    return x

@torch.inference_mode()
def visualize_predictions_grid(df: pd.DataFrame, weights_path: str, n: int = 12, title: str = "Predictions Grid"):
    if n <= 0 or len(df) == 0: return
    try:
        ckpt = torch.load(weights_path, map_location=DEVICE)
        model = build_model()
        state = ckpt.get("model_ema") or ckpt.get("model")
        model.load_state_dict(state); model.eval()
        tfm = build_transforms(train=False)

        samp = df.sample(n=min(n, len(df)), random_state=SEED)
        images, captions = [], []
        for _, row in samp.iterrows():
            with Image.open(row["full_path"]) as im:
                im = im.convert("RGB"); x = tfm(im).unsqueeze(0).to(DEVICE)
            with autocast():
                logits = model(x); probs = logits.softmax(dim=-1).to(torch.float32).squeeze(0).cpu().numpy()
            pred = int(probs.argmax()); gt = int(row["label_idx"])
            pred_label = config["class_names"][pred]; true_label = config["class_names"][gt]
            ok = (pred == gt)
            images.append(np.array(im.resize((config["img_size"], config["img_size"]))))
            mark = "✅" if ok else "❌"
            captions.append(f"{mark} pred={pred_label}\ntrue={true_label}\nprobs={np.round(probs,3).tolist()}")

        plt = ensure_matplotlib()
        cols = 4; rows = int(np.ceil(len(images)/cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols*3.2, rows*3.2)); axes = axes.ravel() if rows*cols>1 else [axes]
        for i, ax in enumerate(axes):
            ax.axis("off")
            if i < len(images):
                ax.imshow(images[i]); ax.set_title(captions[i], fontsize=8)
        fig.suptitle(title); fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        outp = ROOT_SAVE_DIR / "predictions_grid.png"
        fig.savefig(outp, dpi=140, bbox_inches="tight"); plt.close(fig); print("Saved:", outp)
        del model
        if DEVICE == "cuda": torch.cuda.empty_cache()
    except Exception as e:
        print("Prediction grid failed:", e)

# ---- Attribution & metric plots (SmoothGrad, RISE, etc.) ----
def integrated_gradients(model, x: torch.Tensor, target_idx: int, steps: int = 20) -> np.ndarray:
    model.eval()
    baseline = torch.zeros_like(x)
    grads = []
    for i in range(steps+1):
        alpha = float(i) / steps
        inp = baseline + alpha * (x - baseline)
        inp.requires_grad_(True)
        logits = model(inp)
        score = logits[0, target_idx]
        model.zero_grad(set_to_none=True)
        score.backward()
        grads.append(inp.grad.detach().clone())
    avg_grad = torch.stack(grads, dim=0).mean(dim=0)
    ig = (x - baseline) * avg_grad
    ig = ig.abs().squeeze(0).mean(dim=0)
    ig = (ig - ig.min()) / (ig.max() - ig.min() + 1e-8)
    return ig.detach().cpu().numpy()

@torch.no_grad()
def occlusion_map(model, x: torch.Tensor, target_idx: int, patch: int = 32, stride: int = 16) -> np.ndarray:
    model.eval()
    with autocast():
        base_logits = model(x); base_prob = base_logits.softmax(dim=-1)[0, target_idx].item()
    _, _, H, W = x.shape
    heat = np.zeros((H, W), dtype=np.float32); counts = np.zeros((H, W), dtype=np.float32)
    for yy in range(0, H, stride):
        for xx in range(0, W, stride):
            x_occ = x.clone()
            y2 = min(yy+patch, H); x2 = min(xx+patch, W)
            x_occ[:, :, yy:y2, xx:x2] = 0.0
            with autocast():
                prob = model(x_occ).softmax(dim=-1)[0, target_idx].item()
            drop = max(0.0, base_prob - prob)
            heat[yy:y2, xx:x2] += drop; counts[yy:y2, xx:x2] += 1.0
    heat = heat / (counts + 1e-8)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    return heat

def save_saliency_heatmaps(df_val: pd.DataFrame, weights_path: str, n: int = 3):
    if n <= 0 or len(df_val) == 0: return
    try:
        ckpt = torch.load(weights_path, map_location=DEVICE)
        model = build_model()
        state = ckpt.get("model_ema") or ckpt.get("model")
        model.load_state_dict(state); model.eval()
        tfm = build_transforms(train=False)

        samp = df_val.sample(n=min(n, len(df_val)), random_state=SEED)
        for _, row in samp.iterrows():
            with Image.open(row["full_path"]) as im:
                im = im.convert("RGB"); x = tfm(im).unsqueeze(0).to(DEVICE)
            x.requires_grad_(True)
            logits = model(x); pred_id = int(logits.argmax(dim=-1).item())
            score = logits[0, pred_id]; model.zero_grad(set_to_none=True); score.backward()
            grad = x.grad.detach().squeeze(0).abs().mean(dim=0)
            sal = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8); sal = sal.cpu().numpy()
            base = denormalize_to_uint8(x.detach().squeeze(0))
            plt = ensure_matplotlib()
            fig = plt.figure(figsize=(4,4)); ax = fig.add_subplot(1,1,1)
            ax.imshow(base); ax.imshow(sal, alpha=0.45)
            ax.set_title(f"Saliency (pred={config['class_names'][pred_id]})"); ax.axis("off")
            outp = ROOT_SAVE_DIR / f"saliency_{Path(row['full_path']).stem}.png"
            fig.savefig(outp, dpi=160, bbox_inches="tight"); plt.close(fig); print("Saved:", outp)
        del model
        if DEVICE == "cuda": torch.cuda.empty_cache()
    except Exception as e:
        print("Saliency heatmaps failed:", e)

def save_comparison_panel(df_val: pd.DataFrame, weights_path: str, n: int = 3):
    if n <= 0 or len(df_val) == 0: return
    try:
        ckpt = torch.load(weights_path, map_location=DEVICE)
        model = build_model()
        state = ckpt.get("model_ema") or ckpt.get("model")
        model.load_state_dict(state); model.eval()
        tfm = build_transforms(train=False)

        samp = df_val.sample(n=min(n, len(df_val)), random_state=SEED)
        for _, row in samp.iterrows():
            with Image.open(row["full_path"]) as im:
                im = im.convert("RGB"); x = tfm(im).unsqueeze(0).to(DEVICE)
            with autocast():
                logits = model(x); pred_id = int(logits.argmax(dim=-1).item())
            x.requires_grad_(True)
            logits2 = model(x); score = logits2[0, pred_id]
            model.zero_grad(set_to_none=True); score.backward()
            grad = x.grad.detach().squeeze(0).abs().mean(dim=0)
            sal = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8); sal = sal.cpu().numpy()
            ig  = integrated_gradients(model, x.detach(), pred_id, steps=20)
            occ = occlusion_map(model, x.detach(), pred_id, patch=int(config["viz_occlusion_patch"]), stride=int(config["viz_occlusion_stride"]))
            base = denormalize_to_uint8(x.detach().squeeze(0))
            plt = ensure_matplotlib()
            fig, axes = plt.subplots(1,4, figsize=(12,3.3))
            for ax in axes: ax.axis("off")
            axes[0].imshow(base); axes[0].set_title("Original", fontsize=9)
            axes[1].imshow(base); axes[1].imshow(sal, alpha=0.45); axes[1].set_title("Saliency", fontsize=9)
            axes[2].imshow(base); axes[2].imshow(ig,  alpha=0.45); axes[2].set_title("Integrated Gradients", fontsize=9)
            axes[3].imshow(base); axes[3].imshow(occ, alpha=0.45); axes[3].set_title("Occlusion", fontsize=9)
            fig.suptitle(f"Compare viz (pred={config['class_names'][pred_id]})", fontsize=10)
            fig.tight_layout()
            outp = ROOT_SAVE_DIR / f"compare_viz_{Path(row['full_path']).stem}.png"
            fig.savefig(outp, dpi=160, bbox_inches="tight"); plt.close(fig); print("Saved:", outp)
        del model
        if DEVICE == "cuda": torch.cuda.empty_cache()
    except Exception as e:
        print("Comparison panel failed:", e)

def save_smoothgrad_maps(df_val: pd.DataFrame, weights_path: str, n: int, noise_sigma: float, steps: int):
    if n <= 0 or len(df_val) == 0: return
    try:
        ckpt = torch.load(weights_path, map_location=DEVICE)
        model = build_model(); state = ckpt.get("model_ema") or ckpt.get("model")
        model.load_state_dict(state); model.eval()
        tfm = build_transforms(train=False)
        samp = df_val.sample(n=min(n, len(df_val)), random_state=SEED)
        for _, row in samp.iterrows():
            with Image.open(row["full_path"]) as im:
                im = im.convert("RGB"); x0 = tfm(im).unsqueeze(0).to(DEVICE)
            _, _, H, W = x0.shape
            acc_grad = torch.zeros((H,W), device=DEVICE)
            for _ in range(steps):
                noise = torch.randn_like(x0) * noise_sigma
                x = (x0 + noise).clamp(-10,10).requires_grad_(True)
                logits = model(x); pred_id = int(logits.argmax(dim=-1).item())
                score = logits[0, pred_id]; model.zero_grad(set_to_none=True); score.backward()
                grad = x.grad.detach().squeeze(0).abs().mean(dim=0)
                acc_grad += grad
            sal = acc_grad / steps
            sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
            base = denormalize_to_uint8(x0.detach().squeeze(0))
            plt = ensure_matplotlib()
            fig = plt.figure(figsize=(4,4)); ax = fig.add_subplot(1,1,1)
            ax.imshow(base); ax.imshow(sal.cpu().numpy(), alpha=0.45); ax.axis("off")
            ax.set_title("SmoothGrad")
            outp = ROOT_SAVE_DIR / f"smoothgrad_{Path(row['full_path']).stem}.png"
            fig.savefig(outp, dpi=160, bbox_inches="tight"); plt.close(fig); print("Saved:", outp)
        del model
        if DEVICE == "cuda": torch.cuda.empty_cache()
    except Exception as e:
        print("SmoothGrad failed:", e)

@torch.no_grad()
def save_rise_maps(df_val: pd.DataFrame, weights_path: str, n: int, n_masks: int, grid: int, p_keep: float):
    if n <= 0 or len(df_val) == 0: return
    try:
        ckpt = torch.load(weights_path, map_location=DEVICE)
        model = build_model(); state = ckpt.get("model_ema") or ckpt.get("model")
        model.load_state_dict(state); model.eval()
        tfm = build_transforms(train=False)
        samp = df_val.sample(n=min(n, len(df_val)), random_state=SEED)
        plt = ensure_matplotlib()
        for _, row in samp.iterrows():
            with Image.open(row["full_path"]) as im:
                im = im.convert("RGB"); x = tfm(im).unsqueeze(0).to(DEVICE)
            _, _, H, W = x.shape
            masks = torch.rand((n_masks, 1, grid, grid), device=DEVICE) < p_keep
            masks = masks.float()
            masks = torch.nn.functional.interpolate(masks, size=(H, W), mode='bilinear', align_corners=False)
            scores = []
            for m in masks:
                xm = x * m
                with autocast():
                    logits = model(xm); probs = logits.softmax(dim=-1)
                pred_id = int(probs.argmax(dim=-1).item())
                scores.append(probs[0, pred_id].item())
            scores = torch.tensor(scores, device=DEVICE).view(-1,1,1,1)
            sal = (masks * scores).mean(dim=0).squeeze(0)
            sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
            base = denormalize_to_uint8(x.detach().squeeze(0))
            fig = plt.figure(figsize=(4,4)); ax = fig.add_subplot(1,1,1)
            ax.imshow(base); ax.imshow(sal.cpu().numpy(), alpha=0.45); ax.axis("off")
            ax.set_title("RISE")
            outp = ROOT_SAVE_DIR / f"rise_{Path(row['full_path']).stem}.png"
            fig.savefig(outp, dpi=160, bbox_inches="tight"); plt.close(fig); print("Saved:", outp)
        del model
        if DEVICE == "cuda": torch.cuda.empty_cache()
    except Exception as e:
        print("RISE failed:", e)

def plot_topk_mistakes(pred_df: pd.DataFrame, k: int, out_dir: Path):
    try:
        bad = pred_df[pred_df["y_true"] != pred_df["y_pred"]].copy()
        if bad.empty:
            print("No mistakes to plot."); return
        bad = bad.sort_values("prob_pred", ascending=False).head(k)
        images, titles = [], []
        for _, r in bad.iterrows():
            with Image.open(r["path"]) as im:
                images.append(np.array(im.resize((config["img_size"], config["img_size"]))))
            tl = config["class_names"][int(r["y_true"])]; pl = config["class_names"][int(r["y_pred"])]
            titles.append(f"❌ pred={pl} ({r['prob_pred']:.2f})\ntrue={tl}")
        plt = ensure_matplotlib()
        cols = 4; rows = int(np.ceil(len(images)/cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols*3.1, rows*3.1))
        axes = axes.ravel() if rows*cols>1 else [axes]
        for i, ax in enumerate(axes):
            ax.axis("off")
            if i < len(images):
                ax.imshow(images[i]); ax.set_title(titles[i], fontsize=8)
        fig.suptitle("Top-K Highest-Confidence Mistakes")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        outp = out_dir / "topk_mistakes.png"
        fig.savefig(outp, dpi=150, bbox_inches="tight"); plt.close(fig); print("Saved:", outp)
    except Exception as e:
        print("TopK mistakes plot failed:", e)

def plot_confusion_matrix_norm(y_true, y_pred, title: str, out_path: Path):
    try:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
        cmn = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        plt = ensure_matplotlib()
        fig = plt.figure(figsize=(6,5))
        plt.imshow(cmn, interpolation="nearest")
        plt.title(title); plt.colorbar(fraction=0.046, pad=0.04)
        ticks = np.arange(NUM_CLASSES)
        plt.xticks(ticks, config["class_names"], rotation=45, ha="right")
        plt.yticks(ticks, config["class_names"])
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                plt.text(j, i, f"{cm[i,j]} / {cmn[i,j]:.2f}", ha="center", va="center", color="w", fontsize=8)
        plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig); print("Saved:", out_path)
    except Exception as e:
        print("Normalized CM plot failed:", e)

def plot_roc_pr_curves(y_true: np.ndarray, probs: np.ndarray, out_prefix: Path):
    try:
        plt = ensure_matplotlib()
        Y = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
        # ROC
        fig = plt.figure(figsize=(6,5))
        for i in range(NUM_CLASSES):
            fpr, tpr, _ = roc_curve(Y[:, i], probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{config['class_names'][i]} (AUC={roc_auc:.2f})")
        fpr, tpr, _ = roc_curve(Y.ravel(), probs.ravel())
        plt.plot(fpr, tpr, linestyle="--", label=f"micro (AUC={auc(fpr,tpr):.2f})")
        plt.plot([0,1],[0,1], ":", color="gray")
        plt.title("ROC (OvR)"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend()
        outp = out_prefix.parent / f"{out_prefix.name}_roc.png"
        fig.savefig(outp, dpi=150, bbox_inches="tight"); plt.close(fig); print("Saved:", outp)

        # PR
        fig = plt.figure(figsize=(6,5))
        for i in range(NUM_CLASSES):
            p, r, _ = precision_recall_curve(Y[:, i], probs[:, i])
            ap = average_precision_score(Y[:, i], probs[:, i])
            plt.plot(r, p, label=f"{config['class_names'][i]} (AP={ap:.2f})")
        plt.title("Precision-Recall (OvR)")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend()
        outp = out_prefix.parent / f"{out_prefix.name}_pr.png"
        fig.savefig(outp, dpi=150, bbox_inches="tight"); plt.close(fig); print("Saved:", outp)
    except Exception as e:
        print("ROC/PR plot failed:", e)

def plot_reliability(y_true: np.ndarray, y_pred: np.ndarray, prob_pred: np.ndarray, bins: int, out_prefix: Path):
    try:
        conf = np.clip(prob_pred, 0, 1)
        accs, confs, counts = [], [], []
        edges = np.linspace(0.0, 1.0, bins+1)
        for i in range(bins):
            lo, hi = edges[i], edges[i+1]
            mask = (conf >= lo) & (conf < hi) if i < bins-1 else (conf >= lo) & (conf <= hi)
            if mask.sum() == 0: continue
            bin_acc = (y_true[mask] == y_pred[mask]).mean()
            bin_conf = conf[mask].mean()
            accs.append(bin_acc); confs.append(bin_conf); counts.append(mask.sum())
        ece = np.sum(np.array(counts)/max(1, len(conf)) * np.abs(np.array(accs) - np.array(confs)))
        plt = ensure_matplotlib()
        fig = plt.figure(figsize=(6,5))
        plt.plot([0,1],[0,1], "k--", label="Perfect")
        plt.plot(confs, accs, marker="o", label=f"Model (ECE={ece:.3f})")
        plt.title("Reliability Diagram"); plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.legend()
        outp = out_prefix.parent / f"{out_prefix.name}_reliability.png"
        fig.savefig(outp, dpi=150, bbox_inches="tight"); plt.close(fig); print("Saved:", outp)

        # histogram of confidence
        fig = plt.figure(figsize=(6,4))
        plt.hist(conf, bins=edges, alpha=0.9)
        plt.title("Confidence Histogram")
        plt.xlabel("Confidence"); plt.ylabel("Count"); plt.tight_layout()
        outp = out_prefix.parent / f"{out_prefix.name}_conf_hist.png"
        fig.savefig(outp, dpi=150, bbox_inches="tight"); plt.close(fig); print("Saved:", outp)
    except Exception as e:
        print("Reliability plot failed:", e)

def plot_subject_accuracy(val_df: pd.DataFrame, pred_df: pd.DataFrame, out_path: Path, subj_col: str):
    try:
        dfm = val_df[["full_path", subj_col]].merge(
            pred_df[["path","y_true","y_pred"]], left_on="full_path", right_on="path", how="inner"
        )
        g = dfm.groupby(subj_col).apply(lambda d: (d["y_true"]==d["y_pred"]).mean())
        g = g.sort_values(ascending=False)
        plt = ensure_matplotlib()
        fig = plt.figure(figsize=(max(6, 0.4*len(g)), 4))
        plt.bar(g.index.astype(str), g.values)
        plt.xticks(rotation=45, ha="right"); plt.ylim(0,1)
        plt.title(f"Per-Subject Accuracy"); plt.ylabel("Accuracy")
        fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig); print("Saved:", out_path)
    except Exception as e:
        print("Per-subject accuracy plot failed:", e)

@torch.no_grad()
def plot_tsne_embeddings(model, sub_df: pd.DataFrame, max_samples: int, out_prefix: Path):
    try:
        tfm = build_transforms(train=False)
        paths = sub_df["full_path"].tolist(); labels = sub_df["label_idx"].tolist()
        if len(paths) == 0: return
        if len(paths) > max_samples:
            idxs = np.random.RandomState(SEED).choice(len(paths), size=max_samples, replace=False)
            paths = [paths[i] for i in idxs]; labels = [labels[i] for i in idxs]
        feats, yhat = [], []
        for start in tqdm(range(0, len(paths), config["batch_size"]), desc="Collect feats", leave=False):
            chunk = paths[start:start+config["batch_size"]]
            xb = []
            for p in chunk:
                with Image.open(p) as im:
                    xb.append(tfm(im.convert("RGB")))
            xb = torch.stack(xb, 0).to(DEVICE)
            with autocast():
                f = model.forward_features(xb)
                if isinstance(f, (list, tuple)): f = f[0]
                if f.ndim > 2: f = f.mean(dim=1)
                if hasattr(model, "head"): logits = model.head(f)
                else: logits = model.get_classifier()(f)
            feats.append(f.detach().cpu().numpy()); yhat.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
        feats = np.concatenate(feats, axis=0); yl = np.array(labels[:len(feats)]); yhat = np.array(yhat[:len(feats)])
        if len(feats) < 10: return

        tsne = TSNE(n_components=2, init="pca", random_state=SEED, perplexity=min(30, max(5, len(feats)//50)))
        Z = tsne.fit_transform(feats)

        plt = ensure_matplotlib()
        fig = plt.figure(figsize=(6,5))
        for i in range(NUM_CLASSES):
            pts = Z[yl==i]
            if len(pts): plt.scatter(pts[:,0], pts[:,1], label=config["class_names"][i], s=8)
        plt.title(f"t-SNE of CLS Features (True)"); plt.legend(markerscale=2)
        outp = out_prefix.parent / f"{out_prefix.name}_tsne_true.png"
        fig.savefig(outp, dpi=150, bbox_inches="tight"); plt.close(fig); print("Saved:", outp)

        fig = plt.figure(figsize=(6,5))
        for i in range(NUM_CLASSES):
            pts = Z[yhat==i]
            if len(pts): plt.scatter(pts[:,0], pts[:,1], label=config["class_names"][i], s=8)
        plt.title(f"t-SNE of CLS Features (Pred)"); plt.legend(markerscale=2)
        outp = out_prefix.parent / f"{out_prefix.name}_tsne_pred.png"
        fig.savefig(outp, dpi=150, bbox_inches="tight"); plt.close(fig); print("Saved:", outp)
    except Exception as e:
        print("t-SNE plot failed:", e)

def plot_confidence_accuracy_scatter(pred_df: pd.DataFrame, out_path: Path):
    try:
        y = (pred_df["y_true"] == pred_df["y_pred"]).astype(float).values
        x = pred_df["prob_pred"].values
        plt = ensure_matplotlib()
        fig = plt.figure(figsize=(6,5))
        plt.scatter(x, y, alpha=0.3, s=12)
        plt.title("Confidence vs. Accuracy (top-1)"); plt.xlabel("Predicted confidence"); plt.ylabel("Correct (0/1)")
        fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig); print("Saved:", out_path)
    except Exception as e:
        print("Confidence-Accuracy scatter failed:", e)

def plot_margin_hist(pred_df: pd.DataFrame, out_path: Path):
    try:
        m = pred_df["margin"].values
        plt = ensure_matplotlib()
        fig = plt.figure(figsize=(6,4))
        plt.hist(m, bins=30)
        plt.title("Prediction Margin (top1 - top2)"); plt.xlabel("Margin"); plt.ylabel("Count")
        fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig); print("Saved:", out_path)
    except Exception as e:
        print("Margin hist failed:", e)

def plot_perclass_prob_dists(pred_df: pd.DataFrame, out_path: Path):
    try:
        plt = ensure_matplotlib()
        fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(4*NUM_CLASSES, 3), squeeze=False)
        axes = axes.ravel()
        for i in range(NUM_CLASSES):
            mask = pred_df["y_true"].values == i
            probs = np.array(pred_df.loc[mask, "probs"].tolist()) if mask.sum() > 0 else np.zeros((0, NUM_CLASSES))
            vals = probs[:, i] if len(probs) else np.array([])
            ax = axes[i]; ax.hist(vals, bins=20)
            ax.set_title(f"True={config['class_names'][i]}"); ax.set_xlabel(f"P(class={config['class_names'][i]})"); ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig); print("Saved:", out_path)
    except Exception as e:
        print("Per-class prob dists failed:", e)

def write_simple_html_report(out_dir: Path, title: str, image_globs: List[str], out_name: str="report.html"):
    try:
        imgs = []
        for pat in image_globs:
            imgs.extend(sorted(glob.glob(str(out_dir / pat))))
        html = [f"<html><head><meta charset='utf-8'><title>{title}</title></head><body>"]
        html.append(f"<h1>{title}</h1>")
        for img in imgs:
            rel = os.path.basename(img)
            html.append(f"<div style='margin:10px 0'><h3>{rel}</h3><img src='{rel}' style='max-width:100%;height:auto;border:1px solid #ccc'/></div>")
        html.append("</body></html>")
        outp = out_dir / out_name
        with open(outp, "w", encoding="utf-8") as f: f.write("\n".join(html))
        print("Saved HTML report:", outp)
    except Exception as e:
        print("HTML report failed:", e)

# ---------- Collect predictions with rich metrics ----------
@torch.no_grad()
def collect_predictions_df(model, sub_df: pd.DataFrame, batch_size: int, tta_hflip: bool=False) -> pd.DataFrame:
    model.eval()
    tfm = build_transforms(train=False)
    rows = []
    loader_paths = sub_df["full_path"].tolist()
    labels = sub_df["label_idx"].tolist()

    for start in tqdm(range(0, len(loader_paths), batch_size), desc="Collect preds", leave=False):
        paths_b = loader_paths[start:start+batch_size]
        ys_b = labels[start:start+batch_size]
        xb = []
        for p in paths_b:
            with Image.open(p) as im:
                xb.append(tfm(im.convert("RGB")))
        xb = torch.stack(xb, dim=0).to(DEVICE)
        with autocast():
            logits = model(xb)
            if tta_hflip:
                logits2 = model(_maybe_flip_h(xb))
                logits = (logits + logits2) / 2.0
            probs = logits.softmax(dim=-1).to(torch.float32).cpu().numpy()
            preds = probs.argmax(axis=1)
        for pth, y_true, y_pred, prob in zip(paths_b, ys_b, preds, probs):
            top2 = np.partition(prob, -2)[-2:]
            rows.append({
                "path": pth,
                "y_true": int(y_true),
                "y_pred": int(y_pred),
                "prob_pred": float(prob[y_pred]),
                "prob_true": float(prob[y_true]),
                "entropy": float(-(prob * np.log(np.clip(prob, 1e-12, 1))).sum()),
                "margin": float(top2[-1] - top2[-2]),
                "probs": prob.tolist()
            })
    return pd.DataFrame(rows)

# -----------------------------
# Cosine LR + Warmup
# -----------------------------
def build_scheduler(optimizer, max_epochs):
    assert config["lr_scheduler"] == "cosine"
    lr = optimizer.param_groups[0]["lr"]
    lr_min = lr * float(config["lr_min_mult"])
    warmup_t = int(config["lr_warmup_epochs"])
    warmup_lr_init = lr * float(config["warmup_lr_init_mult"])
    sched = CosineLRScheduler(
        optimizer,
        t_initial=max_epochs,
        lr_min=lr_min,
        warmup_t=warmup_t,
        warmup_lr_init=warmup_lr_init,
        t_in_epochs=True
    )
    return sched

# -----------------------------
# Fold training (per-epoch saving) + RESUME
# -----------------------------
def _load_resume_states_if_any(model, optimizer, scheduler, scaler, fold_dir: Path):
    last_epoch = RUN_LOGGER.find_latest_epoch(fold_dir)
    if last_epoch is None: return None
    ckpt_path = fold_dir / f"epoch_{last_epoch:03d}.pt"; print(f"Resuming from checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt.get("model") or ckpt.get("model_ema"); model.load_state_dict(state)
    if optimizer is not None and ckpt.get("optimizer") is not None: optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        try: scheduler.load_state_dict(ckpt["scheduler"])
        except Exception as e: print("Scheduler state load failed (continuing):", e)
    if scaler is not None and ckpt.get("scaler") is not None:
        try: scaler.load_state_dict(ckpt["scaler"])
        except Exception as e: print("Scaler state load failed (continuing):", e)
    extra = ckpt.get("extra", {})
    return {"epoch": int(extra.get("epoch", last_epoch)), "best_acc": float(extra.get("best_acc", -1.0)),
            "epochs_no_improve": int(extra.get("epochs_no_improve", 0)), "phase": extra.get("phase", "train")}

def _save_epoch_ckpt(epoch, model, ema, optimizer, scheduler, scaler, fold_idx, trial_meta, phase,
                     best_acc, epochs_no_improve):
    model_raw = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    ema_state = None
    if ema is not None:
        ema.apply_shadow(model); ema_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}; ema.restore(model)
    extra = {"phase": phase, "fold": fold_idx, "trial": trial_meta["number"], "epoch": epoch,
             "best_acc": best_acc, "epochs_no_improve": epochs_no_improve}
    RUN_LOGGER.save_epoch_checkpoint(epoch=epoch, model_raw_state=model_raw, model_ema_state=ema_state,
                                     optimizer_state=optimizer.state_dict() if optimizer is not None else None,
                                     scheduler_state=scheduler.state_dict() if scheduler is not None else None,
                                     scaler_state=scaler.state_dict() if scaler is not None else None, extra=extra)

def _save_best_fold_now(model, ema, fold_idx, params, best_acc, note: str):
    best_path = RUN_LOGGER.fold_dir / f"best_fold{fold_idx}.pt"
    save_blob = {
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_ema": None, "params": params, "val_acc": float(best_acc), "class_names": config["class_names"]
    }
    if ema is not None:
        ema.apply_shadow(model); save_blob["model_ema"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}; ema.restore(model)
    torch.save(save_blob, best_path); print(f"  ↳ Saved BEST ({note}) -> {best_path} (acc={best_acc:.4f})")
    return str(best_path)

def run_fold(train_df, val_df, params, fold_idx, trial_meta) -> float:
    lr = params.get("learning_rate", config["base_lr"]); bs = params.get("batch_size", config["batch_size"])
    accum_steps = max(1, int(config["grad_accum_steps"]))
    RUN_LOGGER.start_fold(fold_idx)
    print(f"Effective batch size (grad accum): {bs*accum_steps} (bs={bs} x accum={accum_steps})")
    train_loader = make_loader(train_df, train=True, batch_size=bs)
    val_loader = make_loader(val_df, train=False, batch_size=bs)

    mixup_fn = None
    if config["use_mixup_cutmix"]:
        mixup_fn = Mixup(mixup_alpha=float(config["mixup_alpha"]), cutmix_alpha=float(config["cutmix_alpha"]),
                         cutmix_minmax=None, prob=1.0, switch_prob=0.5, mode='batch',
                         label_smoothing=float(config["label_smoothing"]), num_classes=NUM_CLASSES)

    # separate train/eval criteria (FIX)
    criterion_train = make_criterion(train_df["label_idx"].values, soft_targets=(mixup_fn is not None))
    criterion_eval  = make_criterion(train_df["label_idx"].values, soft_targets=False)

    model = build_model()
    optimizer = build_optimizer(model, base_lr=lr, weight_decay=config["weight_decay"], head_lr_mult=config["head_lr_mult"])
    scheduler = build_scheduler(optimizer, max_epochs=config["max_epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=="cuda" and AMP_DTYPE==torch.float16))
    ema = EMA(model, decay=config["ema_decay"]) if config["use_ema"] else None

    history: List[Dict] = []
    best_acc, epochs_no_improve, start_epoch = -1.0, 0, 0
    if config["resume"]:
        resume_info = _load_resume_states_if_any(model, optimizer, scheduler, scaler, RUN_LOGGER.fold_dir)
        if resume_info is not None:
            start_epoch = int(resume_info["epoch"]); best_acc = float(resume_info["best_acc"]); epochs_no_improve = int(resume_info["epochs_no_improve"])
            print(f"Resume state -> epoch={start_epoch}, best_acc={best_acc:.4f}, epochs_no_improve={epochs_no_improve}")

    ema_enabled = not config["ema_update_after_warmup"]
    remaining_warmups = max(0, config["head_warmup_epochs"] - start_epoch)
    best_path = RUN_LOGGER.fold_dir / f"best_fold{fold_idx}.pt"

    if remaining_warmups > 0:
        print(f"Head-only warmup for {remaining_warmups} epoch(s)…")
        freeze_backbone(model, freeze=True)
        for ep_offset in range(remaining_warmups):
            ep = start_epoch + ep_offset + 1; lr_current = optimizer.param_groups[0]["lr"]
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, criterion_train, optimizer, scaler, accum_steps,
                ema, ema_enabled, f"Fold {fold_idx} • Warmup {ep - start_epoch}/{remaining_warmups}",
                config["max_grad_norm"], mixup_fn
            )
            val_loss, val_acc, vpred, vtrue, vprobs = evaluate(
                model, val_loader, criterion_eval, ema=ema, use_ema=False,
                progress_desc=f"Fold {fold_idx} • Warmup Eval", tta_hflip=config["eval_tta_hflip"]
            )
            scheduler.step(ep-1)
            improved = val_acc > best_acc + 1e-4
            if improved:
                best_acc = val_acc; epochs_no_improve = 0
                _save_best_fold_now(model, ema, fold_idx, params, best_acc, note=f"warmup@E{ep}")
            else:
                epochs_no_improve += 1

            history.append({"epoch": ep, "phase": "warmup", "train_loss": tr_loss, "train_acc": tr_acc,
                            "val_loss": val_loss, "val_acc": val_acc, "lr": lr_current})
            RUN_LOGGER.log_epoch(ep, "warmup", tr_loss, tr_acc, val_loss, val_acc, lr_current, improved, best_acc, epochs_no_improve)
            RUN_LOGGER.save_epoch_cm(ep, vtrue, vpred, config["class_names"])
            _save_epoch_ckpt(ep, model, ema, optimizer, scheduler, scaler, fold_idx, trial_meta, "warmup", best_acc, epochs_no_improve)
            print(f"[Warmup {ep - start_epoch}/{remaining_warmups}] train_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f}")
        start_epoch += remaining_warmups

    freeze_backbone(model, freeze=False)
    ema_enabled = True if (ema is not None and config["ema_update_after_warmup"]) else ema_enabled

    stop_flag = False
    for epoch in range(start_epoch + 1, config["max_epochs"] + 1):
        lr_current = optimizer.param_groups[0]["lr"]
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion_train, optimizer, scaler, accum_steps,
            ema, ema_enabled, f"Fold {fold_idx} • Train E{epoch}",
            config["max_grad_norm"], mixup_fn
        )
        val_loss, val_acc, val_pred, val_true, _val_probs = evaluate(
            model, val_loader, criterion_eval, ema=ema, use_ema=(ema is not None),
            progress_desc=f"Fold {fold_idx} • Eval E{epoch}", tta_hflip=config["eval_tta_hflip"]
        )
        print(f"[Fold {fold_idx}] E{epoch:02d} | train_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f}")
        scheduler.step(epoch-1)

        improved = val_acc > best_acc + 1e-4
        if improved:
            best_acc = val_acc
            epochs_no_improve = 0
            _save_best_fold_now(model, ema, fold_idx, params, best_acc, note=f"train@E{epoch}")
        else:
            epochs_no_improve += 1

        history.append({"epoch": epoch, "phase": "train", "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": val_loss, "val_acc": val_acc, "lr": lr_current})
        RUN_LOGGER.log_epoch(epoch, "train", tr_loss, tr_acc, val_loss, val_acc, lr_current, improved, best_acc, epochs_no_improve)
        RUN_LOGGER.save_epoch_cm(epoch, val_true, val_pred, config["class_names"])
        _save_epoch_ckpt(epoch, model, ema, optimizer, scheduler, scaler, fold_idx, trial_meta, "train", best_acc, epochs_no_improve)

        if epochs_no_improve >= int(config["patience"]):
            print(f"Early stopping (no improve for {config['patience']} epochs).")
            stop_flag = True
            break

    # If nothing improved at all (rare), save current model as best to keep pipeline consistent
    if not (RUN_LOGGER.fold_dir / f"best_fold{fold_idx}.pt").exists():
        _save_best_fold_now(model, ema, fold_idx, params, max(0.0, best_acc), note="fallback_final")

    # History + diagnostic plots
    plot_history(history, fold_idx, RUN_LOGGER.fold_dir)

    # Load best weights for visualization and prediction dump
    best_ckpt = torch.load(RUN_LOGGER.fold_dir / f"best_fold{fold_idx}.pt", map_location=DEVICE)
    model_best = build_model()
    state = best_ckpt.get("model_ema") or best_ckpt.get("model")
    model_best.load_state_dict(state); model_best.eval()

    # Collect predictions on validation subset for plots
    pred_df = collect_predictions_df(model_best, val_df, batch_size=params.get("batch_size", config["batch_size"]), tta_hflip=config["eval_tta_hflip"])
    pred_csv = RUN_LOGGER.fold_dir / "val_predictions.csv"
    pred_df.to_csv(pred_csv, index=False); print("Saved:", pred_csv)

    # Confusion matrices
    plot_confusion_matrix_norm(pred_df["y_true"].values, pred_df["y_pred"].values,
                               title=f"Fold {fold_idx} CM (counts & norm)", out_path=RUN_LOGGER.fold_dir / "cm_norm.png")
    # ROC/PR
    probs = np.array(pred_df["probs"].tolist())
    if len(probs) and probs.shape[1] == NUM_CLASSES:
        plot_roc_pr_curves(pred_df["y_true"].values, probs, RUN_LOGGER.fold_dir / "ovr")
        # Reliability, scatter, margin, per-class dists
        plot_reliability(pred_df["y_true"].values, pred_df["y_pred"].values, pred_df["prob_pred"].values, bins=10, out_prefix=RUN_LOGGER.fold_dir / "calib")
        plot_confidence_accuracy_scatter(pred_df, RUN_LOGGER.fold_dir / "conf_acc_scatter.png")
        plot_margin_hist(pred_df, RUN_LOGGER.fold_dir / "margin_hist.png")
        plot_perclass_prob_dists(pred_df, RUN_LOGGER.fold_dir / "perclass_prob_dists.png")

        # Gain/Lift, DET/KS for each class
        for c in range(NUM_CLASSES):
            plot_cumulative_gain_and_lift(pred_df["y_true"].values, probs, c, RUN_LOGGER.fold_dir / f"cls{c}")
            plot_det_and_ks(pred_df["y_true"].values, probs, c, RUN_LOGGER.fold_dir / f"cls{c}")

    # Per-subject accuracy
    plot_subject_accuracy(val_df, pred_df, RUN_LOGGER.fold_dir / "subject_accuracy.png", config["subject_column"])
    # t-SNE (on val)
    plot_tsne_embeddings(
        model_best,
        val_df,
        max_samples=int(config["viz_tsne_max"]),
        out_prefix=RUN_LOGGER.fold_dir / "embeds"
    )

    # Saliency & attribution panels
    save_saliency_heatmaps(
        val_df,
        str(RUN_LOGGER.fold_dir / f"best_fold{fold_idx}.pt"),
        n=int(config["viz_saliency_count"])
    )
    save_comparison_panel(
        val_df,
        str(RUN_LOGGER.fold_dir / f"best_fold{fold_idx}.pt"),
        n=int(config["viz_saliency_count"])
    )
    save_smoothgrad_maps(
        val_df,
        str(RUN_LOGGER.fold_dir / f"best_fold{fold_idx}.pt"),
        n=int(config["viz_smoothgrad_count"]),
        noise_sigma=float(config["smoothgrad_noise_sigma"]),
        steps=int(config["smoothgrad_steps"])
    )
    save_rise_maps(
        val_df,
        str(RUN_LOGGER.fold_dir / f"best_fold{fold_idx}.pt"),
        n=int(config["viz_rise_count"]),
        n_masks=int(config["rise_masks"]),
        grid=int(config["rise_grid"]),
        p_keep=float(config["rise_p"])
    )

    # Quick predictions grid (random sample of val images)
    visualize_predictions_grid(
        val_df,
        str(RUN_LOGGER.fold_dir / f"best_fold{fold_idx}.pt"),
        n=int(config["viz_sample_count"]),
        title=f"Fold {fold_idx} Predictions"
    )

    # Lightweight HTML report of key artifacts
    write_simple_html_report(
        RUN_LOGGER.fold_dir,
        title=f"Trial {trial_meta['number']} • Fold {fold_idx} Report",
        image_globs=[
            "fold*_history.png", "fold*_lr.png",
            "cm_norm.png", "ovr_roc.png", "ovr_pr.png",
            "calib_reliability.png", "calib_conf_hist.png",
            "conf_acc_scatter.png", "margin_hist.png",
            "perclass_prob_dists.png", "subject_accuracy.png",
            "embeds_tsne_true.png", "embeds_tsne_pred.png",
            "topk_mistakes.png", "predictions_grid.png",
            "saliency_*.png", "compare_viz_*.png",
            "smoothgrad_*.png", "rise_*.png"
        ],
        out_name="report.html"
    )

    return float(best_acc)


# -----------------------------
# Extra plots: Gain/Lift and DET/KS (one-vs-rest)
# -----------------------------
def plot_cumulative_gain_and_lift(y_true: np.ndarray, probs: np.ndarray, cls_idx: int, out_prefix: Path):
    try:
        y = (y_true == cls_idx).astype(int)
        s = probs[:, cls_idx].astype(float)
        if len(y) == 0:
            return
        order = np.argsort(-s)
        y_sorted = y[order]
        cum_positives = np.cumsum(y_sorted)
        total_positives = max(1, y.sum())
        n = len(y)

        # Cumulative Gain
        pct_samples = np.arange(1, n + 1) / n
        gain = cum_positives / total_positives

        plt = ensure_matplotlib()
        fig = plt.figure(figsize=(6, 5))
        plt.plot(pct_samples, gain, label="Model")
        plt.plot([0, 1], [0, 1], "--", label="Baseline")
        plt.title(f"Cumulative Gain – class={config['class_names'][cls_idx]}")
        plt.xlabel("% of Samples (sorted by score)")
        plt.ylabel("% of Positives Captured")
        plt.legend()
        outp = out_prefix.parent / f"{out_prefix.name}_gain.png"
        fig.savefig(outp, dpi=150, bbox_inches="tight");
        plt.close(fig);
        print("Saved:", outp)

        # Lift = gain / baseline
        baseline = np.clip(pct_samples, 1e-9, 1.0)
        lift = gain / baseline
        fig = plt.figure(figsize=(6, 5))
        plt.plot(pct_samples, lift)
        plt.title(f"Lift Chart – class={config['class_names'][cls_idx]}")
        plt.xlabel("% of Samples (sorted by score)")
        plt.ylabel("Lift")
        outp = out_prefix.parent / f"{out_prefix.name}_lift.png"
        fig.savefig(outp, dpi=150, bbox_inches="tight");
        plt.close(fig);
        print("Saved:", outp)
    except Exception as e:
        print("Gain/Lift plot failed:", e)


def plot_det_and_ks(y_true: np.ndarray, probs: np.ndarray, cls_idx: int, out_prefix: Path):
    try:
        y = (y_true == cls_idx).astype(int)
        s = probs[:, cls_idx].astype(float)

        # Threshold sweep
        thresholds = np.unique(np.sort(s))
        if thresholds.size > 512:  # thin for speed
            thresholds = np.linspace(thresholds.min(), thresholds.max(), 512)

        P = max(1, y.sum())
        N = max(1, (y == 0).sum())
        TPRs, FPRs = [], []
        for t in thresholds:
            yhat = (s >= t).astype(int)
            TP = ((yhat == 1) & (y == 1)).sum()
            FP = ((yhat == 1) & (y == 0)).sum()
            FN = ((yhat == 0) & (y == 1)).sum()
            TN = ((yhat == 0) & (y == 0)).sum()
            TPR = TP / P
            FPR = FP / N
            TPRs.append(TPR);
            FPRs.append(FPR)

        TPRs = np.array(TPRs);
        FPRs = np.array(FPRs)
        FNRs = 1.0 - TPRs

        # DET (FPR vs FNR)
        plt = ensure_matplotlib()
        fig = plt.figure(figsize=(6, 5))
        plt.plot(FPRs, FNRs)
        plt.title(f"DET Curve – class={config['class_names'][cls_idx]}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("False Negative Rate")
        outp = out_prefix.parent / f"{out_prefix.name}_det.png"
        fig.savefig(outp, dpi=150, bbox_inches="tight");
        plt.close(fig);
        print("Saved:", outp)

        # KS statistic
        ks = np.max(np.abs(TPRs - FPRs))
        fig = plt.figure(figsize=(6, 4))
        plt.plot(thresholds, TPRs, label="TPR")
        plt.plot(thresholds, FPRs, label="FPR")
        plt.title(f"KS Curve – KS={ks:.3f} – class={config['class_names'][cls_idx]}")
        plt.xlabel("Threshold");
        plt.ylabel("Rate")
        plt.legend()
        outp = out_prefix.parent / f"{out_prefix.name}_ks.png"
        fig.savefig(outp, dpi=150, bbox_inches="tight");
        plt.close(fig);
        print("Saved:", outp)
    except Exception as e:
        print("DET/KS plot failed:", e)


# -----------------------------
# Folds preparation
# -----------------------------
def make_folds(df: pd.DataFrame):
    n_folds = int(config["n_folds"])
    use_subject = bool(config["use_subject_splits"])
    label_idx = df["label_idx"].values
    if use_subject:
        groups = df[config["subject_column"]].astype(str).values
        gkf = GroupKFold(n_splits=n_folds)
        splits = list(gkf.split(df, label_idx, groups))
    else:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        splits = list(skf.split(df, label_idx))
    return splits


# -----------------------------
# Optuna Objective
# -----------------------------
def objective(trial: optuna.trial.Trial, df: pd.DataFrame) -> float:
    # Suggest params within 8 GB-friendly envelope
    lr = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    bs_options = [8, 12, 16, 24] if DEVICE != "cuda" else [12, 16, 24, 32]
    bs = trial.suggest_categorical("batch_size", bs_options)

    params = {
        "learning_rate": float(lr),
        "batch_size": int(bs)
    }

    trial_dir = RUN_LOGGER.start_trial(trial.number, params)
    splits = make_folds(df)

    fold_accs = []
    for fold_idx, (tr_idx, va_idx) in enumerate(splits, start=1):
        RUN_LOGGER.start_fold(fold_idx)
        train_df = df.iloc[tr_idx].reset_index(drop=True)
        val_df = df.iloc[va_idx].reset_index(drop=True)
        acc = run_fold(train_df, val_df, params, fold_idx=fold_idx, trial_meta={"number": trial.number})
        fold_accs.append(acc)

    mean_acc = float(np.mean(fold_accs)) if len(fold_accs) else 0.0
    RUN_LOGGER.write_trial_summary(
        {"trial": trial.number, "params": params, "fold_accs": fold_accs, "mean_acc": mean_acc})
    return mean_acc


# -----------------------------
# Main
# -----------------------------
def main():
    df = load_metadata(IMG_ROOT)

    # Ensure Optuna storage path is absolute under BASE
    storage_path = BASE / config["optuna_storage_path"]
    storage_url = f"sqlite:///{storage_path}"

    if config["do_optuna"]:
        study_name = str(config["optuna_study_name"])
        # Create/re-use study
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            load_if_exists=True,
            direction="maximize"
        )
        print(f"[Optuna] Study: {study.study_name} @ {storage_path}")
        study.optimize(lambda t: objective(t, df), n_trials=int(config["n_trials"]))
        print("[Optuna] Best value:", study.best_value)
        print("[Optuna] Best params:", study.best_params)

        # Persist plots & CSV at ROOT_SAVE_DIR
        save_optuna_plots(study, ROOT_SAVE_DIR)

        # Also export best trial's fold best weights to top-level convenience folder
        best_trial = study.best_trial
        best_dir = ROOT_SAVE_DIR / f"trial_{best_trial.number:03d}"
        if best_dir.exists():
            # Copy best_fold*.pt to ROOT_SAVE_DIR/best/
            best_out = ROOT_SAVE_DIR / "best"
            ensure_dir(best_out)
            for bf in sorted(best_dir.rglob("best_fold*.pt")):
                shutil.copy2(bf, best_out / bf.name)
            print("Copied best fold weights to:", best_out)
    else:
        # Single run without tuning (use config base_lr/batch_size)
        params = {"learning_rate": float(config["base_lr"]), "batch_size": int(config["batch_size"])}
        trial_no = RUN_LOGGER.find_latest_trial_number()
        trial_no = 0 if trial_no is None else trial_no + 1
        RUN_LOGGER.start_trial(trial_no, params)
        splits = make_folds(df)
        fold_accs = []
        for fold_idx, (tr_idx, va_idx) in enumerate(splits, start=1):
            RUN_LOGGER.start_fold(fold_idx)
            train_df = df.iloc[tr_idx].reset_index(drop=True)
            val_df = df.iloc[va_idx].reset_index(drop=True)
            acc = run_fold(train_df, val_df, params, fold_idx=fold_idx, trial_meta={"number": trial_no})
            fold_accs.append(acc)
        summary = {"trial": trial_no, "params": params, "fold_accs": fold_accs, "mean_acc": float(np.mean(fold_accs))}
        RUN_LOGGER.write_trial_summary(summary)
        print("Summary:", summary)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
