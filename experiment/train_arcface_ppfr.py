import argparse
import math
import os
import random
import time
import warnings

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from experiment.models.arcface_backbone import Backbone
from ldp_slicing import dp_slicing_dwt, get_privacy_budget

try:
    # Optional DCT privacy method (not included in this repo).
    from dp_slicing_batch import dp_slicing_perceptual  # type: ignore
except ImportError:  # pragma: no cover
    dp_slicing_perceptual = None

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.",
    category=UserWarning,
)


def resolve_ablation_eps(ablation: str, epsilon: float, color_weight: str):
    if ablation == "no_ll":
        eps_y, eps_c, _, budget_key = get_privacy_budget(color_weight, epsilon)
        return False, 1.0, eps_y, eps_c, "_no_ll", budget_key
    if ablation == "uniform":
        return True, 1e-5, float(epsilon), float(epsilon), "_uniform", "uniform"
    if ablation == "dynamic":
        dynamic_budget = (0.184, 0.260, 0.368, 0.521, 0.736, 1.041, 1.473, 2.083)
        return True, 1e-5, dynamic_budget, dynamic_budget, "_dynamic", "dynamic"

    eps_y, eps_c, _, budget_key = get_privacy_budget(color_weight, epsilon)
    suffix = "" if ablation == "lagrangian" else "_remove_ll"
    return True, 1e-5, eps_y, eps_c, suffix, budget_key


def warmup_lr_scheduler(optimizer, epoch, warmup_epochs, warmup_start_lr, target_lr):
    if epoch < warmup_epochs:
        lr = warmup_start_lr + (target_lr - warmup_start_lr) * epoch / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr
    return None


class FileListDataset(Dataset):
    def __init__(self, root_dir, file_list_path):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
        ])
        with open(file_list_path, "r", encoding="utf-8") as f:
            self.samples = [line.strip().split() for line in f.readlines() if line.strip()]
        self.num_classes = max(int(s[1]) for s in self.samples) + 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label_str = self.samples[index]
        full_img_path = os.path.join(self.root_dir, img_path)
        label = int(label_str)
        with Image.open(full_img_path).convert("RGB") as img:
            tensor_img = self.transform(img)
        return tensor_img, label


class ArcFaceHead(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.5):
        super().__init__()
        self.s = s
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.clamp(cosine.pow(2), 0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s


def protect_batch(inputs, args, epsilon_y, epsilon_c, remove_ll, ll_scale, device):
    with torch.no_grad():
        if args.dp_method == "dwt":
            return dp_slicing_dwt(
                inputs,
                wavelet=args.wavelet,
                level=1,
                remove_ll=remove_ll,
                ll_scale=ll_scale,
                epsilon_y=epsilon_y,
                epsilon_c=epsilon_c,
                device=device,
            )
        if args.dp_method == "dct":
            if dp_slicing_perceptual is None:
                raise RuntimeError(
                    "DCT method requires dp_slicing_batch.dp_slicing_perceptual, "
                    "but that module is not present."
                )
            return dp_slicing_perceptual(
                inputs,
                chs_remove=[0],
                epsilon_y=epsilon_y,
                epsilon_c=epsilon_c,
                device=device,
            )
        raise ValueError(f"Unknown dp_method: {args.dp_method}")


def train_worker(local_rank, world_size, args):
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    remove_ll, ll_scale, epsilon_y, epsilon_c, ablation_suffix, budget_key = resolve_ablation_eps(
        args.ablation, args.epsilon, args.color_weight
    )
    method_suffix = f"{args.dp_method}_{args.wavelet}" if args.dp_method == "dwt" else args.dp_method
    output_dir = os.path.join("checkpoint", f"arcface_{method_suffix}_eps{args.epsilon}{ablation_suffix}")

    log_f = None
    if local_rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        log_f = open(os.path.join(output_dir, "training_log.txt"), "w", encoding="utf-8")
        print("=" * 80)
        print("ARCFACE TRAINING WITH DP-SLICING")
        print("=" * 80)
        print(f"Method: {args.dp_method.upper()} | Wavelet: {args.wavelet}")
        print(f"Ablation: {args.ablation} | Budget table: {budget_key}")
        print(f"epsilon_y: {epsilon_y}")
        print(f"epsilon_c: {epsilon_c}")
        print("=" * 80)

    dataset = FileListDataset(root_dir=args.data_root, file_list_path=args.file_list)
    num_classes = dataset.num_classes
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=max(1, args.batch_size // world_size),
        num_workers=args.num_workers,
        sampler=sampler,
        pin_memory=True,
        drop_last=True,
    )

    backbone = Backbone(input_size=(112, 112), num_layers=50, mode="ir").to(device)
    head = ArcFaceHead(512, num_classes, s=args.arcface_s, m=args.arcface_m).to(device)
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        checkpoint = torch.load(args.pretrained_path, map_location=device)
        backbone.load_state_dict(checkpoint, strict=False)

    backbone = DDP(backbone, device_ids=[local_rank])
    head = DDP(head, device_ids=[local_rank])
    optimizer = optim.SGD(
        [{"params": backbone.parameters()}, {"params": head.parameters()}],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.lr_milestones,
        gamma=args.lr_gamma,
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        backbone.train()
        head.train()

        if epoch < args.warmup_epochs:
            current_lr = warmup_lr_scheduler(
                optimizer,
                epoch,
                args.warmup_epochs,
                args.warmup_start_lr,
                args.lr,
            )
        else:
            current_lr = optimizer.param_groups[0]["lr"]

        epoch_start = time.time()
        total_loss = 0.0
        num_batches = 0
        skipped_batches = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}", disable=(local_rank != 0))

        for inputs, labels in pbar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            protected_imgs = protect_batch(
                inputs, args, epsilon_y, epsilon_c, remove_ll, ll_scale, device
            )
            if torch.isnan(protected_imgs).any():
                skipped_batches += 1
                continue

            inputs_normalized = (protected_imgs - 0.5) * 2.0
            features = backbone(inputs_normalized)
            outputs = head(features, labels)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            if torch.isnan(loss):
                skipped_batches += 1
                continue
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            if local_rank == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.6f}")

        if epoch >= args.warmup_epochs:
            scheduler.step()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        epoch_time = time.time() - epoch_start
        if local_rank == 0:
            summary = (
                f"Epoch {epoch + 1}/{args.epochs} | Avg Loss: {avg_loss:.4f} | "
                f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s | "
                f"Batches: {num_batches} | Skipped: {skipped_batches}"
            )
            print(summary)
            log_f.write(summary + "\n")
            log_f.flush()

            checkpoint = {
                "epoch": epoch + 1,
                "backbone_state_dict": backbone.module.state_dict(),
                "head_state_dict": head.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "avg_loss": avg_loss,
                "args": vars(args),
            }
            torch.save(checkpoint, os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1}.pth"))

    if local_rank == 0 and log_f is not None:
        log_f.close()
        print("Training completed.")
    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description="Train ArcFace with DP-Slicing (PPFR)")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--file_list", type=str, required=True)
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--dp_method", type=str, default="dwt", choices=["dct", "dwt"])
    parser.add_argument("--wavelet", type=str, default="haar")
    parser.add_argument("--epsilon", type=float, required=True, choices=[1.0, 2.4, 5.2, 12.0, 20.0, 32.0, 58.0])
    parser.add_argument("--color_weight", type=str, default="411", choices=["411", "211", "111"])
    parser.add_argument("--ablation", type=str, default="lagrangian", choices=["lagrangian", "remove_ll", "no_ll", "uniform", "dynamic"])
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_start_lr", type=float, default=1e-5)
    parser.add_argument("--lr_milestones", type=int, nargs="+", default=[10, 18, 22])
    parser.add_argument("--lr_gamma", type=float, default=0.1)
    parser.add_argument("--arcface_s", type=float, default=64.0)
    parser.add_argument("--arcface_m", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--world_size", type=int, default=max(1, torch.cuda.device_count()))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(random.randint(10000, 60000))
    torch.multiprocessing.spawn(
        train_worker,
        args=(args.world_size, args),
        nprocs=args.world_size,
        join=True,
    )

