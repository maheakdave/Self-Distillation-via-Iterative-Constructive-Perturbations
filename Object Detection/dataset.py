from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
import torch
from torchvision.transforms import v2
from torchvision import tv_tensors,datasets
from config import Config


def build_loader():
    cfg = Config()

    train_transforms = v2.Compose([
        v2.ToImage(),
        v2.RandomPhotometricDistort(p=1),
        v2.RandomZoomOut(fill={tv_tensors.Image: (123, 117, 104), "others": 0}),
        v2.RandomIoUCrop(),
        v2.RandomHorizontalFlip(p=1),
        v2.SanitizeBoundingBoxes(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(cfg.size),                  
        v2.Normalize(mean=[0.485, 0.456, 0.406],
             std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(cfg.size),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
             std=[0.229, 0.224, 0.225]),
    ])

    
    train_raw = CocoDetection(
        root= cfg.TRAIN_PATH[0],
        annFile=cfg.TRAIN_PATH[1]
    )
    val_raw   = CocoDetection(
        root=cfg.VAL_PATH[0],
        annFile=cfg.VAL_PATH[1]
    )

    train_ds = datasets.wrap_dataset_for_transforms_v2(
        train_raw,
        target_keys=["boxes", "labels"],
    )
    val_ds = datasets.wrap_dataset_for_transforms_v2(
        val_raw,
        target_keys=["boxes", "labels","image_id"],
    )

    train_ds.transforms = train_transforms
    val_ds.transforms   = val_transforms

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader

def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    targets = [b[1] for b in batch]
    return imgs, targets
