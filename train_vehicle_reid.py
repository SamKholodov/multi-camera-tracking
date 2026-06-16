from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from core.reid.training.vehicle_trainer import VehicleReIDTrainer


def _load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _as_tuple(value, default):
    if value is None:
        return default
    return tuple(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train dual-head vehicle OSNet.")
    parser.add_argument(
        "--config",
        default="config/reid_train/vehicle_osnet.yaml",
        help="Path to the training YAML config.",
    )
    parser.add_argument("--device", default=None, help="Override device from config.")
    parser.add_argument("--data-dir", default=None, help="Override dataset root.")
    parser.add_argument("--name", default=None, help="Override run name.")
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to last.pth (or other training checkpoint) to resume.",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    if args.device is not None:
        cfg["device"] = args.device
    if args.data_dir is not None:
        cfg["data_dir"] = args.data_dir
    if args.name is not None:
        cfg["name"] = args.name
    if args.resume is not None:
        cfg["resume"] = args.resume

    trainer = VehicleReIDTrainer(
        model_name=cfg.get("model", "vehicle_osnet_x1_0"),
        datasets=cfg.get("datasets", ["veri", "vric"]),
        data_dir=cfg.get("data_dir", "reid_datasets"),
        pretrained_path=cfg.get("pretrained_path", "osnet_x1_0_imagenet.pth"),
        img_size=_as_tuple(cfg.get("img_size"), (128, 256)),
        preprocess=cfg.get("preprocess", "pad_ratio_resize"),
        num_view_classes=cfg.get("num_view_classes", 8),
        view_layers=cfg.get("view_layers"),
        view_layer_weights=cfg.get("view_layer_weights"),
        lambda_id=cfg.get("lambda_id", 1.0),
        lambda_triplet=cfg.get("lambda_triplet", 1.0),
        lambda_view=cfg.get("lambda_view", 0.2),
        checkpoint=cfg.get("checkpoint"),
        resume=cfg.get("resume"),
        freeze_backbone=cfg.get("freeze_backbone", False),
        freeze_reid_heads=cfg.get("freeze_reid_heads", False),
        best_metric=cfg.get("best_metric", "veri_mAP"),
        eval_ranking=cfg.get("eval_ranking", True),
        eval_interval=cfg.get("eval_interval", 1),
        p=cfg.get("p", 16),
        k=cfg.get("k", 4),
        fallback_p=cfg.get("fallback_p", 8),
        lr=cfg.get("lr", 3.5e-4),
        weight_decay=cfg.get("weight_decay", 5e-4),
        epochs=cfg.get("epochs", 120),
        warmup_epochs=cfg.get("warmup_epochs", 10),
        label_smooth=cfg.get("label_smooth", 0.1),
        margin=cfg.get("margin", 0.3),
        val_fraction=cfg.get("val_fraction", 0.1),
        batch_size=cfg.get("batch_size"),
        num_workers=cfg.get("num_workers", 4),
        seed=cfg.get("seed", 42),
        device=cfg.get("device", "cpu"),
        project=cfg.get("project", "runs/vehicle_reid"),
        name=cfg.get("name", "exp"),
        log_file=cfg.get("log_file", "osnet_training.log"),
        config=cfg,
    )
    trainer.run()


if __name__ == "__main__":
    main()
