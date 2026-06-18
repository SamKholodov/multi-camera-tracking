from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from core.geometry.contact_point.trainer import ContactPointTrainer


def _load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train contact point regressor.")
    parser.add_argument(
        "--config",
        default="config/contact_point/mobilenetv3_gta.yaml",
        help="Path to training YAML config.",
    )
    parser.add_argument("--device", default=None, help="Override train.device.")
    parser.add_argument("--output-dir", default=None, help="Override prepared dataset dir.")
    parser.add_argument("--name", default=None, help="Override run name.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    cfg = _load_config(args.config)
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.name is not None:
        cfg["name"] = args.name
    train_cfg = dict(cfg.get("train", {}))
    if args.device is not None:
        train_cfg["device"] = args.device
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    cfg["train"] = train_cfg

    trainer = ContactPointTrainer(config=cfg)
    best_path = trainer.run()
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
