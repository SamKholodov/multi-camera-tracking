"""Training loop for ContactPointRegressor."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import ContactPointDataset
from .metrics import baseline_uv_like, contact_metrics
from .model import ContactPointRegressor


LOGGER = logging.getLogger("core.geometry.contact_point")


def setup_logger(save_dir: Path) -> None:
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    LOGGER.addHandler(stream)
    file_handler = logging.FileHandler(save_dir / "training.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    LOGGER.addHandler(file_handler)


class ContactPointTrainer:
    def __init__(self, *, config: dict[str, Any]):
        self.config = config
        train_cfg = config.get("train", {})
        self.output_dir = Path(config.get("output_dir", "datasets/gta_mcmt/contact_point"))
        self.manifest_path = self.output_dir / "manifest.jsonl"
        self.splits_path = self.output_dir / "splits.json"
        self.project = Path(config.get("project", "runs/contact_point"))
        self.name = config.get("name", "mobilenetv3_small_gta")
        self.save_dir = self._resolve_save_dir(self.project, self.name)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        setup_logger(self.save_dir)
        self.device = torch.device(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.img_size = int(train_cfg.get("img_size", 224))
        self.batch_size = int(train_cfg.get("batch_size", 64))
        self.epochs = int(train_cfg.get("epochs", 50))
        self.lr = float(train_cfg.get("lr", 1e-3))
        self.num_workers = int(train_cfg.get("num_workers", 4))
        self.bbox_pad_ratio = float(train_cfg.get("bbox_pad_ratio", 0.05))
        self.require_viz = bool(train_cfg.get("require_viz", True))

    @staticmethod
    def _resolve_save_dir(project: Path, name: str) -> Path:
        base = project / name
        if not base.exists():
            return base
        i = 1
        while (project / f"{name}_{i}").exists():
            i += 1
        return project / f"{name}_{i}"

    def _check_viz_gate(self) -> None:
        if not self.require_viz:
            return
        viz_dir = self.output_dir / "debug" / "viz"
        count = len(list(viz_dir.glob("*.png"))) if viz_dir.is_dir() else 0
        if count < 50:
            raise RuntimeError(f"Expected at least 50 debug PNGs in {viz_dir}; run prepare with --viz-count first.")

    def _loader(self, split: str, train: bool) -> DataLoader:
        ds = ContactPointDataset(
            self.manifest_path,
            split=split,
            splits_path=self.splits_path,
            img_size=self.img_size,
            train=train,
            bbox_pad_ratio=self.bbox_pad_ratio,
        )
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
        )

    def run(self) -> Path:
        self._check_viz_gate()
        (self.save_dir / "hparams.json").write_text(json.dumps(self.config, indent=2), encoding="utf-8")
        model = ContactPointRegressor(pretrained=True).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, self.epochs))
        loss_fn = nn.SmoothL1Loss()
        train_loader = self._loader("train", train=True)
        val_loader = self._loader("val", train=False)
        sanity_loader = self._loader("sanity_cam_holdout", train=False)

        best_metric = float("inf")
        history: list[dict] = []
        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(model, train_loader, optimizer, loss_fn)
            val = self._evaluate(model, val_loader, loss_fn)
            sanity = self._evaluate(model, sanity_loader, loss_fn) if len(sanity_loader.dataset) else None
            scheduler.step()
            row = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train": {"model": {"loss": train_loss}},
                "val": val,
                "sanity_cam_holdout": sanity,
            }
            history.append(row)
            (self.save_dir / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            LOGGER.info(
                "epoch=%d train_loss=%.6f val_pixel_mae=%.4f baseline_pixel_mae=%.4f",
                epoch,
                train_loss,
                val["model"]["pixel_mae"],
                val["baseline"]["pixel_mae"],
            )
            if val["model"]["pixel_mae"] > val["baseline"]["pixel_mae"]:
                LOGGER.warning("Model is worse than bottom-center baseline on val.")
            self._save_checkpoint(self.save_dir / "last.pth", model, optimizer, scheduler, epoch, best_metric)
            if val["model"]["pixel_mae"] < best_metric:
                best_metric = val["model"]["pixel_mae"]
                self._save_checkpoint(self.save_dir / "best.pth", model, optimizer, scheduler, epoch, best_metric)
        return self.save_dir / "best.pth"

    def _train_epoch(self, model, loader, optimizer, loss_fn) -> float:
        model.train()
        total = 0.0
        count = 0
        for batch in loader:
            images = batch["image"].to(self.device)
            target = batch["target_uv"].to(self.device)
            pred = model(images)
            loss = loss_fn(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * images.size(0)
            count += images.size(0)
        return total / max(count, 1)

    @torch.no_grad()
    def _evaluate(self, model, loader, loss_fn) -> dict[str, dict[str, float]]:
        model.eval()
        pred_all: list[torch.Tensor] = []
        target_all: list[torch.Tensor] = []
        bbox_all: list[torch.Tensor] = []
        losses = []
        for batch in loader:
            images = batch["image"].to(self.device)
            target = batch["target_uv"].to(self.device)
            bbox_wh = batch["bbox_wh"].to(self.device)
            pred = model(images)
            losses.append(float(loss_fn(pred, target).item()) * images.size(0))
            pred_all.append(pred.cpu())
            target_all.append(target.cpu())
            bbox_all.append(bbox_wh.cpu())
        if not pred_all:
            empty = {"loss": float("nan"), "mae_u": float("nan"), "mae_v": float("nan"), "pixel_mae": float("nan")}
            return {"model": empty, "baseline": empty, "delta_pixel_mae": float("nan")}
        pred_uv = torch.cat(pred_all)
        target_uv = torch.cat(target_all)
        bbox_wh = torch.cat(bbox_all)
        baseline = baseline_uv_like(target_uv)
        model_metrics = contact_metrics(pred_uv, target_uv, bbox_wh)
        base_metrics = contact_metrics(baseline, target_uv, bbox_wh)
        model_metrics["loss"] = sum(losses) / max(len(target_uv), 1)
        base_metrics["loss"] = float(loss_fn(baseline, target_uv).item())
        return {
            "model": model_metrics,
            "baseline": base_metrics,
            "delta_pixel_mae": model_metrics["pixel_mae"] - base_metrics["pixel_mae"],
        }

    def _save_checkpoint(self, path: Path, model, optimizer, scheduler, epoch: int, best_metric: float) -> None:
        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_pixel_mae": best_metric,
                "img_size": self.img_size,
                "config": self.config,
            },
            path,
        )
