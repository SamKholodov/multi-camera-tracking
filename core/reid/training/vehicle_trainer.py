"""Vehicle dual-head OSNet training."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from core.reid.backbones.vehicle_osnet import (
    normalize_view_layer_weights,
    normalize_view_layers,
    remap_vehicle_view_state_dict,
)
from core.reid.core.registry import ReIDModelRegistry
from core.reid.datasets import build_dataset
from core.reid.datasets.base import DatasetSplit, ReIDSample
from core.reid.datasets.sampler import PKSampler, count_eligible_pids, resolve_pk
from core.reid.datasets.torch_dataset import ReIDImageDataset
from core.reid.datasets.transforms import (
    ViewAwareHorizontalFlip,
    build_test_transforms,
    build_train_transforms,
)
from core.reid.training.evaluator import (
    compute_distance_matrix,
    evaluate_ranking,
    extract_features,
)
from core.reid.training.losses import CrossEntropyLabelSmooth, TripletLoss
from core.reid.utils import logger as LOGGER


@dataclass
class VehicleTrainMetrics:
    epoch: int
    lr: float
    train_loss_total: float
    train_loss_id: float
    train_loss_triplet: float
    train_loss_view: float


@dataclass
class VehicleValMetrics:
    dataset: str
    val_loss_total: float = math.nan
    val_loss_id: float = math.nan
    val_loss_triplet: float = math.nan
    val_loss_view: float = math.nan
    val_view_acc: float = math.nan
    val_mAP: float = 0.0
    val_rank1: float = 0.0


@dataclass
class VehicleTrainResult:
    best_epoch: int
    best_mAP: float
    weights_path: Path
    history: List[VehicleTrainMetrics] = field(default_factory=list)
    val_history: List[Dict[str, VehicleValMetrics]] = field(default_factory=list)


class VehicleReIDTrainer:
    """Train OSNet for vehicle ReID and 8-way view classification."""

    def __init__(
        self,
        *,
        model_name: str = "vehicle_osnet_x1_0",
        datasets: List[str] | Tuple[str, ...] = ("veri", "vric"),
        data_dir: str = "reid_datasets",
        pretrained_path: str | None = "osnet_x1_0_imagenet.pth",
        img_size: Tuple[int, int] = (128, 256),
        preprocess: str = "pad_ratio_resize",
        num_view_classes: int = 8,
        view_layers: List[str] | Tuple[str, ...] | str | None = None,
        view_layer_weights: dict[str, float] | None = None,
        lambda_id: float = 1.0,
        lambda_triplet: float = 1.0,
        lambda_view: float = 0.2,
        checkpoint: str | None = None,
        resume: str | None = None,
        freeze_backbone: bool = False,
        freeze_reid_heads: bool = False,
        best_metric: str = "veri_mAP",
        eval_ranking: bool = True,
        eval_interval: int = 1,
        p: int = 16,
        k: int = 4,
        fallback_p: int = 8,
        lr: float = 3.5e-4,
        weight_decay: float = 5e-4,
        epochs: int = 120,
        warmup_epochs: int = 10,
        label_smooth: float = 0.1,
        margin: float = 0.3,
        val_fraction: float = 0.1,
        batch_size: Optional[int] = None,
        num_workers: int = 4,
        seed: int = 42,
        device: str = "cpu",
        project: str = "runs/vehicle_reid",
        name: str = "exp",
        log_file: str = "osnet_training.log",
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        self.model_name = model_name
        self.datasets = list(datasets)
        self.data_dir = data_dir
        self.pretrained_path = pretrained_path
        self.img_size = img_size
        self.preprocess = preprocess
        self.num_view_classes = num_view_classes
        self.view_layers = normalize_view_layers(view_layers)
        self.view_layer_weights = normalize_view_layer_weights(
            view_layer_weights,
            self.view_layers,
        )
        self.lambda_id = lambda_id
        self.lambda_triplet = lambda_triplet
        self.lambda_view = lambda_view
        self.checkpoint = checkpoint
        self.resume = resume
        self.freeze_backbone = freeze_backbone
        self.freeze_reid_heads = freeze_reid_heads
        self.best_metric = best_metric
        self.eval_ranking = eval_ranking
        self.eval_interval = max(int(eval_interval), 1)
        self.p = p
        self.k = k
        self.fallback_p = fallback_p
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.label_smooth = label_smooth
        self.margin = margin
        self.val_fraction = val_fraction
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.device = torch.device(device)
        self.project = Path(project)
        self.name = name
        self.log_file = log_file
        self.config = config or {}

    def run(self) -> VehicleTrainResult:
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        save_dir = self._make_save_dir()
        self._configure_file_logger(save_dir)

        (
            train_samples,
            val_samples_by_dataset,
            ranking_datasets,
            num_classes,
        ) = self._prepare_datasets()
        effective_p, effective_k = self._resolve_pk(train_samples)
        train_batch_size = effective_p * effective_k
        eval_batch_size = self.batch_size or train_batch_size

        model_num_classes = num_classes
        resume_path = Path(self.resume) if self.resume else None
        weight_ckpt_path = resume_path or (Path(self.checkpoint) if self.checkpoint else None)
        if weight_ckpt_path is not None:
            ckpt_meta = torch.load(
                weight_ckpt_path, map_location="cpu", weights_only=False
            )
            if ckpt_meta.get("num_classes") is not None:
                model_num_classes = int(ckpt_meta["num_classes"])
                if model_num_classes != num_classes:
                    LOGGER.info(
                        f"Using num_classes={model_num_classes} from checkpoint "
                        f"(dataset split has {num_classes})."
                    )

        model = self._build_model(model_num_classes).to(self.device)
        if self.checkpoint and not self.resume:
            self._load_checkpoint(model, self.checkpoint)
        self._apply_freeze(model)

        criterion_id = CrossEntropyLabelSmooth(num_classes, epsilon=self.label_smooth)
        criterion_triplet = TripletLoss(margin=self.margin, soft_margin=False)
        criterion_view = nn.CrossEntropyLoss()

        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("No trainable parameters after freeze/checkpoint setup.")
        LOGGER.info(
            f"Trainable parameters: {sum(p.numel() for p in trainable):,} / "
            f"{sum(p.numel() for p in model.parameters()):,}"
        )

        optimizer = torch.optim.Adam(
            trainable,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(self.epochs - self.warmup_epochs, 1),
            eta_min=1e-7,
        )
        for group in optimizer.param_groups:
            group["_base_lr"] = group["lr"]
            if self.warmup_epochs > 0:
                group["lr"] = group["lr"] / self.warmup_epochs

        train_loader = self._build_train_loader(train_samples, effective_p, effective_k)
        val_loss_loaders = {
            name: self._build_loss_loader(samples, eval_batch_size)
            for name, samples in val_samples_by_dataset.items()
            if samples and name in self.datasets
        }
        ranking_loaders = {
            name: self._build_ranking_loaders(ds, eval_batch_size)
            for name, ds in ranking_datasets.items()
        }

        hparams = self._hparams(num_classes, effective_p, effective_k, train_batch_size)
        (save_dir / "hparams.json").write_text(json.dumps(hparams, indent=2))

        best_epoch = 0
        best_score = float("-inf")
        best_path = save_dir / "best.pth"
        history: List[VehicleTrainMetrics] = []
        val_history: List[Dict[str, VehicleValMetrics]] = []
        start_epoch = 1

        if self.resume:
            start_epoch, best_epoch, best_score, history, val_history = (
                self._load_resume_state(
                    Path(self.resume),
                    model,
                    optimizer,
                    scheduler,
                    save_dir,
                )
            )
            if start_epoch > self.epochs:
                LOGGER.info(
                    f"Checkpoint epoch {start_epoch - 1} >= target epochs "
                    f"{self.epochs}; nothing to resume."
                )
                return VehicleTrainResult(
                    best_epoch=best_epoch,
                    best_mAP=best_score,
                    weights_path=best_path,
                    history=history,
                    val_history=val_history,
                )

        for epoch in range(start_epoch, self.epochs + 1):
            train_metrics = self._train_epoch(
                epoch,
                model,
                train_loader,
                criterion_id,
                criterion_triplet,
                criterion_view,
                optimizer,
                scheduler,
            )
            history.append(train_metrics)

            run_full_eval = self._should_run_full_eval(epoch)
            if run_full_eval:
                epoch_vals = self._validate_all(
                    model,
                    val_loss_loaders,
                    ranking_loaders,
                    criterion_id,
                    criterion_triplet,
                    criterion_view,
                )
                val_history.append(epoch_vals)

                epoch_score = self._selection_score(epoch_vals)
                if epoch_score > best_score:
                    best_score = epoch_score
                    best_epoch = epoch
                    self._save_checkpoint(
                        best_path,
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        best_score,
                        hparams,
                    )
            else:
                epoch_vals = {}
                val_history.append(epoch_vals)
                LOGGER.info(
                    f"Skipping full validation at epoch {epoch} "
                    f"(eval_interval={self.eval_interval})"
                )

            self._save_checkpoint(
                save_dir / "last.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                best_score,
                hparams,
            )
            if epoch % 5 == 0:
                self._save_checkpoint(
                    save_dir / f"epoch_{epoch}.pth",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_score,
                    hparams,
                )
            self._write_epoch_log(
                train_metrics,
                epoch_vals,
                skip_val=not run_full_eval,
            )
            self._save_metrics(save_dir, history, val_history, best_epoch, best_score)

        return VehicleTrainResult(
            best_epoch=best_epoch,
            best_mAP=best_score,
            weights_path=best_path,
            history=history,
            val_history=val_history,
        )

    def _prepare_datasets(
        self,
    ) -> tuple[list[ReIDSample], dict[str, list[ReIDSample]], dict[str, Any], int]:
        train_samples: list[ReIDSample] = []
        val_samples_by_dataset: dict[str, list[ReIDSample]] = {}
        ranking_datasets: dict[str, Any] = {}
        pid_offset = 0
        cam_offset = 0
        source_id_by_name = {name: idx for idx, name in enumerate(self.datasets)}

        for name in self.datasets:
            ds = build_dataset(name, self.data_dir)
            source_id = source_id_by_name[name]
            train_split, val_split = ds.train.split_by_pid(self.val_fraction, self.seed)
            train_samples.extend(
                self._offset_samples(train_split.samples, pid_offset, cam_offset, name, source_id)
            )
            val_samples_by_dataset[name] = self._offset_samples(
                val_split.samples,
                pid_offset,
                cam_offset,
                name,
                source_id,
            )
            ranking_datasets[name] = ds
            pid_offset += ds.train.num_pids
            cam_offset += ds.train.num_cams

        return train_samples, val_samples_by_dataset, ranking_datasets, pid_offset

    @staticmethod
    def _offset_samples(
        samples: list[ReIDSample],
        pid_offset: int,
        cam_offset: int,
        source: str,
        source_id: int,
    ) -> list[ReIDSample]:
        return [
            ReIDSample(
                img_path=s.img_path,
                pid=s.pid + pid_offset,
                camid=s.camid + cam_offset,
                view_id=s.view_id,
                has_view=s.has_view,
                source=source,
                source_id=source_id,
            )
            for s in samples
        ]

    def _resolve_pk(self, train_samples: list[ReIDSample]) -> tuple[int, int]:
        eligible = count_eligible_pids(train_samples, self.k)
        p, k = resolve_pk(eligible, self.p, self.k, self.fallback_p)
        if p != self.p:
            LOGGER.info(
                f"Using fallback PK sampler: eligible_pids={eligible}, P={p}, K={k}"
            )
        return p, k

    def _build_model(self, num_classes: int) -> nn.Module:
        use_imagenet = self.pretrained_path and not self.checkpoint and not self.resume
        return ReIDModelRegistry.build_model(
            name=self.model_name,
            weights=Path("vehicle_osnet_x1_0.pth"),
            num_classes=num_classes,
            loss="triplet",
            pretrained=bool(use_imagenet),
            use_gpu=self.device.type != "cpu",
            pretrained_path=self.pretrained_path if use_imagenet else None,
            num_view_classes=self.num_view_classes,
            view_layers=self.view_layers,
        )

    def _load_checkpoint(self, model: nn.Module, path: str | Path) -> None:
        ckpt_path = Path(path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self._load_model_state(model, checkpoint)
        LOGGER.info(
            f"Loaded checkpoint {ckpt_path} "
            f"(epoch={checkpoint.get('epoch', '?')}, "
            f"best={checkpoint.get('best_mAP', '?')})"
        )

    def _load_model_state(self, model: nn.Module, checkpoint: dict[str, Any]) -> None:
        state_dict = checkpoint.get("state_dict", checkpoint)
        if hasattr(model, "view_heads"):
            state_dict = remap_vehicle_view_state_dict(state_dict)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            LOGGER.warning(f"Checkpoint missing keys: {missing}")
        if unexpected:
            LOGGER.warning(f"Checkpoint unexpected keys: {unexpected}")

    def _load_resume_state(
        self,
        path: Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
        save_dir: Path,
    ) -> tuple[int, int, float, list[VehicleTrainMetrics], list[dict[str, VehicleValMetrics]]]:
        if not path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self._load_model_state(model, checkpoint)

        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])

        for group in optimizer.param_groups:
            group.setdefault("_base_lr", self.lr)

        completed_epoch = int(checkpoint.get("epoch", 0))
        start_epoch = completed_epoch + 1
        history, val_history, best_epoch, best_score = self._load_metrics_history(save_dir)
        if best_score == float("-inf"):
            best_score = float(checkpoint.get("best_mAP", float("-inf")))

        LOGGER.info(
            f"Resuming training from epoch {start_epoch} "
            f"(completed epoch {completed_epoch}, best_epoch={best_epoch}, "
            f"best_mAP={best_score:.6f})"
        )
        return start_epoch, best_epoch, best_score, history, val_history

    @staticmethod
    def _load_metrics_history(
        save_dir: Path,
    ) -> tuple[list[VehicleTrainMetrics], list[dict[str, VehicleValMetrics]], int, float]:
        metrics_path = save_dir / "metrics.json"
        if not metrics_path.is_file():
            return [], [], 0, float("-inf")

        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        history = [VehicleTrainMetrics(**entry) for entry in data.get("train", [])]
        val_history: list[dict[str, VehicleValMetrics]] = []
        for epoch_vals in data.get("val", []):
            val_history.append(
                {
                    name: VehicleValMetrics(**metrics)
                    for name, metrics in epoch_vals.items()
                }
            )
        return (
            history,
            val_history,
            int(data.get("best_epoch", 0)),
            float(data.get("best_mAP", float("-inf"))),
        )

    def _apply_freeze(self, model: nn.Module) -> None:
        if not self.freeze_backbone and not self.freeze_reid_heads:
            return
        frozen: list[str] = []
        if self.freeze_backbone and hasattr(model, "backbone"):
            for param in model.backbone.parameters():
                param.requires_grad = False
            frozen.append("backbone")
        if self.freeze_reid_heads:
            for name in ("bottleneck", "classifier", "id_classifier"):
                module = getattr(model, name, None)
                if module is not None:
                    for param in module.parameters():
                        param.requires_grad = False
            frozen.append("reid_heads")
        if frozen:
            LOGGER.info(f"Frozen modules: {', '.join(frozen)}")

    def _selection_score(self, epoch_vals: dict[str, VehicleValMetrics]) -> float:
        if self.best_metric == "veri_view_acc":
            return float(epoch_vals.get("veri", VehicleValMetrics("veri")).val_view_acc or -1.0)
        return float(epoch_vals.get("veri", VehicleValMetrics("veri")).val_mAP)

    def _should_run_full_eval(self, epoch: int) -> bool:
        if self.eval_interval <= 1:
            return True
        return epoch % self.eval_interval == 0 or epoch == self.epochs

    def _build_train_loader(
        self,
        samples: list[ReIDSample],
        p: int,
        k: int,
    ) -> DataLoader:
        transform = build_train_transforms(
            self.img_size,
            preprocess=self.preprocess,
            horizontal_flip=False,
        )
        ds = ReIDImageDataset(
            samples,
            transform=transform,
            include_metadata=True,
            sample_transform=ViewAwareHorizontalFlip(p=0.5),
        )
        return DataLoader(
            ds,
            batch_size=p * k,
            sampler=PKSampler(samples, p=p, k=k),
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def _build_loss_loader(
        self,
        samples: list[ReIDSample],
        batch_size: int,
    ) -> DataLoader:
        transform = build_test_transforms(self.img_size, preprocess=self.preprocess)
        ds = ReIDImageDataset(samples, transform=transform, include_metadata=True)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,
            drop_last=True,
        )

    def _build_ranking_loaders(self, dataset: Any, batch_size: int) -> tuple[DataLoader, DataLoader]:
        transform = build_test_transforms(self.img_size, preprocess=self.preprocess)
        kwargs = dict(
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,
        )
        return (
            DataLoader(ReIDImageDataset(dataset.query.samples, transform=transform), **kwargs),
            DataLoader(ReIDImageDataset(dataset.gallery.samples, transform=transform), **kwargs),
        )

    def _train_epoch(
        self,
        epoch: int,
        model: nn.Module,
        loader: DataLoader,
        criterion_id: nn.Module,
        criterion_triplet: nn.Module,
        criterion_view: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
    ) -> VehicleTrainMetrics:
        model.train()
        totals = {"loss": 0.0, "id": 0.0, "triplet": 0.0, "view": 0.0}
        n_batches = 0
        use_amp = self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        t0 = time.monotonic()

        for imgs, pids, _, view_ids, has_view, _ in loader:
            imgs = imgs.to(self.device)
            pids = pids.to(self.device)
            view_ids = view_ids.to(self.device)
            has_view = has_view.to(self.device, dtype=torch.bool)

            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(imgs, return_dict=True)
                loss, loss_id, loss_tri, loss_view = self._compute_loss(
                    output,
                    pids,
                    view_ids,
                    has_view,
                    criterion_id,
                    criterion_triplet,
                    criterion_view,
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            totals["loss"] += loss.item()
            totals["id"] += loss_id.item()
            totals["triplet"] += loss_tri.item()
            totals["view"] += loss_view.item()
            n_batches += 1

        if epoch > self.warmup_epochs:
            scheduler.step()
        elif self.warmup_epochs > 0:
            warmup_factor = epoch / self.warmup_epochs
            for group in optimizer.param_groups:
                group["lr"] = group["_base_lr"] * warmup_factor

        LOGGER.debug(f"Epoch {epoch} train elapsed {time.monotonic() - t0:.1f}s")
        denom = max(n_batches, 1)
        return VehicleTrainMetrics(
            epoch=epoch,
            lr=optimizer.param_groups[0]["lr"],
            train_loss_total=totals["loss"] / denom,
            train_loss_id=totals["id"] / denom,
            train_loss_triplet=totals["triplet"] / denom,
            train_loss_view=totals["view"] / denom,
        )

    def _compute_loss(
        self,
        output: dict[str, torch.Tensor],
        pids: torch.Tensor,
        view_ids: torch.Tensor,
        has_view: torch.Tensor,
        criterion_id: nn.Module,
        criterion_triplet: nn.Module,
        criterion_view: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_id = criterion_id(output["id_logits"], pids)
        loss_triplet = criterion_triplet(output["embedding"], pids)
        loss_view = self._compute_view_loss(output, view_ids, has_view, criterion_view)
        loss = (
            self.lambda_id * loss_id
            + self.lambda_triplet * loss_triplet
            + self.lambda_view * loss_view
        )
        return loss, loss_id, loss_triplet, loss_view

    def _compute_view_loss(
        self,
        output: dict[str, torch.Tensor],
        view_ids: torch.Tensor,
        has_view: torch.Tensor,
        criterion_view: nn.Module,
    ) -> torch.Tensor:
        if not has_view.any():
            return torch.tensor(0.0, device=self.device)

        by_layer = output.get("view_logits_by_layer")
        if not by_layer:
            return criterion_view(output["view_logits"][has_view], view_ids[has_view])

        total_weight = 0.0
        loss_view = torch.tensor(0.0, device=self.device)
        for layer in self.view_layers:
            logits = by_layer.get(layer)
            if logits is None:
                continue
            weight = self.view_layer_weights[layer]
            layer_loss = criterion_view(logits[has_view], view_ids[has_view])
            loss_view = loss_view + weight * layer_loss
            total_weight += weight
        if total_weight <= 0:
            return torch.tensor(0.0, device=self.device)
        return loss_view / total_weight

    @torch.no_grad()
    def _validate_all(
        self,
        model: nn.Module,
        val_loss_loaders: dict[str, DataLoader],
        ranking_loaders: dict[str, tuple[DataLoader, DataLoader]],
        criterion_id: nn.Module,
        criterion_triplet: nn.Module,
        criterion_view: nn.Module,
    ) -> dict[str, VehicleValMetrics]:
        model.eval()
        results: dict[str, VehicleValMetrics] = {}
        for name, loader in val_loss_loaders.items():
            results[name] = self._validate_loss(
                name,
                model,
                loader,
                criterion_id,
                criterion_triplet,
                criterion_view,
            )

        if self.eval_ranking:
            model.eval()
            for name, (query_loader, gallery_loader) in ranking_loaders.items():
                q_feats, q_pids, q_camids = extract_features(
                    model, query_loader, self.device, desc=f"{name} query"
                )
                g_feats, g_pids, g_camids = extract_features(
                    model, gallery_loader, self.device, desc=f"{name} gallery"
                )
                distmat = compute_distance_matrix(q_feats, g_feats)
                cmc, mAP = evaluate_ranking(distmat, q_pids, g_pids, q_camids, g_camids)
                result = results.setdefault(name, VehicleValMetrics(dataset=name))
                result.val_mAP = float(mAP)
                result.val_rank1 = float(cmc[0]) if len(cmc) else 0.0
        return results

    def _validate_loss(
        self,
        name: str,
        model: nn.Module,
        loader: DataLoader,
        criterion_id: nn.Module,
        criterion_triplet: nn.Module,
        criterion_view: nn.Module,
    ) -> VehicleValMetrics:
        totals = {"loss": 0.0, "id": 0.0, "triplet": 0.0, "view": 0.0}
        correct_view = 0
        total_view = 0
        n_batches = 0

        for imgs, pids, _, view_ids, has_view, _ in loader:
            imgs = imgs.to(self.device)
            pids = pids.to(self.device)
            view_ids = view_ids.to(self.device)
            has_view = has_view.to(self.device, dtype=torch.bool)
            output = model(imgs, return_dict=True)
            loss, loss_id, loss_tri, loss_view = self._compute_loss(
                output,
                pids,
                view_ids,
                has_view,
                criterion_id,
                criterion_triplet,
                criterion_view,
            )
            totals["loss"] += loss.item()
            totals["id"] += loss_id.item()
            totals["triplet"] += loss_tri.item()
            totals["view"] += loss_view.item()
            if has_view.any():
                preds = output["view_logits"][has_view].argmax(dim=1)
                correct_view += (preds == view_ids[has_view]).sum().item()
                total_view += int(has_view.sum().item())
            n_batches += 1

        denom = max(n_batches, 1)
        return VehicleValMetrics(
            dataset=name,
            val_loss_total=totals["loss"] / denom,
            val_loss_id=totals["id"] / denom,
            val_loss_triplet=totals["triplet"] / denom,
            val_loss_view=totals["view"] / denom if total_view > 0 else math.nan,
            val_view_acc=(correct_view / total_view) if total_view > 0 else math.nan,
        )

    def _save_checkpoint(
        self,
        path: Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
        epoch: int,
        best_mAP: float,
        hparams: dict[str, Any],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_mAP": best_mAP,
                "model_name": self.model_name,
                "num_classes": model.num_classes,
                "num_view_classes": self.num_view_classes,
                "view_layers": list(self.view_layers),
                "view_layer_weights": self.view_layer_weights,
                "preprocess": self.preprocess,
                "img_size": list(self.img_size),
                "config": hparams,
            },
            path,
        )

    def _write_epoch_log(
        self,
        train: VehicleTrainMetrics,
        vals: dict[str, VehicleValMetrics],
        *,
        skip_val: bool = False,
    ) -> None:
        parts = [
            f"epoch={train.epoch}",
            f"lr={train.lr:.8f}",
            f"train_loss_total={train.train_loss_total:.6f}",
            f"train_loss_id={train.train_loss_id:.6f}",
            f"train_loss_triplet={train.train_loss_triplet:.6f}",
            f"train_loss_view={train.train_loss_view:.6f}",
        ]
        if skip_val:
            parts.append("val=skipped")
        else:
            for name in self.datasets:
                val = vals.get(name, VehicleValMetrics(dataset=name))
                prefix = f"{name}_val"
                parts.extend([
                    f"{prefix}_loss_total={val.val_loss_total:.6f}",
                    f"{prefix}_loss_id={val.val_loss_id:.6f}",
                    f"{prefix}_loss_triplet={val.val_loss_triplet:.6f}",
                    f"{prefix}_loss_view={val.val_loss_view:.6f}",
                    f"{prefix}_view_acc={val.val_view_acc:.6f}",
                    f"{prefix}_mAP={val.val_mAP:.6f}",
                    f"{prefix}_rank1={val.val_rank1:.6f}",
                ])
        LOGGER.info(" ".join(parts))

    def _save_metrics(
        self,
        save_dir: Path,
        history: list[VehicleTrainMetrics],
        val_history: list[dict[str, VehicleValMetrics]],
        best_epoch: int,
        best_mAP: float,
    ) -> None:
        data = {
            "best_epoch": best_epoch,
            "best_mAP": best_mAP,
            "train": [m.__dict__ for m in history],
            "val": [
                {name: metrics.__dict__ for name, metrics in epoch_vals.items()}
                for epoch_vals in val_history
            ],
        }
        (save_dir / "metrics.json").write_text(json.dumps(data, indent=2))

    def _hparams(
        self,
        num_classes: int,
        effective_p: int,
        effective_k: int,
        batch_size: int,
    ) -> dict[str, Any]:
        return {
            **self.config,
            "model": self.model_name,
            "datasets": self.datasets,
            "data_dir": self.data_dir,
            "num_classes": num_classes,
            "num_view_classes": self.num_view_classes,
            "view_layers": list(self.view_layers),
            "view_layer_weights": self.view_layer_weights,
            "img_size": list(self.img_size),
            "preprocess": self.preprocess,
            "lambda_id": self.lambda_id,
            "lambda_triplet": self.lambda_triplet,
            "lambda_view": self.lambda_view,
            "checkpoint": self.checkpoint,
            "resume": self.resume,
            "freeze_backbone": self.freeze_backbone,
            "freeze_reid_heads": self.freeze_reid_heads,
            "best_metric": self.best_metric,
            "eval_ranking": self.eval_ranking,
            "eval_interval": self.eval_interval,
            "p": effective_p,
            "k": effective_k,
            "batch_size": batch_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "seed": self.seed,
        }

    def _make_save_dir(self) -> Path:
        if self.resume:
            save_dir = Path(self.resume).resolve().parent
            if not save_dir.is_dir():
                raise FileNotFoundError(f"Resume run directory not found: {save_dir}")
            return save_dir

        base = self.project / self.name
        if base.exists():
            idx = 1
            while (self.project / f"{self.name}_{idx}").exists():
                idx += 1
            base = self.project / f"{self.name}_{idx}"
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _configure_file_logger(self, save_dir: Path) -> None:
        log_path = save_dir / self.log_file
        if any(
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename) == log_path
            for handler in LOGGER.handlers
        ):
            return
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        LOGGER.addHandler(handler)
