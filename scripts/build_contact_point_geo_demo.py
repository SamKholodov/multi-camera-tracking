"""Contact point vs bottom-center cross-camera world distance demo.

Usage:
    python scripts/build_contact_point_geo_demo.py --dataset both --num-examples 6
    python scripts/build_contact_point_geo_demo.py --charts-only --dataset both \\
        --chart-legend-fontsize 24 --chart-dist-fontsize 20 --chart-figsize 9
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.geometry.contact_point.inference import ContactPointInference
from core.geometry.contact_point.model import uv_to_pixel
from core.io.calibration import (
    load_homography_image_to_world,
    project_bbox_bottom_center,
    project_point,
    world_distance,
)
from core.io.gta_mcmt import GtaMcmtDataset, image_path_for_cam_dir
from core.visualization.visualizer import Visualizer
from scripts.cityflow_ablation_common import GT_ROOT as CF_GT_ROOT, S02_CAM_IDS
from scripts.cityflow_sync_eval import apply_sync_alignment
from scripts.duplicate_track_utils import iou_xyxy, load_mot_by_frame
from scripts.eval_s02 import load_mot
from scripts.visualize_gta_mcmt import clip_box, stack_camera_views

DEFAULT_RUNS = {
    "gta": _ROOT / "outputs/configs_gta/geo_ablation/geo_tight",
    "cityflow": _ROOT / "outputs/configs_cityflow/temporal_ablation/temporal_penalty_N50",
}
CONTACT_WEIGHTS = _ROOT / "runs/contact_point/mobilenetv3_small_gta_points/best.pth"

GTA_CAMERAS = [0, 1, 2, 3]
GTA_GT_ROOT = _ROOT / "datasets/gta_mcmt"

CF_VIDEO = {
    6: CF_GT_ROOT / "c006/vdo_synch.avi",
    7: CF_GT_ROOT / "c007/vdo_synch.avi",
    8: CF_GT_ROOT / "c008/vdo_synch.avi",
    9: CF_GT_ROOT / "c009/vdo_synch.avi",
}

# High-contrast overlay colors (avoid red/blue — typical bbox tints from color_from_id).
COLOR_BOTTOM = (0, 140, 255)  # orange (BGR)
COLOR_CONTACT = (255, 255, 0)  # cyan (BGR)
COLOR_MARKER_OUTLINE = (0, 0, 0)

CHART_TITLE = "Проекция в мировые координаты"
LABEL_BOTTOM = "Нижняя точка (○)"
LABEL_CONTACT = "Точка контакта (×)"
CHART_TITLE_EN = "World projection"
LABEL_BOTTOM_EN = "Bottom center (○)"
LABEL_CONTACT_EN = "Contact point (×)"
CHART_COLOR_BOTTOM = "#FF8F00"
CHART_COLOR_CONTACT = "#00BFA5"
CHART_LABEL_COLOR_BOTTOM = "#E65100"
CHART_LABEL_COLOR_CONTACT = "#00695C"
CHART_LABEL_SUFFIX_BOTTOM = "○"
CHART_LABEL_SUFFIX_CONTACT = "×"
CHART_MARKER_BOTTOM = "o"
CHART_MARKER_CONTACT = "x"
CROP_PAD_RATIO = 0.40
CROP_MOSAIC_WIDTH = 640
CHART_LINE_WIDTH = 2.2
CHART_MARKER_SIZE = 160
CHART_MARKER_EDGE_WIDTH = 1.8
CHART_CONTACT_MARKER_EDGE_WIDTH = 3.0
CHART_LEGEND_FONTSIZE = 22
CHART_LEGEND_MARKERSIZE = 26
CHART_DIST_FONTSIZE = 18
CHART_DIST_BOX_PAD = 0.65
CHART_DIST_CORNER_PAD = 0.55
CHART_AXIS_LABEL_FONTSIZE = 12.0
CHART_TITLE_FONTSIZE = 14.0
CHART_TICK_FONTSIZE = 11.0
CROP_LEGEND_MAX_HEIGHT = 480


@dataclass(frozen=True)
class ChartStyle:
    figsize: float = 8.0
    dpi: int = 150
    line_width: float = CHART_LINE_WIDTH
    marker_size: float = CHART_MARKER_SIZE
    marker_edge_width: float = CHART_MARKER_EDGE_WIDTH
    contact_marker_edge_width: float = CHART_CONTACT_MARKER_EDGE_WIDTH
    legend_fontsize: float = CHART_LEGEND_FONTSIZE
    legend_markersize: float = CHART_LEGEND_MARKERSIZE
    dist_fontsize: float = CHART_DIST_FONTSIZE
    dist_box_pad: float = CHART_DIST_BOX_PAD
    dist_corner_pad: float = CHART_DIST_CORNER_PAD
    label_fontsize: float = 10.0
    axis_label_fontsize: float = CHART_AXIS_LABEL_FONTSIZE
    title_fontsize: float = CHART_TITLE_FONTSIZE
    tick_fontsize: float = CHART_TICK_FONTSIZE

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ChartStyle":
        return cls(
            figsize=float(args.chart_figsize),
            dpi=int(args.chart_dpi),
            line_width=float(args.chart_line_width),
            marker_size=float(args.chart_marker_size),
            marker_edge_width=float(args.chart_marker_edge_width),
            contact_marker_edge_width=float(args.chart_contact_marker_edge_width),
            legend_fontsize=float(args.chart_legend_fontsize),
            legend_markersize=float(args.chart_legend_markersize),
            dist_fontsize=float(args.chart_dist_fontsize),
            dist_box_pad=float(args.chart_dist_box_pad),
            dist_corner_pad=float(args.chart_dist_corner_pad),
            label_fontsize=float(args.chart_label_fontsize),
            axis_label_fontsize=float(args.chart_axis_label_fontsize),
            title_fontsize=float(args.chart_title_fontsize),
            tick_fontsize=float(args.chart_tick_fontsize),
        )


@dataclass
class PredMatch:
    cam: int
    frame: int
    gt_id: int
    local_id: int
    bbox_xyxy: list[float]
    bottom_px: tuple[float, float]
    contact_px: tuple[float, float] | None
    world_bottom: tuple[float, float]
    world_contact: tuple[float, float] | None
    iou: float

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def min_side(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return min(x2 - x1, y2 - y1)


@dataclass
class ExampleCandidate:
    frame: int
    gt_id: int
    views: list[PredMatch]
    pair_distances_bottom: dict[str, float] = field(default_factory=dict)
    pair_distances_contact: dict[str, float] = field(default_factory=dict)
    mean_improvement: float = 0.0
    score: float = 0.0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["gta", "cityflow", "both"], default="both")
    ap.add_argument("--locale", choices=["ru", "en"], default="ru")
    ap.add_argument("--num-examples", type=int, default=6)
    ap.add_argument("--scan-stride", type=int, default=None)
    ap.add_argument("--min-frame-gap", type=int, default=None)
    ap.add_argument("--iou-thresh", type=float, default=0.5)
    ap.add_argument("--out-dir", type=Path, default=_ROOT / "outputs/demos/contact_point_geo")
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--contact-weights", type=Path, default=CONTACT_WEIGHTS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--charts-only",
        action="store_true",
        help="Re-render world_projection.png from existing example meta.json (no rescan)",
    )
    ap.add_argument("--chart-figsize", type=float, default=8.0, help="Chart figure size (inches)")
    ap.add_argument("--chart-dpi", type=int, default=150, help="Chart PNG DPI")
    ap.add_argument("--chart-line-width", type=float, default=CHART_LINE_WIDTH)
    ap.add_argument("--chart-marker-size", type=float, default=CHART_MARKER_SIZE)
    ap.add_argument("--chart-marker-edge-width", type=float, default=CHART_MARKER_EDGE_WIDTH)
    ap.add_argument(
        "--chart-contact-marker-edge-width",
        type=float,
        default=CHART_CONTACT_MARKER_EDGE_WIDTH,
    )
    ap.add_argument("--chart-legend-fontsize", type=float, default=CHART_LEGEND_FONTSIZE)
    ap.add_argument("--chart-legend-markersize", type=float, default=CHART_LEGEND_MARKERSIZE)
    ap.add_argument("--chart-dist-fontsize", type=float, default=CHART_DIST_FONTSIZE)
    ap.add_argument(
        "--chart-dist-box-pad",
        type=float,
        default=CHART_DIST_BOX_PAD,
        help="Inner padding of distance box (in font-size units)",
    )
    ap.add_argument(
        "--chart-dist-corner-pad",
        type=float,
        default=CHART_DIST_CORNER_PAD,
        help="Margin of distance box from plot corner (in font-size units)",
    )
    ap.add_argument("--chart-label-fontsize", type=float, default=10.0, help="Cam labels on chart")
    ap.add_argument(
        "--chart-axis-label-fontsize",
        type=float,
        default=CHART_AXIS_LABEL_FONTSIZE,
        help="X/Y axis labels (e.g. 'X, m')",
    )
    ap.add_argument(
        "--chart-title-fontsize",
        type=float,
        default=CHART_TITLE_FONTSIZE,
        help="Chart title font size",
    )
    ap.add_argument(
        "--chart-tick-fontsize",
        type=float,
        default=CHART_TICK_FONTSIZE,
        help="Axis tick numbers font size",
    )
    return ap.parse_args()


def localized_labels(locale: str) -> tuple[str, str, str, str]:
    if locale == "en":
        return CHART_TITLE_EN, LABEL_BOTTOM_EN, LABEL_CONTACT_EN, "m"
    return CHART_TITLE, LABEL_BOTTOM, LABEL_CONTACT, "м"


def tlwh_to_xyxy(x: float, y: float, w: float, h: float) -> list[float]:
    return [x, y, x + w, y + h]


def gt_rows_at_frame(gt: np.ndarray, frame: int) -> list[tuple[int, list[float], float]]:
    if len(gt) == 0:
        return []
    rows = gt[gt[:, 0].astype(int) == frame]
    out: list[tuple[int, list[float], float]] = []
    for row in rows:
        x, y, w, h = map(float, row[2:6])
        out.append((int(row[1]), tlwh_to_xyxy(x, y, w, h), float(row[6])))
    return out


def greedy_match_frame(
    gt_rows: list[tuple[int, list[float], float]],
    local_rows: list[tuple[int, list[float], float]],
    *,
    cam: int,
    frame: int,
    iou_thresh: float,
) -> list[tuple[int, int, list[float], float]]:
    """Return (local_id, gt_id, bbox, iou)."""
    out: list[tuple[int, int, list[float], float]] = []
    used_gt: set[int] = set()
    for local_tid, pred_box, _ in sorted(
        local_rows, key=lambda r: -(r[1][2] - r[1][0]) * (r[1][3] - r[1][1])
    ):
        best_gt, best_iou = None, 0.0
        for gt_id, gt_box, _ in gt_rows:
            if gt_id in used_gt:
                continue
            iv = iou_xyxy(pred_box, gt_box)
            if iv > best_iou:
                best_iou, best_gt = iv, gt_id
        if best_gt is None or best_iou < iou_thresh:
            continue
        used_gt.add(best_gt)
        out.append((local_tid, best_gt, pred_box, best_iou))
    return out


def load_mot_by_frame_from_array(data: np.ndarray) -> dict[int, list[tuple[int, list[float], float]]]:
    by_frame: dict[int, list[tuple[int, list[float], float]]] = {}
    if len(data) == 0:
        return by_frame
    for row in data:
        frame = int(row[0])
        tid = int(row[1])
        x, y, w, h = map(float, row[2:6])
        conf = float(row[6])
        by_frame.setdefault(frame, []).append((tid, tlwh_to_xyxy(x, y, w, h), conf))
    return by_frame


def load_gta_data(run_dir: Path) -> tuple[dict, dict, dict, dict, int]:
    gt_by_cam, local_by_cam, homos = {}, {}, {}
    max_frame = 0
    for cam in GTA_CAMERAS:
        gt = load_mot(GTA_GT_ROOT / f"cam-{cam}/gt/gt.txt")
        gt_by_cam[cam] = gt
        local_by_cam[cam] = load_mot_by_frame(run_dir / "per_cam_local" / f"c{cam:03d}.txt")
        homos[cam] = load_homography_image_to_world(GTA_GT_ROOT / f"cam-{cam}/calibration.txt")
        if len(gt):
            max_frame = max(max_frame, int(gt[:, 0].max()))
    return gt_by_cam, local_by_cam, homos, {}, max_frame


def load_cityflow_data(run_dir: Path) -> tuple[dict, dict, dict, dict, int]:
    gt_by_cam, pr_local = {}, {}
    for cam in S02_CAM_IDS:
        gt_by_cam[cam] = load_mot(CF_GT_ROOT / f"c{cam:03d}/gt/gt.txt")
        pr_local[cam] = load_mot(run_dir / "per_cam_local" / f"c{cam:03d}.txt")
    gt_by_cam, pr_local, manifest = apply_sync_alignment(gt_by_cam, pr_local, CF_GT_ROOT)
    sync_len = int(manifest.get("sync_length_frames", 1920)) if manifest else 1920
    local_by_cam = {cam: load_mot_by_frame_from_array(pr_local[cam]) for cam in S02_CAM_IDS}
    homos = {
        cam: load_homography_image_to_world(CF_GT_ROOT / f"c{cam:03d}/calibration.txt")
        for cam in S02_CAM_IDS
    }
    return gt_by_cam, local_by_cam, homos, {}, sync_len


def bottom_px_from_bbox(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, y2


def contact_px_for_bbox(
    contact: ContactPointInference,
    frame_bgr: np.ndarray,
    bbox: list[float],
) -> tuple[float, float] | None:
    box = np.asarray([bbox], dtype=np.float32)
    crops = contact.get_crops(box, frame_bgr)
    if crops.shape[0] == 0:
        return None
    uv = contact.predict_uv_batch(crops)[0]
    if not np.all(np.isfinite(uv)):
        return None
    return uv_to_pixel(uv, bbox)


def build_view(
    contact: ContactPointInference,
    frame_bgr: np.ndarray,
    *,
    cam: int,
    frame: int,
    gt_id: int,
    local_id: int,
    bbox: list[float],
    H_i2w: np.ndarray,
    iou: float,
) -> PredMatch | None:
    if frame_bgr is None:
        return None
    w_bottom = project_bbox_bottom_center(H_i2w, *bbox)
    bpx = bottom_px_from_bbox(bbox)
    cpx = contact_px_for_bbox(contact, frame_bgr, bbox)
    w_contact = project_point(H_i2w, *cpx) if cpx is not None else None
    return PredMatch(
        cam=cam,
        frame=frame,
        gt_id=gt_id,
        local_id=local_id,
        bbox_xyxy=bbox,
        bottom_px=bpx,
        contact_px=cpx,
        world_bottom=w_bottom,
        world_contact=w_contact,
        iou=iou,
    )


def pair_key(cam_a: int, cam_b: int, dataset: str) -> str:
    a, b = sorted((cam_a, cam_b))
    if dataset == "cityflow":
        return f"c{a:03d}–c{b:03d}"
    return f"cam-{a}–cam-{b}"


def compute_pair_distances(
    views: list[PredMatch],
    *,
    metric: str,
    dataset: str,
) -> tuple[dict[str, float], dict[str, float], list[float]]:
    d_bottom: dict[str, float] = {}
    d_contact: dict[str, float] = {}
    improvements: list[float] = []
    by_cam = {v.cam: v for v in views}
    for cam_a, cam_b in combinations(sorted(by_cam.keys()), 2):
        va, vb = by_cam[cam_a], by_cam[cam_b]
        key = pair_key(cam_a, cam_b, dataset)
        d_bottom[key] = world_distance(va.world_bottom, vb.world_bottom, metric=metric)
        if va.world_contact is not None and vb.world_contact is not None:
            dc = world_distance(va.world_contact, vb.world_contact, metric=metric)
            d_contact[key] = dc
            improvements.append(d_bottom[key] - dc)
    return d_bottom, d_contact, improvements


_cf_caps: dict[int, cv2.VideoCapture] = {}


def load_gta_frame(dataset: GtaMcmtDataset, cam: int, frame: int) -> np.ndarray | None:
    k = frame - 1
    if k < 0 or k >= len(dataset):
        return None
    snap = dataset.snapshot(cam, k)
    return cv2.imread(str(image_path_for_cam_dir(dataset.cam_dirs[cam], snap.cam_id)))


def load_cityflow_frame(cam: int, frame: int) -> np.ndarray | None:
    if cam not in _cf_caps:
        cap = cv2.VideoCapture(str(CF_VIDEO[cam]))
        if not cap.isOpened():
            return None
        _cf_caps[cam] = cap
    cap = _cf_caps[cam]
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame - 1))
    ok, img = cap.read()
    return img if ok else None


def release_cityflow_caps() -> None:
    for cap in _cf_caps.values():
        cap.release()
    _cf_caps.clear()


def draw_circle_marker(
    img: np.ndarray, x: float, y: float, color: tuple[int, int, int], r: int = 8
) -> None:
    px, py = int(round(x)), int(round(y))
    cv2.circle(img, (px, py), r + 2, COLOR_MARKER_OUTLINE, 3, lineType=cv2.LINE_AA)
    cv2.circle(img, (px, py), r, color, 2, lineType=cv2.LINE_AA)


def draw_cross_marker(
    img: np.ndarray, x: float, y: float, color: tuple[int, int, int], r: int = 8
) -> None:
    px, py = int(round(x)), int(round(y))

    def _draw_x(arm: int, thickness: int, col: tuple[int, int, int]) -> None:
        cv2.line(img, (px - arm, py - arm), (px + arm, py + arm), col, thickness, lineType=cv2.LINE_AA)
        cv2.line(img, (px - arm, py + arm), (px + arm, py - arm), col, thickness, lineType=cv2.LINE_AA)

    _draw_x(r + 2, 5, COLOR_MARKER_OUTLINE)
    _draw_x(r, 2, color)


def _load_cyrillic_font(size: int):
    from PIL import ImageFont

    candidates = [
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _draw_legend_marker_pil(
    draw,
    cx: int,
    cy: int,
    r: int,
    rgb: tuple[int, int, int],
    *,
    kind: str,
) -> None:
    if kind == "bottom":
        draw.ellipse(
            (cx - r, cy - r, cx + r, cy + r),
            fill=rgb,
            outline=(0, 0, 0),
            width=2,
        )
        return
    draw.line((cx - r, cy - r, cx + r, cy + r), fill=(0, 0, 0), width=4)
    draw.line((cx - r, cy + r, cx + r, cy - r), fill=(0, 0, 0), width=4)
    draw.line((cx - r, cy - r, cx + r, cy + r), fill=rgb, width=2)
    draw.line((cx - r, cy + r, cx + r, cy - r), fill=rgb, width=2)


def draw_frame_legend(
    img: np.ndarray,
    *,
    show_contact: bool,
    compact: bool = False,
    locale: str = "ru",
) -> None:
    """Bottom-left legend on camera frames."""
    from PIL import Image, ImageDraw

    _, bottom_label, contact_label, _ = localized_labels(locale)
    entries: list[tuple[str, tuple[int, int, int], str]] = [
        (bottom_label, (COLOR_BOTTOM[2], COLOR_BOTTOM[1], COLOR_BOTTOM[0]), "bottom"),
    ]
    if show_contact:
        entries.append(
            (contact_label, (COLOR_CONTACT[2], COLOR_CONTACT[1], COLOR_CONTACT[0]), "contact")
        )

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    h = pil.height
    use_compact = compact or h < CROP_LEGEND_MAX_HEIGHT
    if use_compact:
        font_size = max(11, h // 90)
        pad_x, pad_y = 6, 5
        marker_r = 4
        gap = 6
        line_h = font_size + 6
        margin = 6
        box_radius = 4
        box_outline = 1
    else:
        font_size = max(16, h // 52)
        pad_x, pad_y = 12, 10
        marker_r = 7
        gap = 10
        line_h = font_size + 12
        margin = 12
        box_radius = 6
        box_outline = 2
    font = _load_cyrillic_font(font_size)

    text_widths = [int(draw.textlength(text, font=font)) for text, _, _ in entries]
    box_w = pad_x * 2 + marker_r * 2 + gap + max(text_widths)
    box_h = pad_y * 2 + line_h * len(entries)

    x0, y0 = margin, h - box_h - margin
    draw.rounded_rectangle(
        (x0, y0, x0 + box_w, y0 + box_h),
        radius=box_radius,
        fill=(255, 255, 255),
        outline=(136, 136, 136),
        width=box_outline,
    )

    for i, (text, rgb, kind) in enumerate(entries):
        cy = y0 + pad_y + i * line_h + line_h // 2
        cx_marker = x0 + pad_x + marker_r
        _draw_legend_marker_pil(draw, cx_marker, cy, marker_r, rgb, kind=kind)
        draw.text(
            (x0 + pad_x + marker_r * 2 + gap, cy - font_size // 2),
            text,
            fill=(33, 33, 33),
            font=font,
        )

    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def padded_crop_bounds(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: int,
    img_h: int,
    *,
    pad_ratio: float = CROP_PAD_RATIO,
) -> tuple[int, int, int, int] | None:
    """Expand bbox by pad_ratio of its size; clip to image (real background, no letterbox)."""
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad_x = bw * pad_ratio
    pad_y = bh * pad_ratio
    return clip_box(x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y, img_w, img_h)


def view_in_crop(
    view: PredMatch,
    *,
    origin_x: int,
    origin_y: int,
    bbox_xyxy: list[float],
) -> PredMatch:
    contact_px = None
    if view.contact_px is not None:
        contact_px = (view.contact_px[0] - origin_x, view.contact_px[1] - origin_y)
    return PredMatch(
        cam=view.cam,
        frame=view.frame,
        gt_id=view.gt_id,
        local_id=view.local_id,
        bbox_xyxy=bbox_xyxy,
        bottom_px=(view.bottom_px[0] - origin_x, view.bottom_px[1] - origin_y),
        contact_px=contact_px,
        world_bottom=view.world_bottom,
        world_contact=view.world_contact,
        iou=view.iou,
    )


def crop_vehicle_frame(
    img: np.ndarray,
    view: PredMatch,
    *,
    pad_ratio: float = CROP_PAD_RATIO,
) -> tuple[np.ndarray, PredMatch] | None:
    clipped = clip_box(*view.bbox_xyxy, img.shape[1], img.shape[0])
    if clipped is None:
        return None
    crop_bounds = padded_crop_bounds(*clipped, img.shape[1], img.shape[0], pad_ratio=pad_ratio)
    if crop_bounds is None:
        return None
    cx1, cy1, cx2, cy2 = crop_bounds
    x1, y1, x2, y2 = clipped
    crop_img = img[cy1:cy2, cx1:cx2].copy()
    crop_view = view_in_crop(
        view,
        origin_x=cx1,
        origin_y=cy1,
        bbox_xyxy=[float(x1 - cx1), float(y1 - cy1), float(x2 - cx1), float(y2 - cy1)],
    )
    return crop_img, crop_view


def draw_cam_frame(
    img: np.ndarray,
    view: PredMatch,
    *,
    mode: str,
    compact_legend: bool = False,
    locale: str = "ru",
) -> np.ndarray:
    out = img.copy()
    clipped = clip_box(*view.bbox_xyxy, out.shape[1], out.shape[0])
    if clipped is None:
        return out
    x1, y1, x2, y2 = clipped
    color = tuple(int(c) for c in Visualizer.color_from_id(view.gt_id))
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 3, lineType=cv2.LINE_AA)
    draw_circle_marker(out, view.bottom_px[0], view.bottom_px[1], COLOR_BOTTOM)
    has_contact = mode == "both" and view.contact_px is not None
    if has_contact:
        draw_cross_marker(out, view.contact_px[0], view.contact_px[1], COLOR_CONTACT)
    draw_frame_legend(out, show_contact=has_contact, compact=compact_legend, locale=locale)
    return out


def cam_label(cam: int, dataset: str) -> str:
    if dataset == "cityflow":
        return f"c{cam:03d}"
    return f"cam-{cam}"


def chart_point_label(cam: int, dataset: str, *, kind: str) -> str:
    suffix = CHART_LABEL_SUFFIX_BOTTOM if kind == "bottom" else CHART_LABEL_SUFFIX_CONTACT
    return f"{cam_label(cam, dataset)}{suffix}"


_EARTH_R_M = 6_371_000.0


def _gps_origin(views: list[PredMatch]) -> tuple[float, float]:
    lats: list[float] = []
    lons: list[float] = []
    for view in views:
        lats.append(view.world_bottom[0])
        lons.append(view.world_bottom[1])
        if view.world_contact is not None:
            lats.append(view.world_contact[0])
            lons.append(view.world_contact[1])
    return float(np.mean(lats)), float(np.mean(lons))


def world_to_plot_xy(
    world: tuple[float, float],
    *,
    metric: str,
    origin: tuple[float, float] | None,
) -> tuple[float, float]:
    """Map stored world coords to plot XY in metres."""
    if metric == "plane":
        return float(world[0]), float(world[1])
    if origin is None:
        raise ValueError("GPS metric requires a local origin")
    lat, lon = world
    lat0, lon0 = origin
    lat0_r = math.radians(lat0)
    east_m = _EARTH_R_M * math.radians(lon - lon0) * math.cos(lat0_r)
    north_m = _EARTH_R_M * math.radians(lat - lat0)
    return east_m, north_m


@dataclass
class _PlotLabel:
    x: float
    y: float
    text: str
    color: str
    kind: str  # "bottom" | "contact"
    cam: int


def _assign_label_offsets(labels: list[_PlotLabel], plot_span: float) -> list[tuple[float, float]]:
    """Place labels in data coords: per-camera quadrant + pairwise repulsion."""
    if not labels:
        return []

    base_dist = max(0.35, plot_span * 0.035)
    quadrant_dirs = ((1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0))
    offsets: list[list[float]] = []

    for lab in labels:
        ux, uy = quadrant_dirs[lab.cam % len(quadrant_dirs)]
        norm = math.hypot(ux, uy)
        ux, uy = ux / norm, uy / norm
        if lab.kind == "contact":
            ux, uy = -uy, ux
        scale = 0.95 if lab.kind == "bottom" else 1.15
        offsets.append([ux * base_dist * scale, uy * base_dist * scale])

    min_sep = max(0.65, plot_span * 0.05)
    anchors = [
        (labels[i].x + offsets[i][0], labels[i].y + offsets[i][1]) for i in range(len(labels))
    ]
    for _ in range(50):
        moved = False
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if labels[i].cam == labels[j].cam:
                    continue
                ddx = anchors[i][0] - anchors[j][0]
                ddy = anchors[i][1] - anchors[j][1]
                dist = math.hypot(ddx, ddy)
                if dist >= min_sep:
                    continue
                if dist < 1e-6:
                    ddx, ddy, dist = 1.0, 0.0, 1.0
                push = 0.6 * (min_sep - dist)
                ux, uy = ddx / dist, ddy / dist
                offsets[i][0] += ux * push
                offsets[i][1] += uy * push
                offsets[j][0] -= ux * push
                offsets[j][1] -= uy * push
                anchors[i] = (labels[i].x + offsets[i][0], labels[i].y + offsets[i][1])
                anchors[j] = (labels[j].x + offsets[j][0], labels[j].y + offsets[j][1])
                moved = True
        if not moved:
            break

    return [(o[0], o[1]) for o in offsets]


def _draw_plot_labels(
    ax,
    labels: list[_PlotLabel],
    offsets: list[tuple[float, float]],
    *,
    fontsize: float,
) -> None:
    for lab, (ox, oy) in zip(labels, offsets):
        ax.annotate(
            lab.text,
            xy=(lab.x, lab.y),
            xytext=(lab.x + ox, lab.y + oy),
            textcoords="data",
            fontsize=fontsize,
            color=lab.color,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=lab.color, alpha=0.9, lw=1.2),
            arrowprops=dict(arrowstyle="-", color=lab.color, lw=1.4, alpha=0.8, shrinkA=2, shrinkB=2),
            zorder=5,
        )



def _mean_cross_cam_distance_m(
    views: list[PredMatch],
    *,
    metric: str,
    dataset: str,
    anchor: str,
) -> float | None:
    d_bottom, d_contact, _ = compute_pair_distances(views, metric=metric, dataset=dataset)
    vals = list(d_bottom.values()) if anchor == "bottom" else list(d_contact.values())
    if not vals:
        return None
    return float(np.mean(vals))


def _cross_cam_plot_segments(
    views: list[PredMatch],
    *,
    metric: str,
    origin: tuple[float, float] | None,
    anchor: str,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Line segments only between different cameras (no same-cam links)."""
    by_cam = {v.cam: v for v in views}
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for cam_a, cam_b in combinations(sorted(by_cam.keys()), 2):
        va, vb = by_cam[cam_a], by_cam[cam_b]
        if anchor == "bottom":
            wa, wb = va.world_bottom, vb.world_bottom
        else:
            if va.world_contact is None or vb.world_contact is None:
                continue
            wa, wb = va.world_contact, vb.world_contact
        pa = world_to_plot_xy(wa, metric=metric, origin=origin)
        pb = world_to_plot_xy(wb, metric=metric, origin=origin)
        segments.append((pa, pb))
    return segments


def save_world_projection_chart(
    views: list[PredMatch],
    path: Path,
    *,
    dataset_name: str,
    metric: str,
    style: ChartStyle | None = None,
    locale: str = "ru",
) -> None:
    """XY scatter in metres: orange = bottom point, teal = contact point."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    chart = style or ChartStyle()
    chart_title, bottom_label, contact_label, unit_label = localized_labels(locale)

    plt.rcParams.update(
        {
            "font.sans-serif": ["Segoe UI", "Arial", "Liberation Sans", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )

    fig, ax = plt.subplots(figsize=(chart.figsize, chart.figsize))
    sorted_views = sorted(views, key=lambda v: v.cam)
    origin = _gps_origin(sorted_views) if metric == "gps" else None

    red_pts: list[tuple[float, float]] = []
    contact_pts: list[tuple[float, float]] = []
    plot_labels: list[_PlotLabel] = []

    for view in sorted_views:
        xb, yb = world_to_plot_xy(view.world_bottom, metric=metric, origin=origin)
        red_pts.append((xb, yb))
        ax.scatter(
            xb,
            yb,
            c=CHART_COLOR_BOTTOM,
            s=chart.marker_size,
            marker=CHART_MARKER_BOTTOM,
            zorder=3,
            edgecolors="black",
            linewidths=chart.marker_edge_width,
        )
        plot_labels.append(
            _PlotLabel(
                x=xb,
                y=yb,
                text=chart_point_label(view.cam, dataset_name, kind="bottom"),
                color=CHART_LABEL_COLOR_BOTTOM,
                kind="bottom",
                cam=view.cam,
            )
        )

        if view.world_contact is not None:
            xc, yc = world_to_plot_xy(view.world_contact, metric=metric, origin=origin)
            contact_pts.append((xc, yc))
            ax.scatter(
                xc,
                yc,
                c=CHART_COLOR_CONTACT,
                s=chart.marker_size,
                marker=CHART_MARKER_CONTACT,
                zorder=4,
                linewidths=chart.contact_marker_edge_width,
            )
            plot_labels.append(
                _PlotLabel(
                    x=xc,
                    y=yc,
                    text=chart_point_label(view.cam, dataset_name, kind="contact"),
                    color=CHART_LABEL_COLOR_CONTACT,
                    kind="contact",
                    cam=view.cam,
                )
            )

    for p1, p2 in _cross_cam_plot_segments(
        sorted_views, metric=metric, origin=origin, anchor="bottom"
    ):
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color=CHART_COLOR_BOTTOM,
            linestyle="--",
            linewidth=chart.line_width,
            alpha=0.65,
            zorder=1,
        )
    for p1, p2 in _cross_cam_plot_segments(
        sorted_views, metric=metric, origin=origin, anchor="contact"
    ):
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color=CHART_COLOR_CONTACT,
            linestyle="--",
            linewidth=chart.line_width,
            alpha=0.65,
            zorder=1,
        )

    red_dist_m = _mean_cross_cam_distance_m(
        sorted_views, metric=metric, dataset=dataset_name, anchor="bottom"
    )
    contact_dist_m = _mean_cross_cam_distance_m(
        sorted_views, metric=metric, dataset=dataset_name, anchor="contact"
    )

    ax.set_xlabel(f"X, {unit_label}", fontsize=chart.axis_label_fontsize)
    ax.set_ylabel(f"Y, {unit_label}", fontsize=chart.axis_label_fontsize)
    ax.set_title(chart_title, fontsize=chart.title_fontsize)
    ax.tick_params(axis="both", labelsize=chart.tick_fontsize)
    ax.grid(True, alpha=0.4, linestyle=":", linewidth=1.0)

    all_x = [p[0] for p in red_pts + contact_pts]
    all_y = [p[1] for p in red_pts + contact_pts]
    plot_span = 3.0
    if all_x:
        xmin, xmax = min(all_x), max(all_x)
        ymin, ymax = min(all_y), max(all_y)
        plot_span = max(xmax - xmin, ymax - ymin, 3.0)
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        half = max(xmax - xmin, ymax - ymin) * 0.5
        half = max(half, 1.5)
        margin = max(half * 0.45, 1.0)
        half += margin
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal", adjustable="box")

    label_offsets = _assign_label_offsets(plot_labels, plot_span)
    _draw_plot_labels(ax, plot_labels, label_offsets, fontsize=chart.label_fontsize)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=CHART_MARKER_BOTTOM,
            color="w",
            markerfacecolor=CHART_COLOR_BOTTOM,
            markeredgecolor="black",
            markeredgewidth=chart.marker_edge_width,
            markersize=chart.legend_markersize,
            label=bottom_label,
        ),
        Line2D(
            [0],
            [0],
            marker=CHART_MARKER_CONTACT,
            color=CHART_COLOR_CONTACT,
            linestyle="None",
            markersize=chart.legend_markersize,
            markeredgewidth=chart.contact_marker_edge_width,
            label=contact_label,
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=chart.legend_fontsize,
        framealpha=0.95,
        borderpad=1.0,
        labelspacing=0.8,
        handlelength=2.4,
        handletextpad=1.0,
    )

    dist_lines: list[str] = []
    if red_dist_m is not None:
        dist_lines.append(f"1 - {bottom_label}: {red_dist_m:.2f} {unit_label}")
    if contact_dist_m is not None:
        dist_lines.append(f"2 - {contact_label}: {contact_dist_m:.2f} {unit_label}")

    fig.tight_layout()

    if dist_lines:
        from matplotlib.offsetbox import AnchoredText

        dist_box = AnchoredText(
            "\n".join(dist_lines),
            loc="lower left",
            prop={"size": chart.dist_fontsize, "color": "#212121"},
            pad=chart.dist_corner_pad,
            borderpad=chart.dist_box_pad,
            frameon=True,
        )
        dist_box.patch.set_boxstyle("round,pad=0.2")
        dist_box.patch.set_facecolor("white")
        dist_box.patch.set_edgecolor("#888888")
        dist_box.patch.set_alpha(0.95)
        dist_box.patch.set_linewidth(1.4)
        dist_box.set_clip_on(True)
        ax.add_artist(dist_box)

    fig.savefig(path, dpi=chart.dpi)
    plt.close(fig)


def views_from_meta(meta: dict) -> list[PredMatch]:
    """Rebuild PredMatch list from exported example meta.json."""
    frame = int(meta["frame"])
    gt_id = int(meta["gt_id"])
    views: list[PredMatch] = []
    for cam in meta.get("cameras", {}).values():
        contact_px = cam.get("contact_px")
        world_contact = cam.get("world_contact")
        views.append(
            PredMatch(
                cam=int(cam["cam"]),
                frame=frame,
                gt_id=gt_id,
                local_id=int(cam["local_id"]),
                bbox_xyxy=[float(v) for v in cam["bbox_xyxy"]],
                bottom_px=(float(cam["bottom_px"][0]), float(cam["bottom_px"][1])),
                contact_px=(
                    (float(contact_px[0]), float(contact_px[1])) if contact_px else None
                ),
                world_bottom=(float(cam["world_bottom"][0]), float(cam["world_bottom"][1])),
                world_contact=(
                    (float(world_contact[0]), float(world_contact[1]))
                    if world_contact
                    else None
                ),
                iou=1.0,
            )
        )
    return views


def regenerate_charts_only(
    out_dir: Path,
    datasets: list[str],
    style: ChartStyle,
    locale: str = "ru",
) -> int:
    """Re-render world_projection.png for every example_* with meta.json."""
    count = 0
    for dataset_name in datasets:
        ds_dir = out_dir / dataset_name
        if not ds_dir.is_dir():
            print(f"[WARN] Missing dataset dir: {ds_dir}", flush=True)
            continue
        for meta_path in sorted(ds_dir.glob("example_*/meta.json")):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            views = views_from_meta(meta)
            if len(views) < 2:
                print(f"[WARN] Skip {meta_path.parent.name}: <2 camera views", flush=True)
                continue
            chart_path = meta_path.parent / "world_projection.png"
            metric = str(meta.get("metric", "plane" if dataset_name == "gta" else "gps"))
            save_world_projection_chart(
                views,
                chart_path,
                dataset_name=dataset_name,
                metric=metric,
                style=style,
                locale=locale,
            )
            print(f"[{dataset_name}] {meta_path.parent.name} -> {chart_path.name}", flush=True)
            count += 1
    return count


def collect_examples_and_stats(
    *,
    dataset_name: str,
    run_dir: Path,
    contact: ContactPointInference,
    scan_stride: int,
    min_side_px: float,
    min_area: float,
    metric: str,
    cameras: list[int],
    load_frame_fn,
    gta_dataset: GtaMcmtDataset | None,
) -> tuple[list[ExampleCandidate], dict]:
    del gta_dataset
    from scripts.contact_point_cross_cam_stats import collect_cross_cam_distance_stats

    result = collect_cross_cam_distance_stats(
        dataset_name=dataset_name,
        run_dir=run_dir,
        contact=contact,
        scan_stride=scan_stride,
        min_side_px=min_side_px,
        min_area=min_area,
        metric=metric,
        cameras=cameras,
        load_frame_fn=load_frame_fn,
        collect_examples=True,
    )
    return result.example_pool, result.stats


def select_diverse_examples(candidates: list[ExampleCandidate], n: int, min_gap: int) -> list[ExampleCandidate]:
    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
    selected: list[ExampleCandidate] = []
    used: list[int] = []
    for cand in ranked:
        if any(abs(cand.frame - f) < min_gap for f in used):
            continue
        selected.append(cand)
        used.append(cand.frame)
        if len(selected) >= n:
            break
    return selected


def export_example(
    ex: ExampleCandidate,
    *,
    example_idx: int,
    out_dir: Path,
    dataset_name: str,
    metric: str,
    load_frame_fn,
    chart_style: ChartStyle | None = None,
    locale: str = "ru",
) -> None:
    ex_dir = out_dir / dataset_name / f"example_{example_idx:03d}"
    ex_dir.mkdir(parents=True, exist_ok=True)

    bottom_frames: list[np.ndarray] = []
    both_frames: list[np.ndarray] = []
    crop_bottom_frames: list[np.ndarray] = []
    crop_both_frames: list[np.ndarray] = []
    cameras_meta: dict = {}

    for view in sorted(ex.views, key=lambda v: v.cam):
        img = load_frame_fn(view.cam, ex.frame)
        if img is None:
            continue

        cam_key = f"cam{view.cam}"
        bottom_img = draw_cam_frame(img, view, mode="bottom", locale=locale)
        both_img = draw_cam_frame(img, view, mode="both", locale=locale)
        cv2.imwrite(str(ex_dir / f"{cam_key}_bottom.jpg"), bottom_img)
        cv2.imwrite(str(ex_dir / f"{cam_key}_both.jpg"), both_img)
        bottom_frames.append(bottom_img)
        both_frames.append(both_img)

        cropped = crop_vehicle_frame(img, view, pad_ratio=CROP_PAD_RATIO)
        if cropped is not None:
            crop_img, crop_view = cropped
            crop_bottom = draw_cam_frame(
                crop_img,
                crop_view,
                mode="bottom",
                compact_legend=True,
                locale=locale,
            )
            crop_both = draw_cam_frame(
                crop_img,
                crop_view,
                mode="both",
                compact_legend=True,
                locale=locale,
            )
            cv2.imwrite(str(ex_dir / f"{cam_key}_crop_bottom.jpg"), crop_bottom)
            cv2.imwrite(str(ex_dir / f"{cam_key}_crop_both.jpg"), crop_both)
            crop_bottom_frames.append(crop_bottom)
            crop_both_frames.append(crop_both)

        cameras_meta[cam_key] = {
            "cam": view.cam,
            "frame": ex.frame,
            "gt_id": view.gt_id,
            "local_id": view.local_id,
            "bbox_xyxy": view.bbox_xyxy,
            "crop_pad_ratio": CROP_PAD_RATIO,
            "bottom_px": list(view.bottom_px),
            "contact_px": list(view.contact_px) if view.contact_px else None,
            "world_bottom": list(view.world_bottom),
            "world_contact": list(view.world_contact) if view.world_contact else None,
        }

    if bottom_frames:
        cv2.imwrite(str(ex_dir / "mosaic_bottom.jpg"), stack_camera_views(bottom_frames))
    if both_frames:
        cv2.imwrite(str(ex_dir / "mosaic_both.jpg"), stack_camera_views(both_frames))
    if crop_bottom_frames:
        cv2.imwrite(
            str(ex_dir / "mosaic_crop_bottom.jpg"),
            stack_camera_views(crop_bottom_frames, target_width=CROP_MOSAIC_WIDTH),
        )
    if crop_both_frames:
        cv2.imwrite(
            str(ex_dir / "mosaic_crop_both.jpg"),
            stack_camera_views(crop_both_frames, target_width=CROP_MOSAIC_WIDTH),
        )

    save_world_projection_chart(
        ex.views,
        ex_dir / "world_projection.png",
        dataset_name=dataset_name,
        metric=metric,
        style=chart_style,
        locale=locale,
    )

    pairs = []
    for key in ex.pair_distances_bottom:
        row = {"pair": key, "bottom_m": ex.pair_distances_bottom[key]}
        if key in ex.pair_distances_contact:
            row["contact_m"] = ex.pair_distances_contact[key]
            row["improvement_m"] = row["bottom_m"] - row["contact_m"]
        pairs.append(row)

    meta = {
        "dataset": dataset_name,
        "frame": ex.frame,
        "gt_id": ex.gt_id,
        "example_index": example_idx,
        "mean_improvement_m": ex.mean_improvement,
        "metric": metric,
        "crop_pad_ratio": CROP_PAD_RATIO,
        "pair_distances": pairs,
        "cameras": cameras_meta,
    }
    (ex_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def process_dataset(
    dataset_name: str,
    run_dir: Path,
    out_dir: Path,
    contact: ContactPointInference,
    args: argparse.Namespace,
) -> dict:
    pred_file = run_dir / "per_cam_local" / ("c000.txt" if dataset_name == "gta" else "c006.txt")
    if not pred_file.is_file():
        raise SystemExit(f"Missing predictions: {pred_file}")

    stride = args.scan_stride or (50 if dataset_name == "gta" else 5)
    min_gap = args.min_frame_gap or (200 if dataset_name == "gta" else 50)
    min_side = 40.0 if dataset_name == "gta" else 20.0
    min_area = min_side * min_side
    metric = "plane" if dataset_name == "gta" else "gps"
    cameras = GTA_CAMERAS if dataset_name == "gta" else S02_CAM_IDS
    gta_dataset = GtaMcmtDataset(GTA_GT_ROOT) if dataset_name == "gta" else None

    def load_frame(cam: int, frame: int) -> np.ndarray | None:
        if dataset_name == "gta":
            assert gta_dataset is not None
            return load_gta_frame(gta_dataset, cam, frame)
        return load_cityflow_frame(cam, frame)

    print(f"[{dataset_name}] Scanning (stride={stride})...", flush=True)
    pool, stats = collect_examples_and_stats(
        dataset_name=dataset_name,
        run_dir=run_dir,
        contact=contact,
        scan_stride=stride,
        min_side_px=min_side,
        min_area=min_area,
        metric=metric,
        cameras=cameras,
        load_frame_fn=load_frame,
        gta_dataset=gta_dataset,
    )
    print(f"[{dataset_name}] Pool={len(pool)} cross-cam pairs in stats={stats['pairs_total']}", flush=True)

    selected = select_diverse_examples(pool, args.num_examples, min_gap)
    print(f"[{dataset_name}] Selected {len(selected)} examples", flush=True)
    chart_style = ChartStyle.from_args(args)

    for idx, ex in enumerate(selected, start=1):
        print(
            f"  example_{idx:03d}: frame={ex.frame} gt_id={ex.gt_id} "
            f"cams={[v.cam for v in ex.views]} mean_imp={ex.mean_improvement:.3f}m",
            flush=True,
        )
        export_example(
            ex,
            example_idx=idx,
            out_dir=out_dir,
            dataset_name=dataset_name,
            metric=metric,
            load_frame_fn=load_frame,
            chart_style=chart_style,
            locale=args.locale,
        )

    if dataset_name == "cityflow":
        release_cityflow_caps()

    stats["examples_exported"] = len(selected)
    return stats


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    chart_style = ChartStyle.from_args(args)
    datasets = ["gta", "cityflow"] if args.dataset == "both" else [args.dataset]

    if args.charts_only:
        if not out_dir.is_dir():
            raise SystemExit(f"Output dir not found: {out_dir}")
        n = regenerate_charts_only(out_dir, datasets, chart_style, locale=args.locale)
        print(f"Done. Re-rendered {n} chart(s) in {out_dir.resolve()}", flush=True)
        return

    if out_dir.exists() and args.force:
        import shutil
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.contact_weights.is_file():
        raise SystemExit(f"Contact weights not found: {args.contact_weights}")

    print(f"Loading contact model: {args.contact_weights}", flush=True)
    contact = ContactPointInference(
        weights=args.contact_weights,
        device=0,
        bbox_pad_ratio=0.05,
        pretrained_backbone=False,
    )

    all_stats: dict = {}

    for ds in datasets:
        run_dir = args.run_dir if args.run_dir is not None else DEFAULT_RUNS[ds]
        if args.run_dir is not None and len(datasets) > 1:
            raise SystemExit("--run-dir only with single --dataset")
        all_stats[ds] = process_dataset(ds, run_dir, out_dir, contact, args)

    stats_path = out_dir / "stats.json"
    stats_path.write_text(json.dumps(all_stats, indent=2), encoding="utf-8")
    print(f"Done. Output: {out_dir.resolve()}", flush=True)
    print(f"Stats: {stats_path}", flush=True)


if __name__ == "__main__":
    main()
