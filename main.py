import cv2
import numpy as np
from pathlib import Path

from core.detector.detector import Detector
from core.io.tracklet_saver import TrackletSaver
from core.mot.tracker.bot_sort_tracker import BotSortTracker
from core.visualization.visualizer import Visualizer


DATASET_DIR = Path("dataset")

# -------------------------------------------
# ВКЛ / ВЫКЛ сохранение дефолтного трекинга
# -------------------------------------------
SAVE_TEST_TRACKING = False


def extract_track_feature_map(tracker_wrapper, tracks):
    feature_map = {}

    inner_tracker = getattr(tracker_wrapper, "tracker", None)
    if inner_tracker is None:
        return feature_map

    possible_track_lists = [
        "active_tracks",
        "tracked_stracks",
        "stracks",
        "tracks",
    ]

    for attr_name in possible_track_lists:
        obj_list = getattr(inner_tracker, attr_name, None)

        if obj_list is None:
            continue

        for obj in obj_list:
            tid = None

            for id_attr in ("id", "track_id", "tid"):
                if hasattr(obj, id_attr):
                    tid = int(getattr(obj, id_attr))
                    break

            if tid is None:
                continue

            feat = None

            for feat_attr in ("curr_feat", "smooth_feat", "feat", "features"):
                if hasattr(obj, feat_attr):
                    value = getattr(obj, feat_attr)

                    if value is None:
                        continue

                    if isinstance(value, list) and len(value) > 0:
                        value = value[-1]

                    try:
                        feat = np.asarray(value, dtype=np.float32).reshape(-1)
                    except Exception:
                        feat = None

                    if feat is not None and feat.size > 0:
                        break

            if feat is not None:
                feature_map[tid] = feat

    return feature_map


def process_video(video_path, detector, tracker):
    seq_name = video_path.parent.name
    cam_name = video_path.stem

    base_dir = Path("test_tracking") / seq_name
    base_dir.mkdir(parents=True, exist_ok=True)

    save_path = base_dir / f"{cam_name}.npz"

    saver = TrackletSaver()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    writer = None
    mot_file = None

    if SAVE_TEST_TRACKING:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 30

        fourcc = cv2.VideoWriter.fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            str(base_dir / f"{cam_name}_track_raw.mp4"),
            fourcc,
            fps,
            (width, height)
        )

        mot_file = open(base_dir / f"{cam_name}_track_raw.txt", "w")

    frame_id = 0

    print(f"Processing: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections, _ = detector.detect(frame)

        if detections is not None and len(detections) > 0:
            dets = np.asarray(detections)
        else:
            dets = np.empty((0, 6))

        tracks = tracker.update(dets, frame)

        if tracks is None:
            tracks = []
        else:
            tracks = np.asarray(tracks)

        track_feature_map = extract_track_feature_map(tracker, tracks)

        saver.update(
            frame_id=frame_id,
            tracks=tracks,
            track_feature_map=track_feature_map
        )

        if SAVE_TEST_TRACKING:
            assert writer is not None and mot_file is not None
            frame_draw = Visualizer.draw_tracks(frame.copy(), tracks)

            for tr in tracks:
                x1, y1, x2, y2, tid = tr[:5]

                w = x2 - x1
                h = y2 - y1
                conf = tr[5] if len(tr) > 5 else 1.0

                line = f"{frame_id},{int(tid)},{x1},{y1},{w},{h},{conf:.2f},-1,-1,-1\n"
                mot_file.write(line)

            writer.write(frame_draw)

        frame_id += 1

    cap.release()

    if SAVE_TEST_TRACKING:
        assert writer is not None and mot_file is not None
        writer.release()
        mot_file.close()

    saver.finalize()
    saver.save(save_path)

    print("Tracklets saved to:", save_path.resolve())


def main():
    video_paths = sorted(DATASET_DIR.glob("*/*.mp4"))

    if len(video_paths) == 0:
        raise RuntimeError(f"No videos found in {DATASET_DIR}")

    detector = Detector(
        model="yolov8m.pt",
        target_classes=0
    )

    for video_path in video_paths:
        tracker = BotSortTracker()
        process_video(video_path, detector, tracker)


if __name__ == "__main__":
    main()