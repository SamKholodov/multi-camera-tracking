import cv2
import numpy as np


class Visualizer:
    @staticmethod
    def color_from_id(track_id):
        np.random.seed(track_id)
        return np.random.randint(0, 255, size=3).tolist()

    @staticmethod
    def draw_boxes(img, dets):
        for i, bbox in enumerate(dets):
            np.random.seed(i)
            color = (
                int(np.random.randint(50, 255)),
                int(np.random.randint(50, 255)),
                int(np.random.randint(50, 255)),
            )

            x1, y1, x2, y2 = map(int, bbox[:4])
            conf = bbox[4]

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                img,
                f"{conf:.3f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                3,
            )
        return img

    @staticmethod
    def draw_tracks(frame, tracks, target_class=None, cam_id=None, ids_dict=None):
        if len(tracks) == 0:
            return frame

        for track in tracks:
            if len(track) < 5:
                continue

            x1, y1, x2, y2, local_id = map(int, track[:5])

            if ids_dict is not None:
                track_id = ids_dict.get(local_id, local_id)
            else:
                track_id = local_id

            np.random.seed(track_id)
            color = (
                int(np.random.randint(50, 255)),
                int(np.random.randint(50, 255)),
                int(np.random.randint(50, 255)),
            )

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                frame,
                f"ID:{track_id}",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                3,
            )

        if cam_id is not None:
            cv2.putText(
                frame,
                f"Cam {cam_id}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                3,
            )

        if target_class is not None:
            cv2.putText(
                frame,
                f"Class: {target_class}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                3,
            )
            active_tracks = len([t for t in tracks if len(t) >= 5])
            cv2.putText(
                frame,
                f"Tracks: {active_tracks}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                3,
            )

        return frame

    @staticmethod
    def create_video_writer(path, frame, fps=30):
        """Open VideoWriter with even width/height (better compatibility on Windows)."""
        h, w = frame.shape[:2]
        w, h = w - (w % 2), h - (h % 2)
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter for {path}")
        return writer

    @staticmethod
    def visualize(
        frames,
        visualize=True,
        save_output=False,
        output_path="output.mp4",
        writer=None,
    ):
        if frames:
            if len(frames) == 1:
                final_display = cv2.resize(frames[0], (-1, -1), fx=0.6, fy=0.6)
            else:
                target_width = 640
                resized = [
                    cv2.resize(
                        f, (target_width, int(f.shape[0] * target_width / f.shape[1]))
                    )
                    for f in frames
                ]
                final_display = cv2.vconcat(resized)

            if visualize:
                cv2.imshow("Multi-Camera Tracking", final_display)

            if save_output:
                if writer is None:
                    h, w = final_display.shape[:2]
                    writer = cv2.VideoWriter(
                        output_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        30,
                        (w, h),
                    )
                writer.write(final_display)
        return writer
