"""
collect_gesture_samples.py — record labeled hand-landmark samples for GestureMLP.

`GestureMLP` (src/visual_ai/gesture_mlp.py) learns hand signs from examples
rather than a hand-coded geometric rule, which means it needs a dataset. This
tool builds one: point a webcam at your hand, hold a pose, press the number
key for its label, and each press appends one normalized feature vector (see
`landmarks_to_features`) to the running dataset. Runs MediaPipe Hands
directly rather than through `VisionPipeline`, since the pipeline's queue
payload does not carry raw landmarks — only the fixed set of derived fields
downstream games consume.

    python tools/collect_gesture_samples.py --labels fist,open_palm,point,peace,unknown --out gesture_samples.npz

Controls
--------
  1..9   record one sample of landmarks[i-1]'s label (i = key pressed)
  s      save the dataset collected so far, without quitting
  q/ESC  save and quit
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

from visual_ai.gesture_mlp import landmarks_to_features

try:
    import mediapipe.solutions.hands as mp_hands_module
except (ImportError, AttributeError):
    try:
        from mediapipe.python.solutions import hands as mp_hands_module
    except (ImportError, AttributeError):
        mp_hands_module = None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", required=True,
                         help="comma-separated label names, in key-press order (1=first, 2=second, ...)")
    parser.add_argument("--out", default="gesture_samples.npz", help="output .npz path")
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument("--append", action="store_true",
                         help="load --out first (if it exists) and append to it, instead of starting empty")
    args = parser.parse_args()

    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    if not labels:
        print("[collect] --labels must name at least one label", file=sys.stderr)
        return 1
    if len(labels) > 9:
        print("[collect] only 9 labels fit on keys 1-9", file=sys.stderr)
        return 1

    if mp_hands_module is None:
        print("[collect] mediapipe is not installed — cannot detect hand landmarks", file=sys.stderr)
        return 1

    features: list[np.ndarray] = []
    sample_labels: list[str] = []
    if args.append:
        try:
            data = np.load(args.out, allow_pickle=False)
            features = list(data["X"])
            sample_labels = list(data["y"])
            print(f"[collect] loaded {len(sample_labels)} existing samples from {args.out}")
        except FileNotFoundError:
            pass

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[collect] could not open camera {args.camera}", file=sys.stderr)
        return 1

    hands = mp_hands_module.Hands(
        static_image_mode=False, max_num_hands=1,
        model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.65,
    )

    print("[collect] Controls: 1-9 record a sample, s = save, q/ESC = save & quit")
    for i, label in enumerate(labels, start=1):
        print(f"    {i} -> {label}")

    last_lm = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            last_lm = None
            if result.multi_hand_landmarks:
                hand_lm = result.multi_hand_landmarks[0]
                last_lm = hand_lm.landmark
                h, w = frame.shape[:2]
                for a, b in mp_hands_module.HAND_CONNECTIONS:
                    pa, pb = hand_lm.landmark[a], hand_lm.landmark[b]
                    cv2.line(frame, (int(pa.x * w), int(pa.y * h)),
                             (int(pb.x * w), int(pb.y * h)), (0, 200, 0), 1)

            counts = {label: sample_labels.count(label) for label in labels}
            y = 24
            cv2.putText(frame, f"total: {len(sample_labels)}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            for i, label in enumerate(labels, start=1):
                y += 22
                cv2.putText(frame, f"[{i}] {label}: {counts[label]}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1)
            if last_lm is None:
                cv2.putText(frame, "no hand detected", (10, frame.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imshow("collect_gesture_samples", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord('q')):
                break
            if key == ord('s'):
                _save(args.out, features, sample_labels)
                continue
            if ord('1') <= key <= ord('9'):
                idx = key - ord('1')
                if idx < len(labels) and last_lm is not None:
                    features.append(landmarks_to_features(last_lm))
                    sample_labels.append(labels[idx])
                    print(f"[collect] +1 {labels[idx]}  (total {len(sample_labels)})")
                elif last_lm is None:
                    print("[collect] no hand detected, sample skipped")
    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()

    _save(args.out, features, sample_labels)
    return 0


def _save(path: str, features: list[np.ndarray], sample_labels: list[str]) -> None:
    if not sample_labels:
        print("[collect] nothing recorded, not writing a file")
        return
    X = np.stack(features).astype(np.float32)
    y = np.array(sample_labels)
    np.savez(path, X=X, y=y)
    print(f"[collect] saved {len(sample_labels)} samples -> {path}")


if __name__ == "__main__":
    raise SystemExit(main())
