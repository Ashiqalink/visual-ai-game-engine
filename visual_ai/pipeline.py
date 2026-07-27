import threading
import queue
import time
import cv2
import numpy as np

HAS_MEDIAPIPE = False
mp_face_detection_module = None

try:
    import mediapipe as mp
    try:
        import mediapipe.solutions.face_detection as mp_face_detection_module
        HAS_MEDIAPIPE = True
    except (ImportError, AttributeError):
        try:
            from mediapipe.python.solutions import face_detection as mp_face_detection_module
            HAS_MEDIAPIPE = True
        except (ImportError, AttributeError):
            HAS_MEDIAPIPE = False
except (ImportError, AttributeError):
    HAS_MEDIAPIPE = False


class VisionPipeline(threading.Thread):
    """
    Background vision thread capturing webcam feed and detecting facial coordinates.
    Pushes detected coordinates into a thread-safe Queue.
    """
    def __init__(self, result_queue: queue.Queue, width: int = 800, height: int = 600, camera_index: int = 0):
        super().__init__(daemon=True)
        self.result_queue = result_queue
        self.width = width
        self.height = height
        self.camera_index = camera_index
        self.running = False
        
        # Mediapipe setup
        self.mp_face_detection = None
        if HAS_MEDIAPIPE and mp_face_detection_module is not None:
            try:
                self.mp_face_detection = mp_face_detection_module.FaceDetection(
                    model_selection=0, min_detection_confidence=0.5
                )
            except Exception as e:
                print(f"[VisionPipeline] MediaPipe FaceDetection init warning: {e}")
                self.mp_face_detection = None

    def run(self):
        self.running = True
        cap = cv2.VideoCapture(self.camera_index)
        
        # Check if camera opened successfully
        camera_available = cap.isOpened()
        if not camera_available:
            print(f"[VisionPipeline] Camera index {self.camera_index} not accessible. Running simulated vision target.")

        sim_angle = 0.0

        while self.running:
            if camera_available:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                # Resize frame to target dimensions
                frame = cv2.resize(frame, (self.width, self.height))
                
                # Flip for natural mirror effect
                frame = cv2.flip(frame, 1)

                target_x, target_y = None, None

                # Process face detection
                if HAS_MEDIAPIPE and self.mp_face_detection:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.mp_face_detection.process(rgb_frame)

                    if results.detections:
                        # Grab first detected face
                        detection = results.detections[0]
                        bboxC = detection.location_data.relative_bounding_box
                        target_x = (bboxC.xmin + bboxC.width / 2.0) * self.width
                        target_y = (bboxC.ymin + bboxC.height / 2.0) * self.height

                # Fallback target if no face detected in current frame
                if target_x is None:
                    target_x = self.width / 2.0
                    target_y = self.height / 2.0

                # Non-blocking push to result queue
                if not self.result_queue.full():
                    self.result_queue.put({"target_x": target_x, "target_y": target_y, "frame": frame})

            else:
                # Simulated circular target movement when no camera is present
                sim_angle += 0.05
                target_x = self.width / 2.0 + np.cos(sim_angle) * 200.0
                target_y = self.height / 2.0 + np.sin(sim_angle) * 150.0

                dummy_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                cv2.putText(
                    dummy_frame,
                    "Simulated Vision Mode (No Camera)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

                if not self.result_queue.full():
                    self.result_queue.put({"target_x": target_x, "target_y": target_y, "frame": dummy_frame})

                time.sleep(0.033)  # ~30 FPS vision update rate

        if cap and cap.isOpened():
            cap.release()

    def stop(self):
        self.running = False
