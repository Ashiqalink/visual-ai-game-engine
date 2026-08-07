import sys
import os
import time
import queue
import cv2
import numpy as np

# Ensure project root and build directory are on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import C++ Core Engine module (pybind11 compiled extension)
try:
    import engine_core
    print("[Main] Successfully loaded C++ engine_core module!")
    CPP_ENGINE_AVAILABLE = True
except ImportError as e:
    print(f"[Main] Notice: C++ engine_core extension not compiled yet ({e}).")
    print("[Main] Using Python fallback engine for testing until C++ module is compiled.")
    CPP_ENGINE_AVAILABLE = False

    class PythonFallbackEngine:
        def __init__(self, width=800.0, height=600.0):
            self.width = width
            self.height = height
            self.x = width / 2.0
            self.y = height / 4.0
            self.vx = 120.0
            self.vy = 0.0
            self.gravity = 400.0
            self.radius = 25.0
            self.target_x = width / 2.0
            self.target_y = height / 2.0

        def set_target_position(self, x, y):
            self.target_x = x
            self.target_y = y
            
        def set_gravity(self, g): self.gravity = g
        def get_gravity(self): return self.gravity
        def set_radius(self, r): self.radius = r
        def get_radius(self): return self.radius
        def set_velocity(self, vx, vy):
            self.vx = vx
            self.vy = vy

        def update(self, dt):
            self.vy += self.gravity * dt
            dx = self.target_x - self.x
            dy = self.target_y - self.y
            dist = np.sqrt(dx * dx + dy * dy)
            if dist > 1.0:
                pull_strength = 150.0
                self.vx += (dx / dist) * pull_strength * dt
                self.vy += (dy / dist) * pull_strength * dt
            self.x += self.vx * dt
            self.y += self.vy * dt
            self.vx *= 0.99
            if self.x - self.radius < 0:
                self.x = self.radius
                self.vx = -self.vx * 0.8
            elif self.x + self.radius > self.width:
                self.x = self.width - self.radius
                self.vx = -self.vx * 0.8
            if self.y - self.radius < 0:
                self.y = self.radius
                self.vy = -self.vy * 0.8
            elif self.y + self.radius > self.height:
                self.y = self.height - self.radius
                self.vy = -self.vy * 0.8

        def get_x(self): return self.x
        def get_y(self): return self.y
        def get_target_x(self): return self.target_x
        def get_target_y(self): return self.target_y
        def get_width(self): return self.width
        def get_height(self): return self.height


from python.pipeline import VisionPipeline

def main():
    WIDTH, HEIGHT = 800, 600
    TARGET_FPS = 30.0
    FRAME_TARGET_TIME = 1.0 / TARGET_FPS  # ~0.0333 seconds (33.3 ms per frame)
    
    # Initialize Engine (C++ if built, fallback otherwise)
    if CPP_ENGINE_AVAILABLE:
        engine = engine_core.GameEngine(float(WIDTH), float(HEIGHT))
    else:
        engine = PythonFallbackEngine(float(WIDTH), float(HEIGHT))

    # Thread-safe queue for camera AI results (Producer-drop overflow model)
    ai_queue = queue.Queue(maxsize=2)
    
    # Start Vision Pipeline in background thread
    pipeline = VisionPipeline(result_queue=ai_queue, width=WIDTH, height=HEIGHT)
    pipeline.start()

    print(f"[Main] Engine loop running with strict target {TARGET_FPS} FPS cap. Press 'q' or 'ESC' to quit.")

    last_time = time.time()
    frame_count = 0
    fps = 0.0
    fps_timer = time.time()

    current_frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    try:
        while True:
            frame_start_time = time.time()
            now = frame_start_time
            dt = now - last_time
            last_time = now

            # 1. Non-blocking queue check for AI vision updates
            try:
                ai_data = ai_queue.get_nowait()
                target_x = ai_data["target_x"]
                target_y = ai_data["target_y"]
                if ai_data["frame"] is not None:
                    current_frame = ai_data["frame"]
                
                # Pass vision coordinates to Engine
                engine.set_target_position(target_x, target_y)
            except queue.Empty:
                pass  # Engine continues seamlessly if AI frame isn't ready yet

            # 2. Update Engine physics
            engine.update(dt)

            # 3. Calculate FPS
            frame_count += 1
            if now - fps_timer >= 1.0:
                fps = frame_count / (now - fps_timer)
                frame_count = 0
                fps_timer = now

            # 4. Render visual overlay (draw directly on current_frame, no redundant copy)
            render_canvas = current_frame

            # Draw AI Target (Crosshair)
            tx, ty = int(engine.get_target_x()), int(engine.get_target_y())
            cv2.drawMarker(render_canvas, (tx, ty), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.circle(render_canvas, (tx, ty), 15, (0, 255, 0), 1)

            # Draw C++ Physics Sprite (Ball)
            bx, by = int(engine.get_x()), int(engine.get_y())
            cv2.circle(render_canvas, (bx, by), 25, (0, 0, 255), -1)
            cv2.circle(render_canvas, (bx, by), 25, (255, 255, 255), 2)

            # Draw HUD
            engine_type = "C++ Core" if CPP_ENGINE_AVAILABLE else "Python Fallback"
            cv2.putText(render_canvas, f"Engine: {engine_type} | FPS: {fps:.1f} (Cap: 30)", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(render_canvas, f"Sprite: ({bx}, {by}) | Vision Target: ({tx}, {ty})", (15, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Display window
            cv2.imshow("Visual AI Game Engine v1.0", render_canvas)

            # Handle exit without blocking
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

            # 5. Frame pacing: enforce strict 30 FPS cap
            elapsed = time.time() - frame_start_time
            sleep_time = FRAME_TARGET_TIME - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[Main] Engine stopped cleanly.")

if __name__ == "__main__":
    main()
