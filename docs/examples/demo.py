"""
Visual AI Game Engine - Developer Demo Example
Demonstrates how game developers import and use the visual_ai library.
"""

import time
import queue
import cv2
import numpy as np

from visual_ai import VisionPipeline, GameEngine, CPP_ENGINE_AVAILABLE


def main():
    WIDTH, HEIGHT = 800, 600

    print(f"[Demo] C++ Core Acceleration: {'Enabled' if CPP_ENGINE_AVAILABLE else 'Disabled (Fallback)'}")

    # Initialize Engine (C++ if built, fallback otherwise)
    engine = GameEngine(float(WIDTH), float(HEIGHT))

    # Thread-safe queue for camera AI results
    ai_queue = queue.Queue(maxsize=2)

    # Start Vision Pipeline in background thread
    pipeline = VisionPipeline(result_queue=ai_queue, width=WIDTH, height=HEIGHT)
    pipeline.start()

    print("[Demo] Engine running. Press 'q' or 'ESC' to quit.")

    last_time = time.time()
    frame_count = 0
    fps = 0.0
    fps_timer = time.time()

    current_frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    try:
        while True:
            now = time.time()
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
                pass

            # 2. Update Engine physics
            engine.update(dt)

            # 3. Calculate FPS
            frame_count += 1
            if now - fps_timer >= 1.0:
                fps = frame_count / (now - fps_timer)
                frame_count = 0
                fps_timer = now

            # 4. Render visual overlay
            render_canvas = current_frame.copy()

            # Draw AI Target (Crosshair)
            tx, ty = int(engine.get_target_x()), int(engine.get_target_y())
            cv2.drawMarker(render_canvas, (tx, ty), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.circle(render_canvas, (tx, ty), 15, (0, 255, 0), 1)

            # Draw Physics Sprite (Ball)
            bx, by = int(engine.get_x()), int(engine.get_y())
            cv2.circle(render_canvas, (bx, by), 25, (0, 0, 255), -1)
            cv2.circle(render_canvas, (bx, by), 25, (255, 255, 255), 2)

            # Draw HUD
            engine_type = "C++ Core" if CPP_ENGINE_AVAILABLE else "Python Fallback"
            cv2.putText(render_canvas, f"Engine: {engine_type} | FPS: {fps:.1f}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(render_canvas, f"Sprite: ({bx}, {by}) | Target: ({tx}, {ty})", (15, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Display window
            cv2.imshow("Visual AI Library Demo", render_canvas)

            # Handle exit
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[Demo] Engine stopped cleanly.")


if __name__ == "__main__":
    main()
