"""
Visual AI Game Engine - Developer Demo Example
Demonstrates how game developers import and use the visual_ai library.
"""

import queue
import time

import cv2
import numpy as np

from visual_ai import CPP_ENGINE_AVAILABLE, GameEngine, VisionPipeline, display


def main():
    WIDTH, HEIGHT = 1920, 1080

    print(f"[Demo] C++ Core Acceleration: "
          f"{'Enabled' if CPP_ENGINE_AVAILABLE else 'Disabled (Fallback)'}")

    # Initialize Engine (C++ if built, fallback otherwise)
    engine = GameEngine(float(WIDTH), float(HEIGHT))

    # Spawn a few blocks for demonstration
    engine.add_block(200, HEIGHT - 30, 40, 60, 100.0)
    engine.add_block(240, HEIGHT - 30, 40, 60, 100.0)
    engine.add_block(220, HEIGHT - 90, 40, 60, 100.0)
    engine.add_block(600, HEIGHT - 30, 40, 60, 100.0)

    # Thread-safe queue for camera AI results
    ai_queue = queue.Queue(maxsize=1)

    # Start Vision Pipeline in background thread
    pipeline = VisionPipeline(result_queue=ai_queue, width=WIDTH, height=HEIGHT)
    pipeline.start()

    print("[Demo] Engine running. Press 'q' or 'ESC' to quit.")

    last_time = time.time()
    frame_count = 0
    fps = 0.0
    fps_timer = time.time()

    current_frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    # Latest gesture snapshot
    gesture = {
        "hand_visible":      False,
        "index_pos":         (0, 0),
        "pinch_pos":         (0, 0),
        "is_pinching":       False,
        "click_just_fired":  False,
        "is_index_isolated": False,
        "z_delta":           0.0,
    }

    try:
        while True:
            now = time.time()
            dt = now - last_time
            last_time = now

            # 1. Drain to the freshest payload — the SDK contract every game
            #    follows: act on the newest frame, never a queued stale one.
            ai_data = None
            while True:
                try:
                    ai_data = ai_queue.get_nowait()
                except queue.Empty:
                    break
            if ai_data is not None:
                target_x = ai_data["target_x"]
                target_y = ai_data["target_y"]
                if ai_data["frame"] is not None:
                    current_frame = ai_data["frame"]

                # Update gesture snapshot
                for key in gesture:
                    if key in ai_data:
                        gesture[key] = ai_data[key]

                # Pass vision coordinates to Engine
                engine.set_target_position(target_x, target_y)

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

            # Draw Blocks
            for block in engine.get_blocks():
                if block.active:
                    bx_b, by_b = int(block.x), int(block.y)
                    bw, bh = int(block.width), int(block.height)
                    cv2.rectangle(render_canvas, (bx_b - bw//2, by_b - bh//2),
                                  (bx_b + bw//2, by_b + bh//2), (0, 150, 200), -1)
                    cv2.rectangle(render_canvas, (bx_b - bw//2, by_b - bh//2),
                                  (bx_b + bw//2, by_b + bh//2), (0, 50, 100), 2)

            # Draw Debris
            for d in engine.get_debris():
                if d.active:
                    dx, dy, dw, dh = int(d.x), int(d.y), int(d.width), int(d.height)
                    alpha = max(0.0, min(1.0, d.lifespan / 3.0))
                    color = (0, int(100 * alpha), int(200 * alpha))
                    cv2.rectangle(render_canvas, (dx - dw//2, dy - dh//2),
                                  (dx + dw//2, dy + dh//2), color, -1)
                    cv2.rectangle(render_canvas, (dx - dw//2, dy - dh//2),
                                  (dx + dw//2, dy + dh//2), (0, 50, 100), 1)

            # Draw Physics Sprite (Ball)
            bx, by = int(engine.get_x()), int(engine.get_y())
            cv2.circle(render_canvas, (bx, by), 25, (0, 0, 255), -1)
            cv2.circle(render_canvas, (bx, by), 25, (255, 255, 255), 2)

            # ── Hand gesture overlay ──────────────────────────────────────────
            if gesture["hand_visible"]:
                ix, iy = gesture["index_pos"]
                px, py = gesture["pinch_pos"]

                # Index-fingertip ring
                cv2.circle(render_canvas, (ix, iy), 12, (0, 255, 200), 2)
                cv2.circle(render_canvas, (ix, iy),  4, (0, 255, 200), -1)

                # Pinch indicator (filled dot at midpoint)
                if gesture["is_pinching"]:
                    cv2.circle(render_canvas, (px, py), 14, (0, 200, 255), -1)
                    cv2.circle(render_canvas, (px, py), 18, (0, 200, 255),  2)
                    cv2.putText(render_canvas, "PINCH", (px + 22, py - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2, cv2.LINE_AA)

                # Z-push click flash
                if gesture["click_just_fired"]:
                    cv2.circle(render_canvas, (ix, iy), 30, (0, 80, 255), 3)
                    cv2.putText(render_canvas, "CLICK!", (ix + 22, iy - 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 2, cv2.LINE_AA)

                # Z-delta bar (debug)
                z_bar = int(min(abs(gesture["z_delta"]) / 0.05 * 80, 80))
                bar_color = (0, 80, 255) if gesture["z_delta"] > 0 else (120, 120, 120)
                cv2.rectangle(render_canvas, (WIDTH - 20, HEIGHT - 10),
                              (WIDTH - 10, HEIGHT - 10 - z_bar), bar_color, -1)
                cv2.putText(render_canvas, "Z", (WIDTH - 22, HEIGHT - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # Draw HUD
            engine_type = "C++ Core" if CPP_ENGINE_AVAILABLE else "Python Fallback"
            cv2.putText(render_canvas, f"Engine: {engine_type} | FPS: {fps:.1f}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(render_canvas, f"Sprite: ({bx}, {by}) | Target: ({tx}, {ty})", (15, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            hand_label = "Hand: " + (
                ("PINCH" if gesture["is_pinching"] else
                 "INDEX" if gesture["is_index_isolated"] else "OPEN")
                if gesture["hand_visible"] else "None"
            )
            cv2.putText(render_canvas, hand_label, (15, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1)

            # Display window. `27` is also what closing the window reports, so
            # the X button now quits the demo instead of killing it mid-frame.
            key = display.show("Visual AI Library Demo", render_canvas)

            # Handle exit
            if key in (27, ord('q')):
                break

    finally:
        pipeline.stop()
        display.close_all()
        print("[Demo] Engine stopped cleanly.")


if __name__ == "__main__":
    main()
