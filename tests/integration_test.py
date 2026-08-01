import unittest
import queue
import time
from visual_ai import VisionPipeline, GameEngine, CPP_ENGINE_AVAILABLE


class TestPipelineIntegration(unittest.TestCase):
    def test_pipeline_to_engine_flow(self):
        """Test full pipeline loop feeding AI vision target coordinates into GameEngine."""
        WIDTH, HEIGHT = 600, 400
        ai_queue = queue.Queue(maxsize=5)

        # Initialize Vision Pipeline in background thread
        pipeline = VisionPipeline(result_queue=ai_queue, width=WIDTH, height=HEIGHT, camera_index=-1)
        pipeline.start()

        engine = GameEngine(float(WIDTH), float(HEIGHT))

        received_frames = 0
        timeout = time.time() + 2.0  # Run test for up to 2 seconds

        try:
            while time.time() < timeout and received_frames < 10:
                try:
                    data = ai_queue.get_nowait()
                    target_x = data["target_x"]
                    target_y = data["target_y"]
                    engine.set_target_position(target_x, target_y)
                    received_frames += 1
                except queue.Empty:
                    time.sleep(0.01)

                engine.update(0.016)

            self.assertGreater(received_frames, 0, "Pipeline should send at least 1 vision coordinate frame")
            self.assertIsNotNone(engine.get_target_x())
            self.assertIsNotNone(engine.get_target_y())

        finally:
            pipeline.stop()
            pipeline.join(timeout=1.0)


if __name__ == "__main__":
    unittest.main()
