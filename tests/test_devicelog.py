"""
test_devicelog.py — the play-time device log.

Every window here is driven by a hand-advanced clock, so the fps, the window
length and the counter deltas are numbers the test knows in advance. A log
whose output cannot be predicted cannot be checked, and this one exists
specifically to be *read back* as evidence about a live session — so its
arithmetic has to be worth trusting.

The opt-in is checked as hard as the arithmetic: this writes files onto the
player's disk, and the default has to be that it does not.
"""

import json
import os
import tempfile
import unittest

from visual_ai import devicelog

#: A frame step that is exact in binary: 30 additions of 1/30 fall a hair short
#: of a second, and a test whose window boundary lands on that hair is testing
#: float addition, not the log.
_STEP = 1.0 / 32
_FPS = 32.0


class _Clock:
    """A clock the test advances by hand."""

    def __init__(self, start=0.0):
        self.t = float(start)

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += float(seconds)


class _TempLog(unittest.TestCase):
    """A log in a throwaway directory, with the environment cleared."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name
        self.path = os.path.join(self.dir, "device.jsonl")
        self.clock = _Clock()
        for key in (devicelog.ENV_ENABLED, devicelog.ENV_DIR):
            previous = os.environ.pop(key, None)
            if previous is not None:
                self.addCleanup(os.environ.__setitem__, key, previous)

    def log(self, **kwargs):
        kwargs.setdefault("path", self.path)
        kwargs.setdefault("directory", self.dir)
        kwargs.setdefault("is_enabled", True)
        kwargs.setdefault("interval_s", 5.0)
        kwargs.setdefault("clock", self.clock)
        kwargs.setdefault("now", lambda: 1_700_000_000.0)
        return devicelog.DeviceLog(**kwargs)

    def records(self):
        return devicelog.load(self.path)


class TestOptIn(_TempLog):

    def test_logging_is_off_unless_it_was_asked_for(self):
        self.assertFalse(devicelog.enabled(self.dir))

    def test_the_marker_turns_it_on_for_this_machine(self):
        devicelog.enable_here(self.dir)
        self.assertTrue(devicelog.enabled(self.dir))

    def test_the_environment_turns_it_on_for_one_run(self):
        os.environ[devicelog.ENV_ENABLED] = "1"
        self.addCleanup(os.environ.pop, devicelog.ENV_ENABLED, None)
        self.assertTrue(devicelog.enabled(self.dir))

    def test_an_explicit_off_beats_the_marker(self):
        devicelog.enable_here(self.dir)
        os.environ[devicelog.ENV_ENABLED] = "0"
        self.addCleanup(os.environ.pop, devicelog.ENV_ENABLED, None)
        self.assertFalse(devicelog.enabled(self.dir))

    def test_disable_here_removes_the_marker_and_leaves_the_logs(self):
        devicelog.enable_here(self.dir)
        log = self.log()
        log.frame(latency_ms=5.0)
        log.close()
        self.assertTrue(devicelog.disable_here(self.dir))
        self.assertFalse(devicelog.enabled(self.dir))
        self.assertTrue(self.records())

    def test_a_disabled_log_writes_no_file_at_all(self):
        log = self.log(is_enabled=False)
        log.session(hand_device="NPU")
        for _ in range(100):
            log.frame(latency_ms=4.0, hands=1)
        self.clock.advance(60.0)
        log.close()
        self.assertFalse(os.path.exists(self.path))
        self.assertEqual(log.status(), "device log off")


class TestSessionHeader(_TempLog):

    def test_the_header_carries_the_device_and_its_fallback_reason(self):
        log = self.log()
        log.session(hand_device="CPU (MediaPipe) (npu unavailable)",
                    devices=["CPU", "GPU"], accel_preset="npu")
        header = self.records()[0]
        self.assertEqual(header["kind"], "session")
        self.assertEqual(header["hand_device"],
                         "CPU (MediaPipe) (npu unavailable)")
        self.assertEqual(header["devices"], ["CPU", "GPU"])
        self.assertEqual(header["accel_preset"], "npu")


class TestWindows(_TempLog):

    def test_a_window_closes_on_the_interval_and_reports_its_own_fps(self):
        log = self.log(interval_s=5.0)
        for _ in range(160):                 # 160 frames over exactly 5 s
            self.clock.advance(_STEP)
            log.frame(latency_ms=4.0, hands=1)
        samples = [r for r in self.records() if r["kind"] == "sample"]
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["frames"], 160)
        self.assertAlmostEqual(samples[0]["seconds"], 5.0, places=3)
        self.assertAlmostEqual(samples[0]["fps"], _FPS, places=1)

    def test_no_window_closes_before_the_interval_is_up(self):
        log = self.log(interval_s=5.0)
        for _ in range(16):
            self.clock.advance(_STEP)         # half a second of play
            log.frame(latency_ms=4.0, hands=1)
        self.assertEqual(self.records(), [])

    def test_windows_tile_the_session_rather_than_overlapping(self):
        log = self.log(interval_s=1.0)
        for _ in range(320):                 # 10 s at 32 fps -> ten windows
            self.clock.advance(_STEP)
            log.frame(latency_ms=4.0, hands=1)
        samples = [r for r in self.records() if r["kind"] == "sample"]
        self.assertEqual(len(samples), 10)
        self.assertEqual([s["sample"] for s in samples], list(range(1, 11)))
        self.assertAlmostEqual(sum(s["seconds"] for s in samples), 10.0, places=2)
        self.assertEqual(sum(s["frames"] for s in samples), 320)

    def test_latency_percentiles_describe_the_window_not_the_session(self):
        log = self.log(interval_s=1.0)
        # A clean first second, then a second one that stutters.
        for _ in range(32):
            self.clock.advance(_STEP)
            log.frame(latency_ms=4.0, hands=1)
        for index in range(32):
            self.clock.advance(_STEP)
            log.frame(latency_ms=4.0 if index < 30 else 40.0, hands=1)
        samples = [r for r in self.records() if r["kind"] == "sample"]
        self.assertEqual(samples[0]["latency_ms"]["p95"], 4.0)
        self.assertEqual(samples[0]["latency_ms"]["max"], 4.0)
        self.assertEqual(samples[1]["latency_ms"]["p50"], 4.0)
        self.assertEqual(samples[1]["latency_ms"]["max"], 40.0)
        self.assertGreater(samples[1]["latency_ms"]["mean"],
                           samples[0]["latency_ms"]["mean"])

    def test_hands_seen_is_averaged_over_the_window(self):
        log = self.log(interval_s=1.0)
        for index in range(32):
            self.clock.advance(_STEP)
            log.frame(latency_ms=4.0, hands=index % 2)      # half the frames
        sample = [r for r in self.records() if r["kind"] == "sample"][0]
        self.assertAlmostEqual(sample["hands_mean"], 0.5, places=2)


class TestCounters(_TempLog):
    """
    The backend's `timings` dict is cumulative; a window has to report the
    difference. Writing the running total instead would make every window after
    the first look worse than the one before it.
    """

    def test_a_window_reports_the_delta_of_a_cumulative_counter(self):
        log = self.log(interval_s=1.0)
        timings = {"palm_ms": 0.0, "land_ms": 0.0, "detector_runs": 0, "frames": 0}
        for window in range(3):
            for _ in range(32):
                self.clock.advance(_STEP)
                timings["land_ms"] += 3.0
                timings["frames"] += 1
                log.frame(latency_ms=4.0, hands=1, counters=timings)
            timings["detector_runs"] += 1
        samples = [r for r in self.records() if r["kind"] == "sample"]
        self.assertEqual(len(samples), 3)
        for sample in samples:
            self.assertAlmostEqual(sample["counters"]["land_ms"], 96.0, places=3)
            self.assertEqual(sample["counters"]["frames"], 32)
        # The detector re-ran once per window, at the window boundary.
        self.assertEqual([s["counters"]["detector_runs"] for s in samples],
                         [0, 1, 1])

    def test_the_close_record_carries_the_session_totals(self):
        log = self.log(interval_s=1.0)
        timings = {"palm_ms": 0.0, "detector_runs": 0}
        for _ in range(64):
            self.clock.advance(_STEP)
            timings["palm_ms"] += 2.0
            timings["detector_runs"] += 1
            log.frame(latency_ms=4.0, hands=1, counters=timings)
        log.close(hand_device="NPU")
        closing = [r for r in self.records() if r["kind"] == "close"][0]
        self.assertEqual(closing["frames"], 64)
        self.assertEqual(closing["samples"], 2)
        self.assertAlmostEqual(closing["seconds"], 2.0, places=3)
        self.assertAlmostEqual(closing["counters"]["palm_ms"], 128.0, places=3)
        self.assertEqual(closing["counters"]["detector_runs"], 64)
        self.assertEqual(closing["hand_device"], "NPU")

    def test_a_backend_with_no_counters_still_logs(self):
        log = self.log(interval_s=1.0)
        for _ in range(32):
            self.clock.advance(_STEP)
            log.frame(latency_ms=4.0, hands=1)
        sample = [r for r in self.records() if r["kind"] == "sample"][0]
        self.assertNotIn("counters", sample)


class TestClose(_TempLog):

    def test_close_flushes_the_part_window_so_a_short_run_is_not_lost(self):
        log = self.log(interval_s=60.0)
        for _ in range(32):
            self.clock.advance(_STEP)
            log.frame(latency_ms=4.0, hands=1)
        log.close()
        kinds = [r["kind"] for r in self.records()]
        self.assertEqual(kinds, ["sample", "close"])

    def test_close_twice_writes_one_set_of_records(self):
        log = self.log()
        self.clock.advance(1.0)
        log.frame(latency_ms=4.0, hands=1)
        log.close()
        log.close()
        self.assertEqual([r["kind"] for r in self.records()], ["sample", "close"])

    def test_a_session_that_never_saw_a_frame_writes_no_empty_window(self):
        log = self.log()
        log.session(hand_device="NPU")
        log.close()
        self.assertEqual([r["kind"] for r in self.records()], ["session", "close"])


class TestFailureIsVisible(_TempLog):
    """A logger that quietly stopped writing looks like a quiet session."""

    def blocked_path(self):
        """A log path whose parent directory cannot exist: a file is there."""
        blocker = os.path.join(self.dir, "blocker")
        with open(blocker, "w", encoding="utf-8") as handle:
            handle.write("not a directory")
        return os.path.join(blocker, "device.jsonl")

    def test_an_unwritable_path_disables_the_log_instead_of_raising(self):
        log = self.log(path=self.blocked_path())
        log.session(hand_device="NPU")
        self.assertFalse(log.enabled)
        self.assertIsNotNone(log.error)

    def test_the_failure_is_reported_on_the_status_line(self):
        log = self.log(path=self.blocked_path())
        log.session(hand_device="NPU")
        self.assertTrue(log.status().startswith("device log off ("))

    def test_play_carries_on_after_the_failure(self):
        log = self.log(path=self.blocked_path())
        log.session(hand_device="NPU")
        for _ in range(100):
            self.clock.advance(_STEP)
            log.frame(latency_ms=4.0, hands=1)
        log.close()          # no exception is the assertion

    def test_a_working_log_counts_what_it_wrote(self):
        log = self.log(interval_s=1.0)
        log.session(hand_device="NPU")
        for _ in range(32):
            self.clock.advance(_STEP)
            log.frame(latency_ms=4.0, hands=1)
        self.assertEqual(log.records_written, 2)
        self.assertIn("device.jsonl", log.status())


class TestLoad(_TempLog):

    def test_a_run_killed_mid_write_does_not_lose_the_lines_before_it(self):
        log = self.log(interval_s=1.0)
        log.session(hand_device="NPU")
        for _ in range(32):
            self.clock.advance(_STEP)
            log.frame(latency_ms=4.0, hands=1)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write('{"kind": "sample", "frames": 1')     # torn line
        records = self.records()
        self.assertEqual([r["kind"] for r in records], ["session", "sample"])

    def test_loading_a_log_that_was_never_written_is_empty_not_an_error(self):
        self.assertEqual(devicelog.load(os.path.join(self.dir, "nothing.jsonl")), [])

    def test_every_record_is_one_json_line(self):
        log = self.log(interval_s=1.0)
        log.session(hand_device="NPU")
        self.clock.advance(1.0)
        log.frame(latency_ms=4.0, hands=1)
        log.close()
        with open(self.path, encoding="utf-8") as handle:
            lines = [line for line in handle if line.strip()]
        self.assertEqual(len(lines), 3)
        for line in lines:
            json.loads(line)


class TestDefaults(_TempLog):

    def test_the_directory_follows_the_environment_override(self):
        os.environ[devicelog.ENV_DIR] = self.dir
        self.addCleanup(os.environ.pop, devicelog.ENV_DIR, None)
        self.assertEqual(devicelog.log_dir(), self.dir)
        self.assertEqual(devicelog.marker_path(),
                         os.path.join(self.dir, devicelog.MARKER_NAME))

    def test_a_default_log_on_a_machine_with_no_opt_in_is_inert(self):
        os.environ[devicelog.ENV_DIR] = self.dir
        self.addCleanup(os.environ.pop, devicelog.ENV_DIR, None)
        log = devicelog.DeviceLog()
        self.assertFalse(log.enabled)
        self.assertTrue(log.path.endswith(".jsonl"))
        self.assertFalse(os.path.exists(log.path))

    def test_the_default_filename_is_dated(self):
        os.environ[devicelog.ENV_DIR] = self.dir
        self.addCleanup(os.environ.pop, devicelog.ENV_DIR, None)
        log = devicelog.DeviceLog()
        name = os.path.basename(log.path)
        self.assertTrue(name.startswith("device-"), name)
        self.assertEqual(len(name), len("device-2026-08-24.jsonl"))


if __name__ == "__main__":
    unittest.main()
