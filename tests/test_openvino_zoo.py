"""
Decoders and device policy of `visual_ai.openvino_zoo`, without hardware.

Every network here is exercised on the real devices by
``benchmarks/bench_npu_zoo.py``; these tests pin the pure parts — anchor
grids, box decoding, letterbox undo, top-k, the CPU fallback note — so a
wrong number shows up as a failed assertion and not as a box in the wrong
corner of the frame.
"""
import hashlib
import math

import numpy as np
import pytest

from visual_ai import openvino_zoo as zoo


def _bare(cls):
    """An instance with no network behind it, for the decode methods."""
    return cls.__new__(cls)


# ── anchors and SSD decoding ─────────────────────────────────────────────────

def test_anchor_counts_match_mediapipe_graphs():
    assert len(zoo._anchors(128, ((8, 2), (16, 6)))) == 896            # BlazeFace short range
    assert len(zoo._anchors(224, ((8, 2), (16, 2), (32, 6)))) == 2254  # pose detection
    assert len(zoo._anchors(192, ((8, 2), (16, 6)))) == 2016           # palm, as openvino_hands


def test_anchor_centres_sit_in_cell_centres():
    anchors = zoo._anchors(16, ((8, 1),))
    assert anchors.shape == (4, 2)
    np.testing.assert_allclose(anchors[0], (0.25, 0.25))
    np.testing.assert_allclose(anchors[3], (0.75, 0.75))


def test_decode_ssd_offsets_box_and_keypoints_from_anchor():
    anchors = np.array([[0.5, 0.5], [0.25, 0.25]], np.float32)
    boxes = np.zeros((2, 16), np.float32)
    boxes[0, :4] = (12.8, -12.8, 64.0, 32.0)      # +0.1, -0.1, w 0.5, h 0.25 at size 128
    boxes[0, 4:6] = (12.8, 12.8)                  # keypoint 0 at anchor + (0.1, 0.1)
    logits = np.array([[3.0], [-3.0]], np.float32)
    found = zoo._decode_ssd(boxes, logits, anchors, 128, 6, 0.5)
    assert len(found) == 1
    got = found[0]
    assert got["score"] == pytest.approx(1 / (1 + math.exp(-3.0)), abs=1e-6)
    np.testing.assert_allclose(got["box"], (0.35, 0.275, 0.85, 0.525), atol=1e-6)
    np.testing.assert_allclose(got["kp"][0], (0.6, 0.6), atol=1e-6)
    assert got["kp"].shape == (6, 2)


def test_decode_ssd_returns_nothing_below_threshold():
    anchors = np.zeros((3, 2), np.float32)
    found = zoo._decode_ssd(np.zeros((3, 12), np.float32),
                            np.full((3, 1), -9.0, np.float32), anchors, 224, 4, 0.5)
    assert found == []


def test_unletterbox_returns_frame_pixels():
    # 640x480 into a 128 square: scale 0.2, no side bars, 16 px top bar.
    detections = [{"score": 1.0, "kp": np.array([[0.5, 0.5]], np.float32),
                   "box": (0.0, 0.125, 1.0, 0.875)}]
    out = zoo._unletterbox(detections, 128, 0.2, 0, 16)
    np.testing.assert_allclose(out[0]["kp"][0], (320, 240), atol=1e-4)
    np.testing.assert_allclose(out[0]["box"], (0, 0, 640, 480), atol=1e-3)


def test_split_ssd_outputs_accepts_either_port_order():
    boxes, logits = np.zeros((1, 896, 16)), np.zeros((1, 896, 1))
    got_b, got_l = zoo._split_ssd_outputs({"a": logits, "b": boxes}, 16)
    assert got_b.shape == (896, 16) and got_l.shape == (896, 1)
    got_b, got_l = zoo._split_ssd_outputs({"b": boxes, "a": logits}, 16)
    assert got_b.shape == (896, 16) and got_l.shape == (896, 1)


# ── per-network decoders ─────────────────────────────────────────────────────

def test_yolo_decode_undoes_letterbox_and_names_classes():
    det = _bare(zoo.ObjectDetector)
    det.min_score = 0.35
    raw = np.array([
        [64, 48, 192, 144, 0.9, 0],       # person
        [0, 0, 10, 10, 0.1, 5],           # below threshold
        [10, 10, 20, 20, 0.5, 999],       # class index outside COCO
    ], np.float32)
    found = det._decode(raw, scale=0.5, left=0, top=32)
    assert [d.label for d in found] == ["person", "999"]
    np.testing.assert_allclose(found[0].box, (128, 32, 384, 224))
    assert found[0].center == (256, 128)


def test_ssd0200_decode_scales_boxes_and_skips_padding_rows():
    det = _bare(zoo.SSDDetector)
    det.min_score, det.kind = 0.5, "person"
    raw = np.array([
        [0, 1, 0.9, 0.25, 0.5, 0.75, 1.0],
        [0, 1, 0.2, 0.0, 0.0, 1.0, 1.0],   # below threshold
        [-1, 0, 0, 0, 0, 0, 0],           # end-of-list padding
    ], np.float32)
    found = det._decode(raw, 640, 480)
    assert len(found) == 1 and found[0].label == "person"
    np.testing.assert_allclose(found[0].box, (160, 240, 480, 480))


def test_classifier_topk_is_softmax_in_descending_order():
    clf = _bare(zoo.ImageClassifier)
    clf.top_k, clf.labels = 2, ["a", "b", "c"]
    top = clf._decode(np.array([1.0, 3.0, 2.0]))
    assert [label for label, _ in top] == ["b", "c"]
    denominator = math.exp(1) + math.exp(2) + math.exp(3)
    assert top[0][1] == pytest.approx(math.exp(3) / denominator)
    assert top[1][1] == pytest.approx(math.exp(2) / denominator)


def test_heatmap_decode_maps_peak_cell_to_frame_pixels():
    pose = _bare(zoo.HeatmapPose)
    pose.min_score = 0.3
    heat = np.zeros((19, 32, 57), np.float32)
    heat[0, 16, 28] = 0.8         # nose
    heat[1, 0, 0] = 0.1           # neck, below threshold
    parts = pose._decode(heat, 456, 256)
    assert set(parts) == {"nose"}
    assert parts["nose"] == pytest.approx((28.5 / 57 * 456, 16.5 / 32 * 256, 0.8))


def test_alignment_rect_is_square_upright_and_scaled():
    cx, cy, size, angle = zoo._alignment_rect((100, 100), (100, 50), 1.25)
    assert (cx, cy) == (100, 100)
    assert size == pytest.approx(125.0)
    assert angle == pytest.approx(0.0)
    _, _, _, angle = zoo._alignment_rect((100, 100), (150, 100), 1.0)
    assert angle == pytest.approx(math.pi / 2)


def test_bodypose_decode_maps_crop_to_frame_and_reads_presence():
    pose = _bare(zoo.BodyPose)
    pose._rect = (100.0, 100.0, 512.0, 0.0)
    raw = np.zeros((39, 5), np.float32)
    raw[:, :2] = 128.0                    # every point at the crop centre
    raw[:, 3] = 5.0                       # visible
    raw[34, :2] = (128.0, 0.0)
    inverse = np.array([[2.0, 0.0, -156.0], [0.0, 2.0, -156.0]], np.float32)
    outputs = {"Identity": raw.reshape(1, 195),
               "Identity_1": np.array([[0.95]], np.float32),
               "Identity_2": np.zeros((1, 256, 256, 1), np.float32)}
    result = pose._decode(outputs, inverse)
    assert result.landmarks.shape == (33, 4)
    assert result.landmarks_all.shape == (39, 4)
    np.testing.assert_allclose(result.landmarks[0, :2], (100.0, 100.0))
    assert result.landmarks[0, 3] > 0.99
    assert result.score == pytest.approx(0.95)
    assert result.crop_mask.shape == (256, 256)
    assert result.crop_mask[0, 0] == pytest.approx(0.5)
    assert result.point("nose") == (100.0, 100.0)


def test_pose_result_mask_warps_crop_mask_over_frame():
    landmarks = np.zeros((33, 4), np.float32)
    crop = np.ones((256, 256), np.float32)
    inverse = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]], np.float32)   # 256 -> 128 px
    result = zoo.PoseResult(landmarks, 0.9, (64, 64, 128, 0.0), crop, inverse)
    mask = result.mask((100, 200))
    assert mask.shape == (100, 200)
    assert mask[:120, :120].min() == pytest.approx(1.0)
    assert mask[-1, -1] == 0.0


# ── device policy ────────────────────────────────────────────────────────────

def test_resolve_device_follows_preset_order(monkeypatch):
    monkeypatch.delenv(zoo.ENV_DEVICE, raising=False)
    monkeypatch.setattr(zoo.accel, "preset", lambda: "auto")
    monkeypatch.setattr(zoo.accel, "available_devices", lambda: ("CPU", "GPU", "NPU"))
    assert zoo.resolve_device() == "NPU"
    monkeypatch.setattr(zoo.accel, "available_devices", lambda: ("CPU", "GPU"))
    assert zoo.resolve_device() == "GPU"
    monkeypatch.setattr(zoo.accel, "preset", lambda: "off")
    assert zoo.resolve_device() == "CPU"
    monkeypatch.setattr(zoo.accel, "preset", lambda: "gpu")
    monkeypatch.setattr(zoo.accel, "available_devices", lambda: ("CPU", "NPU"))
    assert zoo.resolve_device() == "CPU"      # gpu preset never picks the NPU


def test_resolve_device_env_pin_and_explicit_argument(monkeypatch):
    monkeypatch.setattr(zoo.accel, "preset", lambda: "auto")
    monkeypatch.setattr(zoo.accel, "available_devices", lambda: ("CPU", "NPU"))
    monkeypatch.setenv(zoo.ENV_DEVICE, "gpu")
    assert zoo.resolve_device() == "CPU"        # pinned device absent: CPU, not NPU
    assert zoo.resolve_device("npu") == "NPU"   # explicit beats the env pin
    monkeypatch.setenv(zoo.ENV_DEVICE, "off")
    assert zoo.resolve_device() == "CPU"


def test_compile_fallback_names_the_device_that_refused(monkeypatch):
    monkeypatch.delenv(zoo.ENV_DEVICE, raising=False)
    monkeypatch.setattr(zoo.accel, "preset", lambda: "auto")
    monkeypatch.setattr(zoo.accel, "available_devices", lambda: ("CPU", "NPU"))
    tried = []

    def build(device):
        tried.append(device)
        if device == "NPU":
            raise RuntimeError("Unsupported operation\nsecond line")
        return f"net@{device}"

    net, name = zoo._compile_with_fallback(build, None, "thing")
    assert tried == ["NPU", "CPU"]
    assert net == "net@CPU"
    assert name.startswith("CPU (NPU refused: Unsupported operation")
    assert "second line" not in name


def test_compile_fallback_raises_when_cpu_refuses_too(monkeypatch):
    monkeypatch.delenv(zoo.ENV_DEVICE, raising=False)
    monkeypatch.setattr(zoo.accel, "preset", lambda: "cpu")
    monkeypatch.setattr(zoo.accel, "available_devices", lambda: ("CPU",))

    def build(device):
        raise RuntimeError("boom")

    with pytest.raises(zoo.OpenVINOZooUnavailable, match="thing"):
        zoo._compile_with_fallback(build, None, "thing")


# ── weights ──────────────────────────────────────────────────────────────────

def test_model_path_rejects_unknown_names():
    with pytest.raises(KeyError, match="nope"):
        zoo.model_path("nope")


def test_registry_digests_are_sha256_hex():
    for name, (filename, url, digest) in zoo._REGISTRY.items():
        assert len(digest) == 64 and int(digest, 16) >= 0, name
        assert url.startswith("https://"), name
        assert filename and "/" not in filename, name


def test_download_rejects_a_wrong_digest_and_leaves_nothing(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"weights")
    target = tmp_path / "cache" / "w.bin"
    with pytest.raises(zoo.OpenVINOZooUnavailable, match="SHA-256"):
        zoo._download(source.as_uri(), str(target), "0" * 64)
    assert not target.exists()
    assert not target.with_suffix(".bin.part").exists()


def test_download_keeps_a_file_with_the_right_digest(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"weights")
    target = tmp_path / "cache" / "w.bin"
    zoo._download(source.as_uri(), str(target), hashlib.sha256(b"weights").hexdigest())
    assert target.read_bytes() == b"weights"
