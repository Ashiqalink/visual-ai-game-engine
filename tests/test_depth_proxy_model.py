"""The FastDepth architecture proxy — is it actually FastDepth's shape?

A random-weight stand-in is only worth its timing if the graph it times is the
graph that would ship. Nothing about a wrong architecture looks wrong at a
glance: it loads, it runs, it prints milliseconds. So these tests pin the
things that would silently make the bench's answer meaningless -- parameter
count, input and output shape, the depthwise structure that makes MobileNet
cheap -- and the honesty guard that keeps the file from being mistaken for a
trained model.

`onnx` is a bench-only dependency, so every test here skips without it rather
than failing the suite on a machine that never runs benchmarks.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

BENCH_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

onnx = pytest.importorskip("onnx", reason="onnx is a benchmark-only dependency")

import make_depth_proxy as proxy  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return proxy.build_model(seed=0)


def test_parameter_count_matches_published_fastdepth(model):
    """~3.9 M is the published figure; far from it means a wrong graph.

    This is the test that fails if a channel width or a block count is
    mistyped. Everything else about such a model still works -- it just times
    a network nobody is proposing to ship.
    """
    millions = proxy.parameter_count(model) / 1e6
    assert 0.75 * proxy.REFERENCE_PARAMS_M <= millions <= 1.25 * proxy.REFERENCE_PARAMS_M


def test_input_matches_the_backend_default(model):
    """The bench feeds this through `MonocularDepthSource`, which reads the
    declared input shape and resizes to it. A mismatch would silently move the
    measurement to a different resolution."""
    from visual_ai.depth_source import MonocularDepthSource

    shape = [d.dim_value for d in
             model.graph.input[0].type.tensor_type.shape.dim]
    assert shape == [1, 3, *proxy.INPUT_SIZE]
    assert proxy.INPUT_SIZE == MonocularDepthSource.DEFAULT_INPUT


def test_output_is_one_full_resolution_depth_channel(model):
    """Five 2x upsamples from 7x7 must land back on the input size, and the
    head must narrow to one channel -- `_read` rejects anything else.

    Inferred, not declared. The declared shape is whatever `build_model` wrote
    down; only shape inference knows what the convolutions and upsamples add
    up to, and a decoder stage dropped from a network that still declares
    224x224 is exactly the mistake that would otherwise time a smaller graph.
    """
    assert proxy.inferred_output_shape(model) == [1, 1, *proxy.INPUT_SIZE]


def test_convolutions_are_depthwise_separable(model):
    """MobileNet's cost argument is that most convolutions are grouped.

    A proxy built from dense convolutions would be several times slower and
    would answer "unaffordable" about a network that is nothing like it.
    """
    convs = [n for n in model.graph.node if n.op_type == "Conv"]
    grouped = [n for n in convs
               if any(a.name == "group" and a.i > 1 for a in n.attribute)]
    # 13 encoder blocks + 5 decoder stages, one depthwise conv each.
    assert len(grouped) == len(proxy.MOBILENET_BLOCKS) + len(proxy.DECODER_CHANNELS)
    assert len(grouped) > len(convs) / 3


def test_decoder_uses_five_by_five_kernels(model):
    """The "5" in NNConv5. A 3x3 decoder is a cheaper network than the one
    being evaluated, so this would flatter the result."""
    sizes = [tuple(a.ints) for n in model.graph.node if n.op_type == "Conv"
             for a in n.attribute if a.name == "kernel_shape"]
    decoder = (proxy.DECODER_KERNEL, proxy.DECODER_KERNEL)
    assert sizes.count(decoder) == len(proxy.DECODER_CHANNELS)


def test_seed_is_the_only_thing_that_varies(model):
    """Same seed, same weights: a bench run has to be reproducible."""
    a = proxy.build_model(seed=7)
    b = proxy.build_model(seed=7)
    c = proxy.build_model(seed=8)
    first = lambda m: onnx.numpy_helper.to_array(m.graph.initializer[0])  # noqa: E731
    assert np.array_equal(first(a), first(b))
    assert not np.array_equal(first(a), first(c))


def test_model_says_its_weights_are_meaningless(model):
    """The one guard against this file being taken for a usable depth model."""
    assert "RANDOM" in model.doc_string
    assert "timing" in model.doc_string.lower()


def test_it_runs_and_produces_finite_depth(tmp_path, model):
    """End to end through onnxruntime, which is what the bench actually times.

    Finiteness is the real assertion: sixty layers of unscaled random weights
    overflow to inf, and an inf map would make `sanitize`'s valid fraction
    read as a model-range problem instead of a proxy that blew up.
    """
    ort = pytest.importorskip("onnxruntime")
    path = tmp_path / "proxy.onnx"
    onnx.save(model, str(path))

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    frame = np.random.default_rng(0).normal(
        0.0, 1.0, (1, 3, *proxy.INPUT_SIZE)).astype(np.float32)
    out = session.run(None, {session.get_inputs()[0].name: frame})[0]

    assert out.shape == (1, 1, *proxy.INPUT_SIZE)
    assert np.isfinite(out).all()
