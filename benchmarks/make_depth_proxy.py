"""
make_depth_proxy.py — a FastDepth-shaped ONNX with random weights.

`bench_monocular_depth.py` can measure everything around the session run, but
not the run itself: no FastDepth weights exist on this machine and an unpinned
third-party ONNX is not something to auto-download. That left the one number
that decides whether monocular depth is affordable -- inference milliseconds --
unmeasured.

Inference cost does not depend on what the weights *are*. It depends on the
graph: layer shapes, kernel sizes, channel counts, and how many multiply-adds
each one implies. So this builds FastDepth's architecture (Wofk et al., ICRA
2019) with random weights and writes it out as ONNX. Run the bench against it
and the timing is the real timing of the real network on this CPU.

What the proxy does NOT tell you:

  * **accuracy** -- random weights predict noise. Every depth value out of this
    file is meaningless, which is why it writes to a temp path by default and
    carries "proxy" in its name. It must never be mistaken for a usable model.
  * **quantised or graph-optimised exports** -- a real export may fold
    batch-norm into its convolutions or ship int8. This is fp32 with the norms
    kept separate, so it is an upper bound on cost, not a floor.

Run::

    python benchmarks/make_depth_proxy.py                 # writes to a temp dir
    python benchmarks/make_depth_proxy.py -o proxy.onnx
    python benchmarks/make_depth_proxy.py --bench         # build, then time it

Architecture, from the paper's table and the reference implementation:

  encoder  MobileNet-V1 -- a 3x3 stride-2 conv 3->32, then 13 depthwise-
           separable blocks widening 32->1024 and downsampling 224 -> 7.
  decoder  NNConv5 -- five (5x5 depthwise-separable conv, nearest-neighbour 2x
           upsample) stages narrowing 1024->32 and upsampling 7 -> 224, then a
           1x1 pointwise conv down to a single depth channel.

The reference decoder also adds encoder features into the decoder at matching
resolutions. Those are elementwise adds on tensors the convolutions have
already touched; they move the total by well under a percent and are left out
rather than guessed at, so this slightly under-states a skip-connected export.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import bold, cyan, dim, green, header, red  # noqa: E402

try:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
except ImportError:                                       # pragma: no cover
    onnx = None

#: FastDepth's native input, matching `MonocularDepthSource.DEFAULT_INPUT`.
INPUT_SIZE = (224, 224)

#: MobileNet-V1 body: (output channels, stride) per depthwise-separable block,
#: after the initial 3x3 stride-2 convolution takes 3 -> 32.
MOBILENET_BLOCKS = (
    (64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2),
    (512, 1), (512, 1), (512, 1), (512, 1), (512, 1),
    (1024, 2), (1024, 1),
)

#: NNConv5 decoder widths, each stage followed by a 2x nearest upsample.
DECODER_CHANNELS = (512, 256, 128, 64, 32)

#: Kernel size of the decoder's depthwise convolutions. The "5" in NNConv5.
DECODER_KERNEL = 5

#: Published FastDepth is about this big. A proxy that lands far from it is a
#: wrong architecture rather than a slow machine, so the report says so.
REFERENCE_PARAMS_M = 3.9


class _GraphBuilder:
    """Accumulates nodes and initializers; every weight is random."""

    def __init__(self, seed: int = 0) -> None:
        self.nodes: list = []
        self.inits: list = []
        self.rng = np.random.default_rng(seed)
        self._n = 0

    def _init(self, name: str, array: np.ndarray) -> str:
        self.inits.append(numpy_helper.from_array(
            np.asarray(array, dtype=np.float32), name))
        return name

    def _weights(self, name: str, shape: tuple[int, ...]) -> str:
        # He-ish scaling. Irrelevant to timing, but it keeps activations finite,
        # so a NaN out of the bench means a real bug and not just weights that
        # blew up over sixty layers.
        fan_in = int(np.prod(shape[1:]))
        scale = float(np.sqrt(2.0 / max(1, fan_in)))
        return self._init(name, self.rng.normal(0.0, scale, shape))

    def conv(self, x: str, tag: str, in_c: int, out_c: int, kernel: int,
             stride: int, groups: int = 1) -> str:
        """Conv -> BatchNorm -> ReLU6, the MobileNet unit."""
        self._n += 1
        stem = f"{tag}_{self._n}"
        weight = self._weights(f"{stem}_w", (out_c, in_c // groups, kernel, kernel))
        pad = kernel // 2
        conv_out = f"{stem}_conv"
        self.nodes.append(helper.make_node(
            "Conv", [x, weight], [conv_out], name=conv_out,
            kernel_shape=[kernel, kernel], strides=[stride, stride],
            pads=[pad, pad, pad, pad], group=groups))

        # Kept separate rather than folded into the convolution: an exporter
        # may or may not fuse it, and unfused is the slower of the two, so the
        # bench reports the pessimistic case.
        bn_out = f"{stem}_bn"
        self.nodes.append(helper.make_node(
            "BatchNormalization",
            [conv_out,
             self._init(f"{stem}_gamma", np.ones(out_c)),
             self._init(f"{stem}_beta", np.zeros(out_c)),
             self._init(f"{stem}_mean", np.zeros(out_c)),
             self._init(f"{stem}_var", np.ones(out_c))],
            [bn_out], name=bn_out, epsilon=1e-5))

        act = f"{stem}_relu"
        self.nodes.append(helper.make_node(
            "Clip", [bn_out,
                     self._init(f"{stem}_lo", np.array(0.0)),
                     self._init(f"{stem}_hi", np.array(6.0))],
            [act], name=act))
        return act

    def separable(self, x: str, tag: str, in_c: int, out_c: int,
                  kernel: int, stride: int) -> str:
        """Depthwise kxk then pointwise 1x1 — the whole point of MobileNet."""
        dw = self.conv(x, f"{tag}_dw", in_c, in_c, kernel, stride, groups=in_c)
        return self.conv(dw, f"{tag}_pw", in_c, out_c, 1, 1)

    def upsample2x(self, x: str, tag: str) -> str:
        self._n += 1
        out = f"{tag}_{self._n}_up"
        self.nodes.append(helper.make_node(
            "Resize",
            [x, "", self._init(f"{out}_scales", np.array([1.0, 1.0, 2.0, 2.0]))],
            [out], name=out, mode="nearest",
            coordinate_transformation_mode="asymmetric", nearest_mode="floor"))
        return out


def build_model(seed: int = 0):
    """The FastDepth graph, random weights, fixed 1x3x224x224 input."""
    height, width = INPUT_SIZE
    builder = _GraphBuilder(seed)

    x = builder.conv("input", "stem", 3, 32, 3, 2)
    channels = 32
    for index, (out_c, stride) in enumerate(MOBILENET_BLOCKS):
        x = builder.separable(x, f"enc{index}", channels, out_c, 3, stride)
        channels = out_c

    for index, out_c in enumerate(DECODER_CHANNELS):
        x = builder.separable(x, f"dec{index}", channels, out_c, DECODER_KERNEL, 1)
        x = builder.upsample2x(x, f"dec{index}")
        channels = out_c

    # Pointwise down to one depth channel. No activation: the network's output
    # is metres, and clamping at 0 here would hide a sign error downstream.
    builder._n += 1
    weight = builder._weights("head_w", (1, channels, 1, 1))
    builder.nodes.append(helper.make_node(
        "Conv", [x, weight], ["output"], name="head",
        kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0]))

    graph = helper.make_graph(
        builder.nodes, "fastdepth_proxy",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT,
                                       [1, 3, height, width])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT,
                                       [1, 1, height, width])],
        builder.inits)
    model = helper.make_model(
        graph, producer_name="visual_ai.make_depth_proxy",
        opset_imports=[helper.make_opsetid("", 13)])
    model.doc_string = ("FastDepth architecture with RANDOM weights. For timing "
                        "only -- its depth values are meaningless.")
    onnx.checker.check_model(model)

    # The declared output above is an assertion, not a fact: nothing stops a
    # graph from claiming 224x224 and computing 112x112, and onnxruntime would
    # happily time the smaller one. Ask the shape inferencer what the graph
    # actually produces and refuse to hand back a proxy that disagrees.
    computed = inferred_output_shape(model)
    if computed != [1, 1, height, width]:
        raise ValueError(
            f"graph computes {computed}, not the declared [1, 1, {height}, "
            f"{width}] -- encoder strides and decoder upsamples disagree")
    return model


def inferred_output_shape(model) -> list[int]:
    """What the graph really produces, per ONNX shape inference."""
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    except onnx.shape_inference.InferenceError as exc:
        # Strict mode raises on a declared/computed mismatch instead of
        # reporting one, so turn it into the same failure the caller checks
        # for rather than letting an opaque C++ error escape.
        raise ValueError(f"graph shapes do not add up: {exc}") from exc
    tensor = inferred.graph.output[0].type.tensor_type
    return [d.dim_value for d in tensor.shape.dim]


def parameter_count(model) -> int:
    return sum(int(np.prod(init.dims)) for init in model.graph.initializer)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--out", default="",
                        help="where to write the ONNX; default is a temp file")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bench", action="store_true",
                        help="run bench_monocular_depth against the result")
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args(argv)

    print(bold(cyan("FastDepth architecture proxy — random weights")))
    if onnx is None:
        print(red("  onnx is not installed, so the proxy cannot be built "
                  "(pip install onnx)."))
        return 1

    model = build_model(args.seed)
    out = Path(args.out) if args.out else (
        Path(tempfile.gettempdir()) / "fastdepth_proxy.onnx")
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out))

    params_m = parameter_count(model) / 1e6
    print(header("Graph"))
    print(f"  {'nodes':<14}{len(model.graph.node)}")
    print(f"  {'parameters':<14}{params_m:.2f} M"
          f"  (published FastDepth ~{REFERENCE_PARAMS_M} M)")
    print(f"  {'file':<14}{out}  ({out.stat().st_size / 1e6:.1f} MB)")
    if not 0.5 * REFERENCE_PARAMS_M <= params_m <= 2.0 * REFERENCE_PARAMS_M:
        print(red("  parameter count is nowhere near FastDepth's — this proxy "
                  "is the wrong architecture, and its timing means nothing"))
        return 1
    print(green("  weights are random: the timing is real, the depth is not"))

    if args.bench:
        import bench_monocular_depth as bench
        print()
        return bench.main(["--model", str(out), "--repeats", str(args.repeats)])
    print(dim(f"\n  python benchmarks/bench_monocular_depth.py --model {out}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
