"""Generate the tiny synthetic ONNX fixture used by every S02-S06 test (M009/S02).

Matches the real CLIP encoder's exact I/O contract — ``(1, 3, 224, 224)`` float32
in, ``(1, 512)`` float32 out — at a few KB instead of megabytes, so no test in
this suite ever downloads or loads a real model. The graph is deliberately
trivial (``GlobalAveragePool -> Flatten -> Gemm`` against a small fixed-seed
weight matrix) while satisfying the declared shape contract exactly and being
fully deterministic (fixed seed, no random op at inference time).

Not executed by pytest — this is a one-off generator, run manually whenever the
fixture needs regenerating:

    uv run --with onnx python tests/fixtures/gen_tiny_embedding_model.py

``onnx`` is intentionally NOT a ``pyproject.toml`` dependency (CONTEXT locks
``onnxruntime`` as the only new project dependency this milestone) — ``--with``
installs it ephemerally for this one invocation only, never touching the
lockfile. ``onnxruntime`` (a real project dependency by this point in S02) then
verifies the exported graph actually runs and produces the declared output
shape before the ``.onnx`` bytes are written to disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

#: Written next to this script; committed to the repo (a few KB).
OUTPUT_PATH = Path(__file__).parent / "tiny_embedding_model.onnx"

#: Matches curator.taste.embedding.provider.EMBEDDING_DIM — not imported
#: directly so this generator stays runnable via a bare ``uv run --with onnx``
#: invocation with no project-source dependency.
EMBEDDING_DIM = 512


def build_model() -> onnx.ModelProto:
    """Build the tiny synthetic embedding graph and return the ONNX model.

    ``input (1,3,224,224)`` -> ``GlobalAveragePool`` -> ``(1,3,1,1)`` ->
    ``Flatten`` -> ``(1,3)`` -> ``Gemm`` against a fixed-seed ``(3, 512)``
    weight matrix (~6KB) plus a ``(512,)`` zero bias -> ``output (1,512)``.
    """
    weight = np.random.RandomState(0).uniform(-1, 1, size=(3, EMBEDDING_DIM)).astype(np.float32)
    bias = np.zeros((EMBEDDING_DIM,), dtype=np.float32)

    weight_initializer = helper.make_tensor(
        "weight", TensorProto.FLOAT, weight.shape, weight.flatten().tolist()
    )
    bias_initializer = helper.make_tensor(
        "bias", TensorProto.FLOAT, bias.shape, bias.flatten().tolist()
    )

    pool_node = helper.make_node("GlobalAveragePool", ["input"], ["pooled"])
    flatten_node = helper.make_node("Flatten", ["pooled"], ["flattened"], axis=1)
    gemm_node = helper.make_node(
        "Gemm", ["flattened", "weight", "bias"], ["output"], alpha=1.0, beta=1.0
    )

    graph = helper.make_graph(
        [pool_node, flatten_node, gemm_node],
        "tiny_embedding_model",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, EMBEDDING_DIM])],
        [weight_initializer, bias_initializer],
    )
    model = helper.make_model(
        graph,
        producer_name="curator-fixture-generator",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    model = build_model()
    # Verify once with onnxruntime before writing — confirms the exact (1, 512)
    # output shape the real encoder's contract declares, on a fully synthetic
    # graph with no external weights and no network access.
    import onnxruntime as ort

    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    sample = np.zeros((1, 3, 224, 224), dtype=np.float32)
    (output,) = session.run(None, {"input": sample})
    if output.shape != (1, EMBEDDING_DIM):
        raise AssertionError(f"unexpected output shape: {output.shape}")

    OUTPUT_PATH.write_bytes(model.SerializeToString())
    print(f"wrote {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
