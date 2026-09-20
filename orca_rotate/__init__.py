"""vLLM plugin: restore the Hadamard rotation our GPTQ integers were quantized in.

Our recipe quantizes each weight in a rotated basis, W_r = W R^T with
R = (1/sqrt(1024)) H_1024 S, so a stored 4-bit block only means anything against R x --
never against x. vLLM has no concept of that, so this plugin hooks every AutoGPTQ linear
layer and rotates its input with the Dao-AILab CUDA kernel before the Machete/Marlin
matmul.

Three constraints shaped the design:
  * vLLM runs the engine and the model in their own processes, which import plugins via
    the `vllm.general_plugins` entry point; a monkeypatch in the API-server parent never
    reaches them.
  * vLLM compiles the forward with torch.compile. Anything non-tensor in apply() (a file
    read, a lock, an environment lookup) breaks the trace, so the sign vector is resolved
    once in process_weights_after_loading and parked on the layer, and apply() is pure.
  * The kernel is a raw pybind call, invisible to Dynamo, so it is wrapped as a
    torch.library custom op with a fake impl; the graph sees one opaque, cudagraph-safe op.

Activated only when ORCA_ROTATE_SIGNS points at a hadamard_signs.pt, so the same venv
serves unrotated models untouched.
"""
import os

import torch

BLOCK = 1024
_signs = None

# vLLM fuses these projections; every module feeding one shares a single input activation,
# and the pack assigns one sign vector per input activation (verified: gate/up, q/k/v and
# in_proj_qkv/in_proj_z are bit-identical), so any constituent's signs is the right one.
FUSED = {
    "qkv_proj": "q_proj",
    "gate_up_proj": "gate_proj",
    "in_proj_qkvz": "in_proj_qkv",
}


def _norm(name):
    """Drop whatever sits before 'layers.' so model.layers.3.x, language_model.model.layers.3.x
    (the VL wrapper's prefix) and model.language_model.layers.3.x all key the same entry."""
    i = name.find("layers.")
    return name[i:] if i >= 0 else name


def _table():
    global _signs
    if _signs is None:
        raw = torch.load(os.environ["ORCA_ROTATE_SIGNS"], map_location="cpu")
        _signs = {_norm(k): v for k, v in raw.items()}
    return _signs


def _resolve(prefix):
    """vLLM layer prefix -> its sign vector, or None if that layer was never rotated."""
    head, _, tail = _norm(prefix).rpartition(".")
    return _table().get(f"{head}.{FUSED.get(tail, tail)}")


# ---- the rotation as a custom op ----------------------------------------------------
if not hasattr(torch.ops, "orca") or not hasattr(torch.ops.orca, "rotate"):

    @torch.library.custom_op("orca::rotate", mutates_args=())
    def _rotate_op(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
        import fast_hadamard_transform_cuda as fht

        shape = x.shape
        y = (x * signs.to(x.dtype)).reshape(-1, BLOCK).contiguous()
        y = fht.fast_hadamard_transform(y, BLOCK ** -0.5)
        return y.reshape(shape)

    @_rotate_op.register_fake
    def _rotate_fake(x, signs):
        return torch.empty_like(x)


def rotate(x, signs):
    """R x for x [..., in_features]; in_features is a multiple of BLOCK by construction."""
    return torch.ops.orca.rotate(x, signs)


def register():
    if not os.environ.get("ORCA_ROTATE_SIGNS"):
        return

    from vllm.logger import init_logger
    from vllm.model_executor.layers.quantization.auto_gptq import (
        AutoGPTQConfig,
        AutoGPTQLinearMethod,
    )

    logger = init_logger("vllm.orca_rotate")  # under the vllm hierarchy or nothing is emitted
    if getattr(AutoGPTQLinearMethod, "_orca_patched", False):
        return

    # the quant method is built per layer, so the prefix can ride on the instance
    _orig_get = AutoGPTQConfig.get_quant_method

    def get_quant_method(self, layer, prefix):
        qm = _orig_get(self, layer, prefix)
        if isinstance(qm, AutoGPTQLinearMethod):
            qm._orca_prefix = prefix
        return qm

    # runs once per layer after weights load and before compile: the only safe place to
    # touch the filesystem, so the sign vector is parked on the layer here
    _orig_pwal = AutoGPTQLinearMethod.process_weights_after_loading
    counts = {"rotated": 0, "plain": 0}

    def process_weights_after_loading(self, layer):
        _orig_pwal(self, layer)
        prefix = getattr(self, "_orca_prefix", "")
        s = _resolve(prefix)
        if s is None:
            layer._orca_signs = None
            counts["plain"] += 1
            logger.warning("orca_rotate: no signs for quantized layer %s", prefix)
        else:
            dev = next((p.device for p in layer.parameters()), torch.device("cuda"))
            layer._orca_signs = s.to(device=dev, dtype=torch.bfloat16)
            counts["rotated"] += 1
        if counts["rotated"] + counts["plain"] in (1, 400):
            logger.info("orca_rotate: %d layers rotated, %d left plain",
                        counts["rotated"], counts["plain"])

    _orig_apply = AutoGPTQLinearMethod.apply

    def apply(self, layer, x, bias=None):
        s = getattr(layer, "_orca_signs", None)
        if s is not None:
            x = torch.ops.orca.rotate(x, s)
        return _orig_apply(self, layer, x, bias)

    AutoGPTQConfig.get_quant_method = get_quant_method
    AutoGPTQLinearMethod.process_weights_after_loading = process_weights_after_loading
    AutoGPTQLinearMethod.apply = apply
    AutoGPTQLinearMethod._orca_patched = True
    logger.info("orca_rotate armed from %s", os.environ["ORCA_ROTATE_SIGNS"])
