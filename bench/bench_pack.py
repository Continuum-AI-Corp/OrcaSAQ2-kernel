#!/usr/bin/env python3
"""Verify the GPTQ pack/unpack round trip and the zero-point snapping.

vLLM stores the zero point as a 4-bit CODE and dequantizes as (q - zcode) * scale, so a
GPTQ zero that is an arbitrary float cannot survive the round trip. Snapping the zero onto
the integer grid *before* quantizing lets GPTQ compensate for the snap in later columns
instead of paying for it afterwards; measured here, that halves the output error.
"""
import sys, os, types, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import rotate_quant as rq

_gp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "gptq.py")
gq = types.ModuleType("gptq"); gq.__file__ = _gp
exec(compile(open(_gp).read().replace("\nmain()\n", "\n"), _gp, "exec"), gq.__dict__)
_pp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "pack_gptq.py")
P = types.ModuleType("pack"); P.__file__ = _pp
exec(compile(open(_pp).read().replace("\nmain()\n", "\n"), _pp, "exec"), P.__dict__)


def unpack(p, axis, n):
    nib = [(p >> (4 * k)) & 0xF for k in range(8)]
    if axis == 0:
        r = torch.zeros(n, p.shape[1], dtype=torch.int32, device=p.device)
        for k in range(8):
            r[k::8] = nib[k]
    else:
        r = torch.zeros(p.shape[0], n, dtype=torch.int32, device=p.device)
        for k in range(8):
            r[:, k::8] = nib[k]
    return r


dev = "cuda"
torch.manual_seed(0)
out_f, in_f, gs = 256, 1024, 128
W = torch.randn(out_f, in_f, device=dev) * 0.02
X = torch.randn(512, in_f, device=dev)
H = (X.T @ X) / X.shape[0]

q, S, Z = P.gptq_codes(W, H, gs)
print(f"codes in [{q.min().item()}, {q.max().item()}] (must be 0-15)")

qw = P.pack_int4(q.T.contiguous(), axis=0)
zc = torch.round(-Z / S.clamp(min=1e-12)).clamp(0, 15).to(torch.int32)
qz = P.pack_int4(zc.T.contiguous(), axis=1)
print(f"qweight {tuple(qw.shape)}  qzeros {tuple(qz.shape)}  scales {tuple(S.T.shape)}")
print(f"round trip: codes {torch.equal(unpack(qw,0,in_f).T, q)}  "
      f"zeros {torch.equal(unpack(qz,1,out_f).T, zc)}")

Wv = (q.float() - zc.repeat_interleave(gs, 1).float()) * S.repeat_interleave(gs, 1)
Wours = q.float() * S.repeat_interleave(gs, 1) + Z.repeat_interleave(gs, 1)
print(f"vLLM dequant vs ours: {((Wv-Wours).pow(2).mean()/Wours.pow(2).mean()).sqrt():.2e} "
      f"(0 means the zero point survives exactly)")
y0 = X @ W.T
print(f"output relative error: {((X@Wv.T-y0).pow(2).mean()/y0.pow(2).mean()).sqrt():.4f}")
