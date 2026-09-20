#!/usr/bin/env python3
"""GPTQ for Qwen3.8-27B, with optional Hadamard rotation and MSE-optimal clipping.

Second-order post-training quantization (Frantar et al. 2022): quantize each weight
matrix column by column, and after each column push its rounding error onto the
not-yet-quantized columns through the inverse Hessian of that layer's inputs, so the
layer's OUTPUT error is minimised rather than its weight error.

  H      = sum over calibration tokens of x x^T       (per linear layer)
  Hinv   = chol(chol_inverse(H + damp*I))^T
  for each column j:  q_j = round(w_j);  w_{>j} -= (w_j - q_j)/Hinv_jj * Hinv_{j,>j}

Sequential: block i's Hessians are collected on the hidden stream produced by blocks
0..i-1 AFTER they were quantized, so error compounds the way it will at inference.

With --rot the quantization happens in the Hadamard-rotated basis (weights folded with
Bonsai's own sign vectors, inputs rotated the same way, unfolded at the end), so the dense
checkpoint that comes out needs no runtime transform.

Evaluated with the same KLD / top-1 / PPL as 24_kld.py, against the bf16 teacher.
"""
import argparse, json, math, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bonsai_quant_torch as bqt
from bonsai_names import pack_to_module
from st_read import SafeTensors

PACK, BASE, DEV = "/data/models/Ternary-Bonsai-2-27B-mlx-2bit", "/data/models/Qwen3.8-27B", "cuda"
TARGETS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def sub(root, path):
    m = root
    for p in path.split("."):
        if not hasattr(m, p):
            return None
        m = getattr(m, p)
    return m


# ---------------------------------------------------------------- quantizer
def group_params(Wg, bits, clip):
    """Affine (scale, zero) for a [out, group] slab; optional MSE-optimal range shrink."""
    lo, hi = Wg.amin(1, keepdim=True), Wg.amax(1, keepdim=True)
    qmax = 2 ** bits - 1
    ratios = [1.0, 0.97, 0.94, 0.91, 0.88, 0.85, 0.80] if clip else [1.0]
    mid = (lo + hi) / 2
    best = None
    for c in ratios:
        l2, h2 = mid - c * (mid - lo), mid + c * (hi - mid)
        scale = ((h2 - l2) / qmax).clamp(min=1e-8).half().float()
        zero = l2.half().float()
        q = torch.clamp(torch.round((Wg - zero) / scale), 0, qmax) * scale + zero
        err = (q - Wg).pow(2).sum(1, keepdim=True)
        if best is None:
            best = (err, scale, zero)
        else:
            take = err < best[0]
            best = (torch.where(take, err, best[0]), torch.where(take, scale, best[1]),
                    torch.where(take, zero, best[2]))
    return best[1], best[2]


def quant_col(w, scale, zero, bits):
    qmax = 2 ** bits - 1
    return torch.clamp(torch.round((w - zero) / scale), 0, qmax) * scale + zero


@torch.no_grad()
def gptq(W, H, bits, group, clip, damp=0.01, blocksize=128):
    """W [out, in] float32, H [in, in] float32 -> quantized W (dequantized values)."""
    W = W.clone().float()
    out, cols = W.shape
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += torch.eye(cols, device=W.device) * damp * torch.diag(H).mean()
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    Q = torch.zeros_like(W)
    scale = zero = None
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            if col % group == 0:
                # group parameters from the CURRENT (error-compensated) weights
                scale, zero = group_params(W[:, col:col + group], bits, clip)
            w = W1[:, i]
            d = Hinv1[i, i]
            q = quant_col(w, scale[:, 0], zero[:, 0], bits)
            Q1[:, i] = q
            err = (w - q) / d
            W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
        # keep W[:, i1:i2] consistent for later group_params calls that overlap this block
        W[:, i1:i2] = W1
    return Q


# ---------------------------------------------------------------- driver
@torch.no_grad()
def run_block(blk, h, rot, micro):
    outs = []
    for i in range(0, h.shape[0], micro):
        o = blk(h[i:i + micro], position_embeddings=(rot[0][i:i + micro], rot[1][i:i + micro]))
        outs.append(o[0] if isinstance(o, tuple) else o)
    return torch.cat(outs, 0)


@torch.no_grad()
def logprobs(model, ids, seqlen, windows):
    out = []
    for k in range(windows):
        c = ids[k * seqlen:(k + 1) * seqlen].unsqueeze(0).to(DEV)
        lg = model(input_ids=c, use_cache=False).logits[0, :-1].float()
        out.append(torch.log_softmax(lg, -1).half().cpu())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--rot", action="store_true")
    ap.add_argument("--clip", action="store_true")
    ap.add_argument("--head-bits", type=int, default=None, help="bits for embed/lm_head (default: same)")
    ap.add_argument("--calib-seqs", type=int, default=32)
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--eval-windows", type=int, default=8)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default="out/kld.json")
    a = ap.parse_args()
    head_bits = a.head_bits or a.bits
    tag = a.tag or f"gptq_int{a.bits}_g{a.group}" + ("_rot" if a.rot else "") + ("_clip" if a.clip else "") \
        + (f"_head{head_bits}" if head_bits != a.bits else "")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(BASE)
    ids = torch.tensor(tok(open("data/wikitext2_test.txt").read(),
                           return_tensors="np").input_ids[0], dtype=torch.long)
    calib = ids[-a.calib_seqs * a.seqlen:].reshape(a.calib_seqs, a.seqlen)
    ev_targets = torch.stack([ids[k * 2048 + 1:(k + 1) * 2048] for k in range(a.eval_windows)])
    st = SafeTensors(os.path.join(PACK, "model.safetensors"))

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map=DEV)
    model.eval()
    teacher = logprobs(model, ids, 2048, a.eval_windows)
    print(f"[{tag}] teacher cached {time.time()-t0:.0f}s", flush=True)

    layers = model.model.layers
    with torch.no_grad():
        h = model.model.embed_tokens(calib.to(DEV))
    pos = torch.arange(a.seqlen, device=DEV).unsqueeze(0).expand(calib.shape[0], -1)
    rope = model.model.rotary_emb(h, pos)

    def signs_for(li, t):
        return torch.from_numpy(st.get(f"language_model.model.layers.{li}.{t}.signs", np.float32).copy()).to(DEV)

    t0 = time.time()
    for li, blk in enumerate(layers):
        lins = {t: sub(blk, t) for t in TARGETS if isinstance(sub(blk, t), torch.nn.Linear)}
        H = {t: torch.zeros(l.in_features, l.in_features, device=DEV) for t, l in lins.items()}
        cnt = {t: 0 for t in lins}
        sg = {t: signs_for(li, t) for t in lins} if a.rot else {}

        def mk_hook(t):
            def hook(mod, inp, out):
                x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                if a.rot:
                    x = bqt.fwht(x, 1024, sg[t])
                H[t] += x.T @ x
                cnt[t] += x.shape[0]
            return hook
        hooks = [lins[t].register_forward_hook(mk_hook(t)) for t in lins]
        run_block(blk, h, rope, a.micro)              # collect Hessians on the quantized stream
        for hk in hooks:
            hk.remove()

        for t, lin in lins.items():
            Hn = H[t] / max(cnt[t], 1)
            W = lin.weight.data.float()
            if a.rot:
                W = bqt.fwht(W, 1024, sg[t])
            Q = gptq(W, Hn, a.bits, a.group, a.clip, a.damp)
            if a.rot:
                Q = bqt.fwht(Q, 1024, sg[t], inverse=True)
            lin.weight.data.copy_(Q.to(lin.weight.dtype))
        del H
        h = run_block(blk, h, rope, a.micro)          # propagate through the quantized block
        torch.cuda.empty_cache()
        if li % 8 == 7:
            print(f"  L{li} done  {time.time()-t0:.0f}s", flush=True)

    # embeddings + head: plain RTN (embed is a lookup; head Hessian would need the norm output)
    for pk_key in ["language_model.model.embed_tokens", "language_model.lm_head"]:
        p = dict(model.named_parameters())[pack_to_module(pk_key)]
        W = p.data.float()
        if a.rot:
            s_ = torch.from_numpy(st.get(pk_key + ".signs", np.float32).copy()).to(DEV)
            W = bqt.fwht(W, 1024, s_)
        rows, cols = W.shape
        G = W.reshape(rows, -1, a.group)
        Wq = torch.empty_like(W)
        for gi in range(G.shape[1]):
            sc, ze = group_params(G[:, gi], head_bits, a.clip)
            Wq[:, gi * a.group:(gi + 1) * a.group] = quant_col(G[:, gi], sc, ze, head_bits)
        if a.rot:
            Wq = bqt.fwht(Wq, 1024, s_, inverse=True)
        p.data.copy_(Wq.to(p.dtype))
    torch.cuda.empty_cache()
    print(f"[{tag}] quantized in {time.time()-t0:.0f}s", flush=True)

    student = logprobs(model, ids, 2048, a.eval_windows)
    kl = top1 = nll = n = 0.0
    for k, (lt, ls) in enumerate(zip(teacher, student)):
        lt32, ls32 = lt.float(), ls.float()
        kl += (lt32.exp() * (lt32 - ls32)).sum(-1).sum().item()
        top1 += (lt32.argmax(-1) == ls32.argmax(-1)).float().sum().item()
        nll += -ls32.gather(1, ev_targets[k].unsqueeze(1)).sum().item()
        n += lt32.shape[0]
    kl, top1, ppl = kl / n, top1 / n, float(np.exp(nll / n))
    body = 24.353e9; head = 2.54e9
    bpw = (body * (a.bits + 32 / a.group) + head * (head_bits + 32 / a.group)) / (body + head)
    print(f"{tag:<28} KLD={kl:.5f}  top1={top1:.4f}  PPL={ppl:.4f}  bpw={bpw:.3f}", flush=True)
    rec = json.load(open(a.out)) if os.path.exists(a.out) else {}
    rec[tag] = dict(kld=kl, top1=top1, ppl=ppl, bpw=bpw, tokens=n, method="gptq",
                    bits=a.bits, group=a.group, rot=a.rot, clip=a.clip, head_bits=head_bits,
                    calib_tokens=a.calib_seqs * a.seqlen)
    json.dump(rec, open(a.out, "w"), indent=1)


main()
