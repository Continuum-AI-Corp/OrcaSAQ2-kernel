#!/usr/bin/env python3
"""Pack our rotated GPTQ result into the on-disk GPTQ format vLLM loads.

Our recipe quantizes in the Hadamard-rotated basis, so the stored integers are only valid
for a rotated activation. vLLM has no notion of that, so the runtime rotation is supplied
separately by a patch (scripts/54_vllm_rotate.py) that applies the Dao-AILab CUDA kernel
to each quantized layer's input. This script only produces the weights.

Layout (matching vLLM's AutoGPTQLinearMethod.create_weights):
  qweight  int32 [in/8,  out]     4-bit codes, 8 per int32, packed along the INPUT axis
  qzeros   int32 [in/gs, out/8]   zero points, same packing along the OUTPUT axis
  scales   fp16  [in/gs, out]
  g_idx    int32 [in]             group index per input channel (desc_act=False -> i//gs)
"""
import argparse, gc, json, os, shutil, sys, time, types
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bonsai_quant_torch as bqt
from st_read import SafeTensors

PACK, DEV = "/data/models/Ternary-Bonsai-2-27B-mlx-2bit", "cuda"
_gp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "25_gptq.py")
gq = types.ModuleType("gptq_mod"); gq.__file__ = _gp
exec(compile(open(_gp).read().replace("\nmain()\n", "\n"), _gp, "exec"), gq.__dict__)

BITS, PACK_FACTOR = 4, 8


def pack_int4(codes_i32, axis):
    """[.., n, ..] int32 in [0,15] -> pack 8 values per int32 along `axis`."""
    codes_i32 = codes_i32.to(torch.int32)
    if axis == 0:
        n, m = codes_i32.shape
        out = torch.zeros(n // PACK_FACTOR, m, dtype=torch.int32, device=codes_i32.device)
        for k in range(PACK_FACTOR):
            out |= (codes_i32[k::PACK_FACTOR] & 0xF) << (4 * k)
    else:
        n, m = codes_i32.shape
        out = torch.zeros(n, m // PACK_FACTOR, dtype=torch.int32, device=codes_i32.device)
        for k in range(PACK_FACTOR):
            out |= (codes_i32[:, k::PACK_FACTOR] & 0xF) << (4 * k)
    return out


@torch.no_grad()
def gptq_codes(W, H, group, clip=True, damp=0.01, blocksize=128):
    """GPTQ, returning integer codes plus the per-group scale and zero it used.
    W is [out, in] in the rotated basis; codes come back [out, in] in [0,15]."""
    W = W.clone().float(); out, cols = W.shape
    H = H.clone(); dead = torch.diag(H) == 0
    H[dead, dead] = 1; W[:, dead] = 0
    H += torch.eye(cols, device=W.device) * damp * torch.diag(H).mean()
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    qmax = 2 ** BITS - 1; ng = cols // group
    q = torch.zeros(out, cols, dtype=torch.int32, device=W.device)
    S = torch.zeros(out, ng, device=W.device); Z = torch.zeros(out, ng, device=W.device)
    scale = zero = None
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        W1 = W[:, i1:i2].clone(); Err1 = torch.zeros_like(W1); Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            if col % group == 0:
                scale, zero = gq.group_params(W[:, col:col + group], BITS, clip)
                # vLLM stores the zero point as a 4-bit CODE and dequantizes as
                # (q - zcode) * scale, so a fractional zero cannot survive the round trip.
                # Snap it onto the integer grid here and keep quantizing against the snapped
                # value, so GPTQ compensates for the snap instead of paying for it later.
                zcode = torch.clamp(torch.round(-zero / scale.clamp(min=1e-12)), 0, qmax)
                zero = -zcode * scale
                S[:, col // group] = scale[:, 0]; Z[:, col // group] = zero[:, 0]
            w = W1[:, i]; d = Hinv1[i, i]
            qi = torch.clamp(torch.round((w - zero[:, 0]) / scale[:, 0]), 0, qmax)
            q[:, col] = qi.to(torch.int32)
            err = (w - (qi * scale[:, 0] + zero[:, 0])) / d
            W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0); Err1[:, i] = err
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]; W[:, i1:i2] = W1
    return q, S, Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--calib-seqs", type=int, default=256)
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--corpus", default="/data/models/wikitext2_test.txt")
    ap.add_argument("--micro", type=int, default=4)
    a = ap.parse_args()

    # a crashed predecessor leaves VLLM::EngineCore (or our own worker) holding the whole
    # card, and the next load then OOMs on a GPU that nvidia-smi says is busy but ps does not
    import subprocess
    for pid in subprocess.run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],
                              capture_output=True, text=True).stdout.split():
        if pid.strip().isdigit() and int(pid) != os.getpid():
            subprocess.run(["kill","-9",pid.strip()], capture_output=True)
            print(f"cleared stale GPU pid {pid.strip()}", flush=True)
    time.sleep(6)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(a.base)
    ids = tok(open(a.corpus).read(), return_tensors="pt")["input_ids"][0]
    calib = ids[-a.calib_seqs * a.seqlen:].reshape(a.calib_seqs, a.seqlen)
    print(f"calibration {a.calib_seqs}x{a.seqlen} = {a.calib_seqs*a.seqlen:,} tokens", flush=True)

    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map={"": 0})
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    st = SafeTensors(os.path.join(PACK, "model.safetensors"))
    signs_for = lambda li, t: torch.from_numpy(
        st.get(f"language_model.model.layers.{li}.{t}.signs", np.float32).copy()).to(DEV)

    layers = model.model.layers
    with torch.no_grad():
        h = model.model.embed_tokens(calib.to(DEV))
        # this model's rotary takes 3D position ids (mrope: 3 x batch x positions)
        pos = torch.arange(a.seqlen, device=DEV).view(1, 1, -1).expand(3, a.calib_seqs, -1)
        rope = model.model.rotary_emb(h, pos)

    packed, signs_out = {}, {}
    t0 = time.time()
    with torch.no_grad():
        for li, blk in enumerate(layers):
            lins = {t: gq.sub(blk, t) for t in gq.TARGETS if isinstance(gq.sub(blk, t), torch.nn.Linear)}
            sg = {t: signs_for(li, t) for t in lins}
            H = {t: torch.zeros(l.in_features, l.in_features, device=DEV) for t, l in lins.items()}
            cnt = {t: 0 for t in lins}

            def mk(t):
                def hook(m, inp, out):
                    x = bqt.fwht(inp[0].detach().reshape(-1, inp[0].shape[-1]).float(), 1024, sg[t])
                    H[t] += x.T @ x; cnt[t] += x.shape[0]
                return hook
            hooks = [lins[t].register_forward_hook(mk(t)) for t in lins]
            gq.run_block(blk, h, rope, a.micro)
            for hk in hooks:
                hk.remove()

            for t, lin in lins.items():
                Wf = bqt.fwht(lin.weight.data.float(), 1024, sg[t])           # [out, in] rotated
                q, S, Z = gptq_codes(Wf, H[t] / max(cnt[t], 1), a.group)
                name = f"model.layers.{li}.{t}"
                # qweight: pack along input -> transpose to [in, out] first
                packed[name + ".qweight"] = pack_int4(q.T.contiguous(), axis=0).cpu()
                packed[name + ".scales"] = S.T.contiguous().half().cpu()       # [in/gs, out]
                # vLLM dequantizes as (q - zero_code) * scale; our zero is a float offset,
                # so the equivalent integer zero-point is round(-Z/S)
                zc = torch.round(-Z / S.clamp(min=1e-12)).clamp(0, 15).to(torch.int32)  # exact: Z was snapped
                packed[name + ".qzeros"] = pack_int4(zc.T.contiguous(), axis=1).cpu()
                packed[name + ".g_idx"] = (torch.arange(Wf.shape[1], device=DEV) // a.group).to(torch.int32).cpu()
                signs_out[name] = sg[t].half().cpu()
                # keep the model consistent so the next block sees quantized activations
                Wq = (q.float() * S.repeat_interleave(a.group, 1) + Z.repeat_interleave(a.group, 1))
                lin.weight.data.copy_(bqt.fwht(Wq, 1024, sg[t], inverse=True).to(lin.weight.dtype))
                del Wf, q, S, Z, Wq
            del H
            h = gq.run_block(blk, h, rope, a.micro)
            torch.cuda.empty_cache()
            if li % 16 == 15:
                print(f"  L{li} {time.time()-t0:.0f}s", flush=True)

    # embeddings and head stay bf16: vLLM would need lm_head quantization support and they
    # are only 9.5% of the weights
    for pk_key, name in [("language_model.model.embed_tokens", "model.embed_tokens.weight"),
                         ("language_model.lm_head", "lm_head.weight")]:
        p = dict(model.named_parameters())[name]
        s_ = torch.from_numpy(st.get(pk_key + ".signs", np.float32).copy()).to(DEV)
        W = bqt.fwht(p.data.float(), 1024, s_)
        G = W.reshape(W.shape[0], -1, a.group); Wq = torch.empty_like(W)
        for gi in range(G.shape[1]):
            sc, ze = gq.group_params(G[:, gi], 8, True)
            Wq[:, gi*a.group:(gi+1)*a.group] = gq.quant_col(G[:, gi], sc, ze, 8)
        p.data.copy_(bqt.fwht(Wq, 1024, s_, inverse=True).to(p.dtype))
    print(f"quantized in {time.time()-t0:.0f}s", flush=True)

    os.makedirs(a.out, exist_ok=True)
    sd = {k: v for k, v in model.state_dict().items()
          if not any(k.startswith(f"model.layers.{li}.{t}.weight")
                     for li in range(len(layers)) for t in gq.TARGETS)}
    sd = {k: v.cpu() for k, v in sd.items()}
    sd.update(packed)
    from safetensors.torch import save_file
    torch.save(signs_out, os.path.join(a.out, "hadamard_signs.pt"))
    # shard
    shards, cur, curbytes, idx = [], {}, 0, {}
    for k, v in sd.items():
        nb = v.numel() * v.element_size()
        if curbytes + nb > 4_500_000_000 and cur:
            shards.append(cur); cur, curbytes = {}, 0
        cur[k] = v.contiguous(); curbytes += nb
    if cur: shards.append(cur)
    for i, sh in enumerate(shards):
        fn = f"model-{i+1:05d}-of-{len(shards):05d}.safetensors"
        save_file(sh, os.path.join(a.out, fn), metadata={"format": "pt"})
        for k in sh: idx[k] = fn
        print(f"  wrote {fn} ({len(sh)} tensors)", flush=True)
    json.dump({"metadata": {"total_size": sum(v.numel()*v.element_size() for v in sd.values())},
               "weight_map": idx}, open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=1)

    cfg = json.load(open(os.path.join(a.base, "config.json")))
    cfg["quantization_config"] = {"quant_method": "gptq", "bits": BITS, "group_size": a.group,
                                  "desc_act": False, "sym": False, "lm_head": False,
                                  "checkpoint_format": "gptq",
                                  "hadamard_rotation": {"block_size": 1024,
                                                        "signs_file": "hadamard_signs.pt"}}
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    tok.save_pretrained(a.out)
    for f in ("chat_template.jinja", "generation_config.json", "preprocessor_config.json"):
        s2 = os.path.join(a.base, f)
        if os.path.exists(s2): shutil.copy(s2, a.out)
    sz = sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out)
             if f.endswith(".safetensors"))
    print(f"saved to {a.out}: {sz/1e9:.1f} GB", flush=True)


main()
