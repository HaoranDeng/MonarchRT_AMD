import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
WAN_H, WAN_W = 30, 52
FRAME_TOKENS = WAN_H * WAN_W
MONARCH_CFG = dict(f_tied=1, h_reduce=1, w_reduce=1, h=WAN_H, w=WAN_W)
MONARCH_ITERS = (1,)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bm41 = load_module("bm41_attn", ROOT / "wan/modules/bm41_attn.py")
mrt = load_module("monarch_attn", ROOT / "wan/modules/monarch_attn.py")


def _diff(ref, other):
    ref, other = ref.float(), other.float()
    d = (ref - other).abs()
    r = ref.abs().clamp(min=1e-8)
    return d, r


def err(ref, other, tag, *, per_head=False):
    d, r = _diff(ref, other)
    if not per_head:
        mae = d.mean().item()
        print(
            f"{tag}: mae={mae:.4f} rel_mae={mae / r.mean().item() * 100:.4f}% "
            f"rel_l1={d.sum().item() / r.sum().item() * 100:.4f}% max={d.max().item():.4f}"
        )
        return
    mae_h = d.mean(dim=(0, 2, 3))
    rel_mae_h = mae_h / r.mean(dim=(0, 2, 3)).clamp(min=1e-8) * 100
    rel_l1_h = d.sum(dim=(0, 2, 3)) / r.sum(dim=(0, 2, 3)).clamp(min=1e-8) * 100
    max_h = d.amax(dim=(0, 2, 3))
    print(f"\n{tag} (per head):")
    print(f"{'head':>4}  {'mae':>8}  {'rel_mae%':>10}  {'rel_l1%':>10}  {'max':>8}")
    for h in range(ref.size(1)):
        print(
            f"{h:4d}  {mae_h[h].item():8.4f}  {rel_mae_h[h].item():10.4f}  "
            f"{rel_l1_h[h].item():10.4f}  {max_h[h].item():8.4f}"
        )
    worst = int(mae_h.argmax())
    print(f"  worst head: h={worst} mae={mae_h[worst].item():.4f}")


def attn_matrix(q, k):
    logits = torch.matmul(q, k.transpose(-2, -1)) * (q.size(-1) ** -0.5)
    return F.softmax(logits, dim=-1)


def from_bshd(x, seq):
    return x.transpose(1, 2)[:, :, :seq]


def run_bshd(fn, q, k, v, seq, **kwargs):
    qt, kt, vt = (t.transpose(1, 2) for t in (q, k, v))
    return from_bshd(fn(qt, kt, vt, **kwargs), seq)


def parse_csv(value, n, name):
    if not value:
        return None
    parts = [int(x) for x in value.split(",")]
    if len(parts) != n:
        raise ValueError(f"{name} expects {n} comma-separated ints, got {len(parts)}")
    return parts


def infer_frames(seq):
    if seq % FRAME_TOKENS:
        raise ValueError(f"seq {seq} not divisible by Wan grid {FRAME_TOKENS}")
    frames = seq // FRAME_TOKENS
    if frames % MONARCH_CFG["f_tied"]:
        raise ValueError(f"frame count {frames} not divisible by f_tied={MONARCH_CFG['f_tied']}")
    return frames


def print_attn_sample(A_dense, A_bm41, b, h, r0, r1, c0, c1, block_size):
    Ad = A_dense[b, h, r0:r1, c0:c1].float().cpu().numpy()
    Ab = A_bm41[b, h, r0:r1, c0:c1].float().cpu().numpy()
    Dd = np.abs(Ad - Ab)
    nr, nc = Ad.shape
    print(f"\n=== attn sample b={b} h={h} rows=[{r0}:{r1}) cols=[{c0}:{c1}) bs={block_size} ===")
    print(f"{'':>6}", " ".join(f"c{j:>3}" for j in range(c0, c0 + nc)))
    for label, mat in (("dense", Ad), ("BM41", Ab), ("|diff|", Dd)):
        for i in range(nr):
            print(f"q{r0 + i:>4} {label:>6} | {' '.join(f'{x:6.4f}' for x in mat[i])}")
        print()
    print(
        f"  submatrix mae={Dd.mean():.4f} "
        f"rel_mae={Dd.mean() / max(np.abs(Ad).mean(), 1e-8) * 100:.4f}% "
        f"max={Dd.max():.4f} row_sum(dense)={Ad.sum(1)} row_sum(BM41)={Ab.sum(1)}"
    )


def main():
    p = argparse.ArgumentParser(description="Compare dense, BM41, and Monarch attention on saved QKV.")
    p.add_argument("--npz", default="assets/first_qkv/first_attn_qkv_dense_layer0_ts999.npz")
    p.add_argument("--block-size", type=int, required=True)
    p.add_argument("--quick-tokens", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--attn-slice", default="", help="b,h or empty")
    p.add_argument("--attn-sample", default="", help="b,h,r0,r1,c0,c1")
    args = p.parse_args()

    data = np.load(args.npz)
    q = torch.from_numpy(data["q"]).to(args.device)
    k = torch.from_numpy(data["k"]).to(args.device)
    v = torch.from_numpy(data["v"]).to(args.device)
    if args.quick_tokens:
        q, k, v = (t[:, :, : args.quick_tokens] for t in (q, k, v))
    seq = q.size(2)
    grid = (infer_frames(seq), WAN_H, WAN_W)

    with torch.no_grad():
        out_dense = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        outs = {
            "BM41": run_bshd(bm41.bm41_attention, q, k, v, seq, block_size=args.block_size, grid_size=grid),
        }
        for n in MONARCH_ITERS:
            outs[f"MRT-{n}"] = run_bshd(
                mrt.monarch_attn, q, k, v, seq, num_iters=n, **MONARCH_CFG
            )

        need_attn = args.attn_slice or args.attn_sample
        if need_attn:
            A_dense = attn_matrix(q, k)
            A_bm41 = bm41.bm41_attn_matrix(q, k, block_size=args.block_size, grid_size=grid)[:, :, :seq, :seq]

    print(
        f"device={args.device} seq={seq} heads={q.size(1)} dim={q.size(-1)} "
        f"bm41_bs={args.block_size} mrt_grid={grid} monarch={MONARCH_CFG}"
    )
    if "tag" in data:
        print(f"capture_tag={data['tag']}")

    for per_head in (False, True):
        label = "per head" if per_head else "global"
        print(f"\n=== output error vs dense ({label}) ===")
        for name, out in outs.items():
            err(out_dense, out, f"dense vs {name}", per_head=per_head)

    if args.attn_slice:
        b, h = parse_csv(args.attn_slice, 2, "--attn-slice")
        sl = slice(h, h + 1)
        print(f"\n=== slice b={b} h={h} ===")
        err(out_dense[:, sl], outs["BM41"][:, sl], "dense vs BM41 output")
        err(torch.matmul(A_dense[:, sl], v[:, sl]), torch.matmul(A_bm41[:, sl], v[:, sl]), "A_dense@V vs A_bm41@V")
        err(A_dense[:, sl], A_bm41[:, sl], "dense vs BM41 attn matrix")

    sample = parse_csv(args.attn_sample, 6, "--attn-sample")
    if sample:
        print_attn_sample(A_dense, A_bm41, *sample, block_size=args.block_size)


if __name__ == "__main__":
    main()
