import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bm41_mod = load_module("bm41_attn", ROOT / "wan" / "modules" / "bm41_attn.py")
monarch_mod = load_module("monarch_attn", ROOT / "wan" / "modules" / "monarch_attn.py")


def pad_blk(x, block_size):
    pad = (-x.size(2)) % block_size
    return F.pad(x, (0, 0, 0, pad)) if pad else x


def err(ref, other, tag):
    ref = ref.float()
    other = other.float()
    d = (ref - other).abs()
    r = ref.abs().clamp(min=1e-8)
    mae = d.mean().item()
    print(
        f"{tag}: mae={mae:.4f} rel_mae={mae / r.mean().item() * 100:.4f}% "
        f"rel_l1={d.sum().item() / r.sum().item() * 100:.4f}% max={d.max().item():.4f}"
    )


def err_per_head(ref, other, tag):
    ref = ref.float()
    other = other.float()
    d = (ref - other).abs()
    r = ref.abs().clamp(min=1e-8)
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


def dense_attn_matrix(q, k):
    logits = torch.matmul(q, k.transpose(-2, -1)) * (q.size(-1) ** -0.5)
    return F.softmax(logits, dim=-1)


def print_attn_matrix_sample(A_dense, A_bm41, b, h, r0, r1, c0, c1, block_size):
    Ad = A_dense[b, h, r0:r1, c0:c1].float().cpu().numpy()
    Ab = A_bm41[b, h, r0:r1, c0:c1].float().cpu().numpy()
    Dd = np.abs(Ad - Ab)
    nr, nc = Ad.shape
    print(
        f"\n=== attn matrix sample b={b} h={h} "
        f"rows=[{r0}:{r1}) cols=[{c0}:{c1}) block_size={block_size} ==="
    )
    print(f"{'':>6}", " ".join(f"c{j:>3}" for j in range(c0, c0 + nc)))
    for i in range(nr):
        print(f"q{r0 + i:>4} dense | {' '.join(f'{x:6.4f}' for x in Ad[i])}")
    print()
    for i in range(nr):
        print(f"q{r0 + i:>4} BM41  | {' '.join(f'{x:6.4f}' for x in Ab[i])}")
    print()
    for i in range(nr):
        print(f"q{r0 + i:>4} |diff| | {' '.join(f'{x:6.4f}' for x in Dd[i])}")
    print(
        f"  submatrix mae={Dd.mean():.4f} "
        f"rel_mae={Dd.mean() / max(np.abs(Ad).mean(), 1e-8) * 100:.4f}% "
        f"max={Dd.max():.4f} row_sum(dense)={Ad.sum(axis=1)} row_sum(BM41)={Ab.sum(axis=1)}"
    )


def parse_sample(value):
    if not value:
        return None
    parts = [int(x) for x in value.split(",")]
    if len(parts) != 6:
        raise ValueError("--attn-sample must be b,h,r0,r1,c0,c1")
    return parts


def main():
    parser = argparse.ArgumentParser(description="Compare dense, BM41, and Monarch attention on saved QKV.")
    parser.add_argument("--npz", type=str, default="assets/first_qkv/first_attn_qkv_dense_layer0_ts999.npz")
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--quick-tokens", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--monarch-iters", type=int, nargs="+", default=[1, 10])
    parser.add_argument("--attn-slice", type=str, default="", help="b,h or empty to skip")
    parser.add_argument("--attn-sample", type=str, default="", help="b,h,r0,r1,c0,c1")
    args = parser.parse_args()

    data = np.load(args.npz)
    q = torch.from_numpy(data["q"]).to(args.device)
    k = torch.from_numpy(data["k"]).to(args.device)
    v = torch.from_numpy(data["v"]).to(args.device)
    if args.quick_tokens:
        q, k, v = (t[:, :, :args.quick_tokens] for t in (q, k, v))
    seq = q.size(2)

    bm_q, bm_k, bm_v = (pad_blk(t, args.block_size) for t in (q, k, v))
    monarch_block = args.block_size * args.block_size
    m_q, m_k, m_v = (pad_blk(t, monarch_block) for t in (q, k, v))

    with torch.no_grad():
        out_dense = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        outs = {
            "BM41": bm41_mod.bm41_attention(
                bm_q.transpose(1, 2),
                bm_k.transpose(1, 2),
                bm_v.transpose(1, 2),
                block_size=args.block_size,
            ).transpose(1, 2)[:, :, :seq],
        }
        for num_iters in args.monarch_iters:
            outs[f"MRT-{num_iters}"] = monarch_mod.monarch_attn(
                m_q.transpose(1, 2),
                m_k.transpose(1, 2),
                m_v.transpose(1, 2),
                f_tied=1,
                h_reduce=1,
                w_reduce=1,
                h=args.block_size,
                w=args.block_size,
                num_iters=num_iters,
            ).transpose(1, 2)[:, :, :seq]

        need_attn = bool(args.attn_slice) or bool(args.attn_sample)
        if need_attn:
            A_dense = dense_attn_matrix(q, k)
            A_bm41 = bm41_mod.bm41_attn_matrix(
                bm_q, bm_k, block_size=args.block_size)[:, :, :seq, :seq]

    print(
        f"device={args.device} seq={seq} heads={q.size(1)} dim={q.size(-1)} "
        f"bm41_bs={args.block_size} mrt_g={args.block_size} mrt_blk={monarch_block}"
    )
    if "tag" in data:
        print(f"capture_tag={data['tag']}")

    print("\n=== output error vs dense (global) ===")
    for name, out in outs.items():
        err(out_dense, out, f"dense vs {name}")

    print("\n=== output error vs dense (per head) ===")
    for name, out in outs.items():
        err_per_head(out_dense, out, f"dense vs {name}")

    if args.attn_slice:
        b, h = [int(x) for x in args.attn_slice.split(",")]
        sl = slice(h, h + 1)
        print(f"\n=== slice b={b} h={h} ===")
        err(out_dense[:, sl], outs["BM41"][:, sl], "dense vs BM41 output")
        err(torch.matmul(A_dense[:, sl], v[:, sl]), torch.matmul(A_bm41[:, sl], v[:, sl]), "A_dense@V vs A_bm41@V")
        err(A_dense[:, sl], A_bm41[:, sl], "dense vs BM41 attn matrix")

    sample = parse_sample(args.attn_sample)
    if sample is not None:
        print_attn_matrix_sample(A_dense, A_bm41, *sample, block_size=args.block_size)


if __name__ == "__main__":
    main()
