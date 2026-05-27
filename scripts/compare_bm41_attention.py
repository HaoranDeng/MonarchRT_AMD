import argparse
import os
import sys
from pathlib import Path
import importlib.util

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "bm41_attn", ROOT / "wan" / "modules" / "bm41_attn.py")
bm41_attn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bm41_attn)
bm41_attention = bm41_attn.bm41_attention
bm41_attn_matrix = bm41_attn.bm41_attn_matrix


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


def load_qkv(args, device):
    if args.npz:
        data = np.load(args.npz)
        q = torch.from_numpy(data["q"]).to(device)
        k = torch.from_numpy(data["k"]).to(device)
        v = torch.from_numpy(data["v"]).to(device)
    else:
        torch.manual_seed(args.seed)
        q = torch.randn(args.batch, args.heads, args.tokens, args.dim, device=device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

    if args.quick_tokens:
        q = q[:, :, :args.quick_tokens]
        k = k[:, :, :args.quick_tokens]
        v = v[:, :, :args.quick_tokens]
    return q, k, v


def dense_attn_matrix(q, k):
    logits = torch.matmul(q, k.transpose(-2, -1)) * (q.size(-1) ** -0.5)
    return F.softmax(logits, dim=-1)


def main():
    parser = argparse.ArgumentParser(description="Compare dense attention and BM41 attention.")
    parser.add_argument("--npz", type=str, default="", help="Optional npz with q/k/v arrays shaped [B, H, L, D].")
    parser.add_argument("--quick-tokens", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--f-tied", type=int, default=1)
    parser.add_argument("--h-reduce", type=int, default=1)
    parser.add_argument("--w-reduce", type=int, default=1)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-attn-matrix", action="store_true")
    args = parser.parse_args()

    if args.npz and not os.path.exists(args.npz):
        raise FileNotFoundError(args.npz)

    q, k, v = load_qkv(args, args.device)
    seq = q.size(2)
    frame_tokens = args.height * args.width
    if seq % frame_tokens or (seq // frame_tokens) % args.f_tied:
        raise ValueError(
            f"sequence length {seq} does not fit the Monarch layout "
            f"f_tied={args.f_tied}, h={args.height}, w={args.width}."
        )
    bm41_cfg = dict(
        f_tied=args.f_tied,
        h_reduce=args.h_reduce,
        w_reduce=args.w_reduce,
        h=args.height,
        w=args.width,
    )

    with torch.no_grad():
        out_dense = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False)
        out_bm41 = bm41_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            **bm41_cfg,
        ).transpose(1, 2)

        print(
            f"device={args.device} seq={seq} heads={q.size(1)} "
            f"dim={q.size(-1)} shared_monarch_blocks={bm41_cfg}"
        )
        print("\n=== output error vs dense ===")
        err(out_dense, out_bm41, "dense vs BM41")
        err_per_head(out_dense, out_bm41, "dense vs BM41")

        if args.check_attn_matrix:
            print("\n=== attention matrix error vs dense ===")
            A_dense = dense_attn_matrix(q, k)
            A_bm41 = bm41_attn_matrix(q, k, **bm41_cfg)
            err(A_dense, A_bm41, "dense attn vs BM41 attn")


if __name__ == "__main__":
    main()
