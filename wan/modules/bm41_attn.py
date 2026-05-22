import math
import os

import torch
import torch.nn.functional as F

__all__ = [
    "bm41_attention",
    "bm41_attn_matrix",
]


def _pad_tokens(x, block_size):
    pad = (-x.size(2)) % block_size
    if pad == 0:
        return x, pad
    return F.pad(x, (0, 0, 0, pad)), pad


def _key_masks(length, padded_length, block_size, device):
    mask = torch.arange(padded_length, device=device) < length
    block_mask = mask.view(padded_length // block_size, block_size)
    return block_mask, block_mask.any(dim=-1)


def _masked_block_mean(x, token_mask):
    mask = token_mask.view(1, 1, token_mask.size(0), token_mask.size(1), 1)
    denom = mask.sum(dim=3).clamp(min=1)
    return (x * mask.to(dtype=x.dtype)).sum(dim=3) / denom.to(dtype=x.dtype)


def _grid_tile_shape(block_size, height, width):
    candidates = []
    for tile_h in range(1, int(math.sqrt(block_size)) + 1):
        if block_size % tile_h != 0:
            continue
        for h_factor, w_factor in (
            (tile_h, block_size // tile_h),
            (block_size // tile_h, tile_h),
        ):
            if height % h_factor == 0 and width % w_factor == 0:
                candidates.append((h_factor, w_factor))
    if not candidates:
        raise ValueError(
            f"block_size={block_size} cannot tile Wan grid height={height}, "
            f"width={width}.")
    return min(candidates, key=lambda item: (abs(item[0] - item[1]), -item[0]))


def _grid_blockify(x, grid_size, block_size):
    B, H, L, D = x.shape
    frames, height, width = grid_size
    expected = frames * height * width
    if L != expected:
        raise ValueError(
            f"grid_size={grid_size} implies sequence length {expected}, got {L}.")
    tile_h, tile_w = _grid_tile_shape(block_size, height, width)
    x = x.view(B, H, frames, height // tile_h, tile_h,
               width // tile_w, tile_w, D)
    x = x.permute(0, 1, 2, 3, 5, 4, 6, 7).contiguous()
    return x.view(B, H, -1, block_size, D), tile_h, tile_w


def _grid_unblockify(x, grid_size, tile_h, tile_w):
    B, H, _, _, D = x.shape
    frames, height, width = grid_size
    x = x.view(B, H, frames, height // tile_h, width // tile_w,
               tile_h, tile_w, D)
    x = x.permute(0, 1, 2, 3, 5, 4, 6, 7).contiguous()
    return x.view(B, H, frames * height * width, D)


def _grid_block_order_indices(grid_size, block_size, device):
    frames, height, width = grid_size
    tile_h, tile_w = _grid_tile_shape(block_size, height, width)
    index = torch.arange(frames * height * width, device=device)
    index = index.view(frames, height // tile_h, tile_h,
                       width // tile_w, tile_w)
    return index.permute(0, 1, 3, 2, 4).reshape(-1)


def _bm41_attention_from_blocks(
    qr,
    kr,
    vr,
    scale,
    query_block_chunk,
):
    B, H, Rq, block_size, _ = qr.shape
    Rk = kr.size(2)
    value_dim = vr.size(-1)
    out_blocks = torch.empty(
        B, H, Rq, block_size, value_dim, device=qr.device, dtype=vr.dtype)

    for start in range(0, Rq, query_block_chunk):
        end = min(start + query_block_chunk, Rq)
        q_chunk = qr[:, :, start:end]
        q_mean = q_chunk.mean(3)

        w = F.softmax(
            torch.einsum("bhqd,bhksd->bhqks", q_mean, kr).float() * scale,
            dim=-1,
        ).to(qr.dtype)

        wf = w.float().clamp(min=1e-8)
        entropy = -(wf * wf.log()).sum(-1)
        weighted_k = torch.einsum("bhqks,bhksd->bhqkd", w, kr)
        alpha_logits = (
            torch.einsum("bhqtd,bhqkd->bhqtk", q_chunk, weighted_k) * scale
            + entropy.unsqueeze(3)
        )
        alpha = F.softmax(alpha_logits.float(), dim=-1).to(qr.dtype)

        weighted_v = torch.einsum("bhqks,bhksd->bhqkd", w, vr)
        out_blocks[:, :, start:end] = torch.einsum(
            "bhqtk,bhqkd->bhqtd", alpha, weighted_v)

    return out_blocks


def _bm41_w_alpha(q, k, block_size, softmax_scale=None):
    B, H, Lq, D = q.shape
    _, _, Lk, _ = k.shape
    scale = D ** -0.5 if softmax_scale is None else softmax_scale

    q, q_pad = _pad_tokens(q, block_size)
    k, k_pad = _pad_tokens(k, block_size)
    q_len, k_len = Lq, Lk
    Lq_pad, Lk_pad = q.size(2), k.size(2)
    Rq, Rk = Lq_pad // block_size, Lk_pad // block_size

    qr = q.view(B, H, Rq, block_size, D)
    kr = k.view(B, H, Rk, block_size, D)

    q_token_mask, _ = _key_masks(q_len, Lq_pad, block_size, q.device)
    key_token_mask, key_block_mask = _key_masks(
        k_len, Lk_pad, block_size, k.device)
    w_logits = torch.einsum(
        "bhrd,bhcsd->bhrcs", _masked_block_mean(qr, q_token_mask), kr) * scale
    w_logits = w_logits.masked_fill(
        ~key_token_mask.view(1, 1, 1, Rk, block_size),
        torch.finfo(w_logits.dtype).min,
    )
    w = F.softmax(w_logits.float(), dim=-1).to(q.dtype)
    w = w.masked_fill(
        ~key_token_mask.view(1, 1, 1, Rk, block_size), 0)

    wf = w.float().clamp(min=1e-8)
    entropy = -(wf * wf.log()).sum(-1)
    entropy = entropy.masked_fill(
        ~key_block_mask.view(1, 1, 1, Rk), torch.finfo(entropy.dtype).min)

    weighted_k = torch.einsum("bhrcs,bhcsd->bhrcd", w, kr)
    alpha_logits = (
        torch.einsum("bhrtd,bhrcd->bhrtc", qr, weighted_k) * scale
        + entropy.unsqueeze(3)
    )
    alpha_logits = alpha_logits.masked_fill(
        ~key_block_mask.view(1, 1, 1, 1, Rk),
        torch.finfo(alpha_logits.dtype).min,
    )
    alpha = F.softmax(alpha_logits.float(), dim=-1).to(q.dtype)
    return qr, kr, w, alpha, q_pad, k_pad, q_len, k_len, scale


def bm41_attention(q, k, v, block_size, softmax_scale=None, query_block_chunk=None, grid_size=None):
    """
    BM41 attention for tensors shaped [B, L, H, D].

    The implementation accepts different query/key lengths for KV-cache style
    calls. Padding is masked for keys and cropped from queries before returning.
    Query blocks are streamed in chunks so small block sizes do not materialize
    the full [query_blocks, key_blocks] BM41 state.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q, k, and v must be rank-4 tensors [B, L, H, D]")
    if q.size(0) != k.size(0) or q.size(0) != v.size(0):
        raise ValueError("q, k, and v must have the same batch size")
    if q.size(2) != k.size(2) or q.size(2) != v.size(2):
        raise ValueError("q, k, and v must have the same number of heads")
    if q.size(3) != k.size(3):
        raise ValueError("q and k must have the same head dimension")
    if k.size(1) != v.size(1):
        raise ValueError("k and v must have the same sequence length")
    if query_block_chunk is None:
        query_block_chunk = int(os.environ.get("BM41_QUERY_BLOCK_CHUNK", "64"))
    query_block_chunk = max(1, int(query_block_chunk))

    q_bhld = q.transpose(1, 2).contiguous()
    k_bhld = k.transpose(1, 2).contiguous()
    v_bhld = v.transpose(1, 2).contiguous()

    B, H, Lq, D = q_bhld.shape
    _, _, Lk, _ = k_bhld.shape
    value_dim = v.size(-1)
    scale = D ** -0.5 if softmax_scale is None else softmax_scale
    if grid_size is not None:
        if Lq != Lk:
            raise ValueError("grid-aligned BM41 requires equal query/key lengths")
        qr, tile_h, tile_w = _grid_blockify(q_bhld, grid_size, block_size)
        kr, _, _ = _grid_blockify(k_bhld, grid_size, block_size)
        vr, _, _ = _grid_blockify(v_bhld, grid_size, block_size)
        out_blocks = _bm41_attention_from_blocks(
            qr, kr, vr, scale, query_block_chunk)
        out = _grid_unblockify(out_blocks, grid_size, tile_h, tile_w)
        return out.transpose(1, 2).contiguous()

    q_bhld, _ = _pad_tokens(q_bhld, block_size)
    k_bhld, k_pad = _pad_tokens(k_bhld, block_size)
    if k_pad:
        v_bhld = F.pad(v_bhld, (0, 0, 0, k_pad))
    Lq_pad, Lk_pad = q_bhld.size(2), k_bhld.size(2)
    Rq, Rk = Lq_pad // block_size, Lk_pad // block_size

    qr = q_bhld.view(B, H, Rq, block_size, D)
    kr = k_bhld.view(B, H, Rk, block_size, D)
    vr = v_bhld.view(B, H, Rk, block_size, value_dim)
    out_blocks = torch.empty(
        B, H, Rq, block_size, value_dim, device=q.device, dtype=v.dtype)

    key_token_mask, key_block_mask = _key_masks(
        Lk, Lk_pad, block_size, k.device)
    query_token_mask, _ = _key_masks(Lq, Lq_pad, block_size, q.device)
    key_token_mask = key_token_mask.view(1, 1, 1, Rk, block_size)
    key_block_mask = key_block_mask.view(1, 1, 1, Rk)

    for start in range(0, Rq, query_block_chunk):
        end = min(start + query_block_chunk, Rq)
        q_chunk = qr[:, :, start:end]
        q_mean = _masked_block_mean(q_chunk, query_token_mask[start:end])

        w_logits = torch.einsum("bhqd,bhksd->bhqks", q_mean, kr) * scale
        w_logits = w_logits.masked_fill(
            ~key_token_mask, torch.finfo(w_logits.dtype).min)
        w = F.softmax(w_logits.float(), dim=-1).to(q.dtype)
        w = w.masked_fill(~key_token_mask, 0)

        wf = w.float().clamp(min=1e-8)
        entropy = -(wf * wf.log()).sum(-1)
        entropy = entropy.masked_fill(
            ~key_block_mask, torch.finfo(entropy.dtype).min)

        weighted_k = torch.einsum("bhqks,bhksd->bhqkd", w, kr)
        alpha_logits = (
            torch.einsum("bhqtd,bhqkd->bhqtk", q_chunk, weighted_k) * scale
            + entropy.unsqueeze(3)
        )
        alpha_logits = alpha_logits.masked_fill(
            ~key_block_mask.unsqueeze(3), torch.finfo(alpha_logits.dtype).min)
        alpha = F.softmax(alpha_logits.float(), dim=-1).to(q.dtype)

        weighted_v = torch.einsum("bhqks,bhksd->bhqkd", w, vr)
        out_blocks[:, :, start:end] = torch.einsum(
            "bhqtk,bhqkd->bhqtd", alpha, weighted_v)

    out = out_blocks.reshape(B, H, -1, value_dim)[:, :, :Lq]
    return out.transpose(1, 2).contiguous()


def bm41_attn_matrix(q, k, block_size, softmax_scale=None, grid_size=None):
    """Return the amortized BM41 attention matrix for [B, H, L, D] inputs."""
    if q.dim() != 4 or k.dim() != 4:
        raise ValueError("q and k must be rank-4 tensors [B, H, L, D]")
    if grid_size is not None:
        if q.size(2) != k.size(2):
            raise ValueError(
                "grid-aligned BM41 attention matrix requires equal q/k lengths")
        scale = q.size(-1) ** -0.5 if softmax_scale is None else softmax_scale
        qr, _, _ = _grid_blockify(q, grid_size, block_size)
        kr, _, _ = _grid_blockify(k, grid_size, block_size)
        B, H, R, block, _ = qr.shape
        w = F.softmax(
            torch.einsum("bhrd,bhcsd->bhrcs", qr.mean(3), kr).float() * scale,
            dim=-1,
        ).to(q.dtype)
        wf = w.float().clamp(min=1e-8)
        entropy = -(wf * wf.log()).sum(-1)
        weighted_k = torch.einsum("bhrcs,bhcsd->bhrcd", w, kr)
        alpha_logits = (
            torch.einsum("bhrtd,bhrcd->bhrtc", qr, weighted_k) * scale
            + entropy.unsqueeze(3)
        )
        alpha = F.softmax(alpha_logits.float(), dim=-1).to(q.dtype)
        attn = (alpha.unsqueeze(-1) * w.unsqueeze(3))
        attn = attn.reshape(B, H, R * block, R * block)
        block_to_seq = _grid_block_order_indices(grid_size, block_size, q.device)
        seq_to_block = torch.empty_like(block_to_seq)
        seq_to_block[block_to_seq] = torch.arange(
            block_to_seq.numel(), device=q.device)
        return attn[:, :, seq_to_block][:, :, :, seq_to_block]
    _, _, w, alpha, _, _, q_len, k_len, _ = _bm41_w_alpha(
        q, k, block_size, softmax_scale=softmax_scale)
    attn = (alpha.unsqueeze(-1) * w.unsqueeze(3))
    B, H, _, block, _, _ = attn.shape
    return attn.reshape(B, H, -1, attn.size(4) * block)[:, :, :q_len, :k_len]
