from functools import partial

import torch

try:
    from torch.nn.attention.flex_attention import (  # noqa
        create_block_mask,
        flex_attention,
    )

    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False


def block_diff_mask_fn(b, h, q_idx, kv_idx, block_size, n):
    """
    FlexAttention mask function for block diffusion.

    The mask has three components:
    - Block Diagonal: Self-attention within noised blocks
    - Offset Block Causal: Cross-attention for conditional context
    - Block Causal: Attention to update x0

    Args:
        b, h: Batch and head indices (unused for mask logic)
        q_idx, kv_idx: Query and Key indices
        block_size: Size of each block
        n: Sequence length of x0 (and xt)

    Returns:
        Boolean mask (True = attend, False = mask)
    """
    # Indicate whether token belongs to xt (noisy, first half) or x0 (clean, second half)
    x0_flag_q = q_idx >= n
    x0_flag_kv = kv_idx >= n

    # Compute block indices
    block_q = torch.where(x0_flag_q, (q_idx - n) // block_size, q_idx // block_size)
    block_kv = torch.where(x0_flag_kv, (kv_idx - n) // block_size, kv_idx // block_size)

    # Block Diagonal Mask (M_BD): within same block, same sequence type
    block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)

    # Offset Block-Causal Mask (M_OBC): xt attends to previous x0 blocks
    offset_block_causal = (block_q > block_kv) & ~x0_flag_q & x0_flag_kv

    # Block-Causal Mask (M_BC): x0 attends to current and previous x0 blocks
    block_causal = (block_q >= block_kv) & x0_flag_q & x0_flag_kv

    # Combine masks
    return block_diagonal | offset_block_causal | block_causal


def create_block_diffusion_mask(
    seq_len, block_size, device, attn_implementation="sdpa"
):
    """
    Create block diffusion attention mask using FlexAttention if available,
    otherwise fall back to dense mask.
    """
    n = seq_len
    full_len = seq_len * 2

    if FLEX_ATTENTION_AVAILABLE and attn_implementation == "flex_attention":
        # Use FlexAttention's compiled block mask (much faster)
        mask_fn = partial(block_diff_mask_fn, block_size=block_size, n=n)
        block_mask = create_block_mask(
            mask_fn,
            B=None,  # Will broadcast across batch
            H=None,  # Will broadcast across heads
            Q_LEN=full_len,
            KV_LEN=full_len,
            device=device,
        )
        return block_mask
    else:
        # Fallback: Dense mask for SDPA
        q_idx = torch.arange(full_len, device=device)[:, None]
        kv_idx = torch.arange(full_len, device=device)[None, :]

        x0_flag_q = q_idx >= n
        x0_flag_kv = kv_idx >= n

        block_q = torch.where(x0_flag_q, (q_idx - n) // block_size, q_idx // block_size)
        block_kv = torch.where(
            x0_flag_kv, (kv_idx - n) // block_size, kv_idx // block_size
        )

        block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)
        offset_block_causal = (block_q > block_kv) & x0_flag_kv & ~x0_flag_q
        block_causal = (block_q >= block_kv) & x0_flag_kv & x0_flag_q

        mask = block_diagonal | offset_block_causal | block_causal

        attn_mask = torch.zeros(
            (1, 1, full_len, full_len), dtype=torch.bfloat16, device=device
        )
        attn_mask.masked_fill_(~mask, float("-inf"))

        return attn_mask


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
