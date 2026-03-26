#!/usr/bin/env python3

import os
import math
import time
import gc
from functools import partial
from tqdm import tqdm

import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state, checkpoints
from jax.tree_util import tree_map
import optax

# ==========================================================
# Configuration Parameters
# ==========================================================

DATA_PATH = "/mnt/disks/pythia/document.bin"
CKPT_DIR = os.path.abspath("./checkpoints")

# Validation settings
VAL_SPLIT = 0.01          # Allocate 1% of the dataset for validation
VAL_TOKENS = 5_000_000    # Number of tokens to process during each validation run
VAL_INTERVAL = 1_000_000_000  # Trigger validation every 1 billion tokens
LOG_FILE = "training_log.txt"
PPL_FILE = "val_perplexity.txt"

# Model hyperparameters
SEQ_LEN = 1024
VOCAB_SIZE = 50304

# 520M Parameter GPT Configuration
N_LAYERS = 32
D_MODEL = 1024
N_HEADS = 16
D_HEAD = 64
D_FF = 4096
ROTARY_PCT = 0.25         # Percentage of head dimension to apply RoPE

# Training targets
TOTAL_TOKENS_TARGET = 150_000_000_000
WARMUP_FRAC = 0.01

# Checkpoint intervals
SANITY_TOKENS = 10_000_000
CKPT_INTERVAL = 40_000_000_000

# Optimizer settings
LR_MAX = 2.5e-4
LR_MIN = 2e-5
WEIGHT_DECAY = 0.1
CLIP_NORM = 1.0

# Batch settings
PER_DEVICE_BATCH = 16
GRAD_ACCUM = 1

# ==========================================================
# Device and Batch Initialization
# ==========================================================

DEVICE_COUNT = jax.device_count()
GLOBAL_BATCH = PER_DEVICE_BATCH * DEVICE_COUNT
TOKENS_PER_STEP = GLOBAL_BATCH * SEQ_LEN * GRAD_ACCUM

os.makedirs(CKPT_DIR, exist_ok=True)

print(f"Devices available: {DEVICE_COUNT}")
print(f"Tokens processed per step: {TOKENS_PER_STEP}")

# ==========================================================
# Model Architecture
# ==========================================================

def build_rope_cache(seq_len: int, head_dim: int, rotary_pct: float):
    """
    Precomputes the sine and cosine values for Rotary Position Embeddings (RoPE).
    """
    dim = int(head_dim * rotary_pct)
    freqs = 1.0 / (10000 ** (jnp.arange(0, dim, 2) / dim))
    pos = jnp.arange(seq_len)
    angles = jnp.einsum("i,j->ij", pos, freqs)
    
    sin = jnp.sin(angles)
    cos = jnp.cos(angles)
    return sin, cos

ROPE_SIN, ROPE_COS = build_rope_cache(SEQ_LEN, D_HEAD, ROTARY_PCT)

def apply_rope(q: jnp.ndarray, k: jnp.ndarray):
    """
    Applies Rotary Position Embeddings to query and key tensors.
    """
    dim = int(D_HEAD * ROTARY_PCT)

    # Expand dimensions for broadcasting: [1, Sequence Length, 1, Dimension]
    sin = ROPE_SIN[None, :, None, :]
    cos = ROPE_COS[None, :, None, :]

    # Split features into rotation and pass-through segments
    q_rot, q_pass = q[..., :dim], q[..., dim:]
    k_rot, k_pass = k[..., :dim], k[..., dim:]

    # Alternate elements for rotation
    q1, q2 = q_rot[..., ::2], q_rot[..., 1::2]
    k1, k2 = k_rot[..., ::2], k_rot[..., 1::2]

    # Apply rotation matrices
    q_rot = jnp.concatenate([q1 * cos - q2 * sin, q1 * sin + q2 * cos], axis=-1)
    k_rot = jnp.concatenate([k1 * cos - k2 * sin, k1 * sin + k2 * cos], axis=-1)

    # Recombine segments
    return (
        jnp.concatenate([q_rot, q_pass], axis=-1),
        jnp.concatenate([k_rot, k_pass], axis=-1),
    )

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization layer.
    """
    dim: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        scale = self.param("scale", nn.initializers.ones, (self.dim,))
        norm = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return x * (scale / norm)

class Attention(nn.Module):
    """
    Multi-Head Causal Attention block utilizing RoPE.
    """
    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        B, T, C = x.shape

        # Fused linear projection for Query, Key, and Value
        qkv = nn.Dense(3 * C, use_bias=False, dtype=jnp.bfloat16)(x)
        qkv = qkv.reshape(B, T, 3, N_HEADS, D_HEAD)

        q = qkv[:, :, 0]
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]

        q, k = apply_rope(q, k)

        # Compute attention scores
        att = jnp.einsum("bthd,bshd->bhts", q, k)
        att = att / math.sqrt(D_HEAD)
        
        # Apply causal mask
        mask = mask.astype(jnp.float32)
        mask = (1.0 - mask) * -1e10
        att = att + mask
        
        att = nn.softmax(att.astype(jnp.float32), axis=-1)
        att = att.astype(jnp.bfloat16)
        
        # Compute final output
        out = jnp.einsum("bhts,bshd->bthd", att, v)
        out = out.reshape(B, T, C)

        return nn.Dense(C, use_bias=False, dtype=jnp.bfloat16)(out)

class Block(nn.Module):
    """
    Standard Transformer block containing Attention and Feed-Forward networks.
    """
    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        # Pre-norm attention
        h = RMSNorm(D_MODEL)(x)
        h = Attention()(h, mask)
        x = x + h

        # Pre-norm feed-forward
        h = RMSNorm(D_MODEL)(x)
        h = nn.Dense(D_FF, dtype=jnp.bfloat16)(h)
        h = nn.gelu(h)
        h = nn.Dense(D_MODEL, dtype=jnp.bfloat16)(h)

        return x + h

class GPT(nn.Module):
    """
    Full GPT language model architecture.
    """
    @nn.compact
    def __call__(self, input_ids: jnp.ndarray) -> jnp.ndarray:
        batch, seq_len = input_ids.shape

        # Generate causal mask once per forward pass
        mask = nn.attention.make_causal_mask(
            jnp.ones((batch, seq_len), dtype=jnp.bool_)
        )

        x = nn.Embed(
            VOCAB_SIZE,
            D_MODEL,
            embedding_init=nn.initializers.normal(0.02),
            dtype=jnp.bfloat16,
        )(input_ids)

        # Utilize gradient rematerialization to save memory
        RematBlock = nn.remat(Block)

        for _ in range(N_LAYERS):
            x = RematBlock()(x, mask)

        x = RMSNorm(D_MODEL)(x)

        logits = nn.Dense(
            VOCAB_SIZE,
            use_bias=False,
            dtype=jnp.bfloat16
        )(x)

        return logits

# ==========================================================
# Training Utilities
# ==========================================================

def create_state() -> train_state.TrainState:
    """
    Initializes the model, parameters, and optimizer state.
    """
    model = GPT()
    rng = jax.random.PRNGKey(0)

    params = model.init(
        rng,
        jnp.ones((1, SEQ_LEN), dtype=jnp.int32)
    )

    tx = optax.chain(
        optax.clip_by_global_norm(CLIP_NORM),
        optax.adamw(
            learning_rate=lr_schedule,
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=WEIGHT_DECAY,
        )
    )

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )

def lr_schedule(step: int) -> float:
    """
    Computes the learning rate for a given step using a cosine decay schedule with warmup.
    """
    step = jnp.asarray(step, dtype=jnp.float32)
    tokens_seen = step * jnp.asarray(TOKENS_PER_STEP, dtype=jnp.float32)

    total_tokens = jnp.asarray(TOTAL_TOKENS_TARGET, dtype=jnp.float32)
    warmup_tokens = total_tokens * WARMUP_FRAC

    def warmup():
        return LR_MAX * tokens_seen / warmup_tokens

    def cosine():
        progress = (tokens_seen - warmup_tokens) / (total_tokens - warmup_tokens)
        cosine_val = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
        return LR_MIN + (LR_MAX - LR_MIN) * cosine_val

    return jax.lax.cond(
        tokens_seen < warmup_tokens,
        lambda _: warmup(),
        lambda _: cosine(),
        operand=None
    )

# ==========================================================
# Execution Steps
# ==========================================================

@partial(jax.pmap, axis_name="data")
def train_step(state: train_state.TrainState, batch: dict):
    """
    Executes a single distributed training step.
    """
    def loss_fn(params):
        logits = state.apply_fn(params, batch["input_ids"])
        logits = jnp.clip(logits, -30.0, 30.0)

        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits.astype(jnp.float32),
            batch["labels"]
        )
        return loss.mean()

    loss, grads = jax.value_and_grad(loss_fn)(state.params)

    # Average gradients and loss across all devices
    grads = jax.lax.pmean(grads, axis_name="data")
    loss = jax.lax.pmean(loss, axis_name="data")

    updates, new_opt_state = state.tx.update(
        grads, state.opt_state, state.params
    )
    new_params = optax.apply_updates(state.params, updates)

    new_state = state.replace(
        step=state.step + 1,
        params=new_params,
        opt_state=new_opt_state
    )

    return new_state, loss

@partial(jax.pmap, axis_name="data")
def eval_step(state: train_state.TrainState, batch: dict):
    """
    Executes a single distributed evaluation step.
    """
    def loss_fn(params):
        logits = state.apply_fn(params, batch["input_ids"])
        logits = jnp.clip(logits, -30.0, 30.0)

        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits.astype(jnp.float32),
            batch["labels"]
        )
        return loss.mean()

    loss = loss_fn(state.params)
    loss = jax.lax.pmean(loss, axis_name="data")
    return loss

def save_checkpoint_safe(state: train_state.TrainState, tokens_seen: int):
    """
    Safely saves the model state to disk from the primary process.
    """
    if jax.process_index() != 0:
        return

    print(f"Saving checkpoint at {tokens_seen} tokens")

    ckpt_dir = os.path.join(CKPT_DIR, f"ckpt_{tokens_seen}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Extract state from device replica
    host_state = tree_map(lambda x: x[0], state)
    cpu_state = jax.device_get(host_state)
    jax.block_until_ready(cpu_state)

    checkpoints.save_checkpoint(
        ckpt_dir,
        cpu_state,
        step=0,
        overwrite=True,
        keep=1,
    )

    del cpu_state
    del host_state
    gc.collect()

# ==========================================================
# Main Training Loop
# ==========================================================

def main():
    state = create_state()
    state = jax.device_put_replicated(state, jax.devices())

    train_data = np.memmap(DATA_PATH, dtype=np.uint16, mode="r")

    # Segment dataset
    split_idx = int(len(train_data) * (1 - VAL_SPLIT))
    train_tokens = train_data[:split_idx]
    val_tokens   = train_data[split_idx:]

    tokens_seen = 0
    train_ptr = 0
    val_ptr = 0

    next_ckpt = CKPT_INTERVAL
    next_val  = VAL_INTERVAL
    sanity_done = False

    pbar = tqdm(total=TOTAL_TOKENS_TARGET, unit="tok", unit_scale=True)

    while tokens_seen < TOTAL_TOKENS_TARGET:

        # --------------------------------------------------
        # Training Phase
        # --------------------------------------------------
        seqs = []
        for _ in range(GLOBAL_BATCH):
            if train_ptr + SEQ_LEN + 1 >= len(train_tokens):
                train_ptr = 0
            seq = train_tokens[train_ptr:train_ptr + SEQ_LEN + 1]
            train_ptr += SEQ_LEN + 1
            seqs.append(seq)

        batch_np = np.stack(seqs).reshape(
            DEVICE_COUNT,
            PER_DEVICE_BATCH,
            SEQ_LEN + 1
        )

        batch = {
            "input_ids": batch_np[:, :, :-1].astype(np.int32),
            "labels":    batch_np[:, :, 1:].astype(np.int32),
        }

        state, loss = train_step(state, batch)
        jax.block_until_ready(loss)

        tokens_seen += TOKENS_PER_STEP
        train_loss = float(loss[0])

        pbar.set_postfix({"loss": f"{train_loss:.3f}"})
        pbar.update(TOKENS_PER_STEP)

        # --------------------------------------------------
        # Early Sanity Checkpoint
        # --------------------------------------------------
        if not sanity_done and tokens_seen >= SANITY_TOKENS:
            save_checkpoint_safe(state, tokens_seen)
            sanity_done = True

        # --------------------------------------------------
        # Validation Phase
        # --------------------------------------------------
        if tokens_seen >= next_val:
            total_val_loss = 0.0
            val_steps = max(1, VAL_TOKENS // TOKENS_PER_STEP)

            for _ in range(val_steps):
                seqs = []
                for _ in range(GLOBAL_BATCH):
                    if val_ptr + SEQ_LEN + 1 >= len(val_tokens):
                        val_ptr = 0
                    seq = val_tokens[val_ptr:val_ptr + SEQ_LEN + 1]
                    val_ptr += SEQ_LEN + 1
                    seqs.append(seq)

                batch_np = np.stack(seqs).reshape(
                    DEVICE_COUNT,
                    PER_DEVICE_BATCH,
                    SEQ_LEN + 1
                )

                batch = {
                    "input_ids": batch_np[:, :, :-1].astype(np.int32),
                    "labels":    batch_np[:, :, 1:].astype(np.int32),
                }

                val_loss = eval_step(state, batch)
                jax.block_until_ready(val_loss)
                total_val_loss += float(val_loss[0])

            avg_val_loss = total_val_loss / val_steps
            val_perplexity = math.exp(avg_val_loss)

            log_line = (
                f"{tokens_seen} tokens | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={avg_val_loss:.4f}\n"
            )

            print(log_line.strip())

            with open(LOG_FILE, "a") as f:
                f.write(log_line)
            with open(PPL_FILE, "a") as f:
                f.write(
                    f"{tokens_seen} tokens | "
                    f"val_perplexity={val_perplexity:.6f}\n"
                )

            next_val += VAL_INTERVAL

        # --------------------------------------------------
        # Standard Checkpoint Phase
        # --------------------------------------------------
        if tokens_seen >= next_ckpt:
            save_checkpoint_safe(state, tokens_seen)
            next_ckpt += CKPT_INTERVAL

    # --------------------------------------------------
    # Final Operations
    # --------------------------------------------------
    print("Saving FINAL checkpoint...")
    save_checkpoint_safe(state, tokens_seen)
    print("Training complete.")

if __name__ == "__main__":
    main()
