# Instinct-0.5B

# Instinct-1-0.5B

**Instinct-1-0.5B** is a 500M parameter Large Language Model built entirely from scratch under the **AutonomousX** organization. 

Compute for this project was generously supported by **Google's TRC Program (TPU Research Cloud)**.

---

### 👨‍💻 Author Information
**Rohit Yadav** B.Tech 3rd Year  
Dr. B.R. Ambedkar National Institute of Technology (NIT) Jalandhar, India  
**E-mail:** [yrohit1825@gmail.com](mailto:yrohit1825@gmail.com)  
**LinkedIn:** [Rohit Yadav](https://www.linkedin.com/in/rohit-yadav-25535b256/)  
**GitHub:** [YADAV1825](https://github.com/YADAV1825)

**Research interests include:** Large Language Models, MultiModal Pipelines, Systems Programming, AI Infrastructure, Distributed Training.

---

### ⚠️ Disclaimer
**This is a base model, not an SFT (Supervised Fine-Tuned) or RLHF (Reinforcement Learning from Human Feedback) model.** As a raw completion model, it may output undesired, biased, or nonsensical text. It is intended primarily for research and educational purposes.

---

### 📊 Model Overview

| Attribute | Value |
| :--- | :--- |
| **Model Name** | Instinct-1-0.5B |
| **Organization** | AutonomousX |
| **Parameters** | 500 Million |
| **Vocabulary Size** | 50,304 |
| **Dataset** | [The Pile by EleutherAI](https://pile.eleuther.ai/) |
| **Tokenizer** | Pythia Tokenizer |
| **Tokens Seen** | 150 Billion |
| **Training Hardware** | TPU v4-8 |
| **Optimizer** | AdamW |
| **Initial Loss** | 10.82 |
| **Final Roll shards Validation Loss** | ~2.41 |

*Validation was performed using rolling validation shards of The PILE dataset.*

---

### 🧠 Training Details

Instinct-1-0.5B was trained completely **from scratch** using **JAX/Flax on TPU v4-8 hardware**. 

The training pipeline includes:
* Dataset streaming from **The PILE by EleutherAI**.
* **Pythia Tokenizer** with a 50,304 vocabulary size.
* TPU optimized **JAX / Flax** training loop.
* **AdamW** optimizer for stable convergence.
* Checkpointing and validation during training.
* Rolling validation shard evaluation.

#### 📈 Training Curves
The loss curves are saved in `training_log.txt` and `val_perplexity.txt`. Below is the visualization of the training progress:

![Training Curves](training_curves.png)

---

### 🔄 Reproducibility

The entire pipeline used to train the model is fully reproducible. This includes the dataset pipeline, tokenizer creation, model architecture, TPU training loop, and checkpointing system.

**Full training pipeline repository:** [train.py](https://github.com/YADAV1825/Instinct-0.5B/blob/main/train.py)

---

### 🚀 Run Inference (Colab TPU/GPU)

The trained LLM inference script and model weights are available at: [autonomousX/Instinct-1-0.5B on Hugging Face](https://huggingface.co/autonomousX/Instinct-1-0.5B).

A ready-to-run Google Colab TPU/GPU inference script is provided below. Simply open a notebook, set your runtime to TPU or GPU, and run it. *(Please be patient, it may take around 20 mins to run the model initialization).*

<div style="max-height:600px; overflow:auto;">

```python
#please be patient It may take 20 mins to run the model
# Install huggingface_hub if not installed
!pip install -q huggingface_hub

from huggingface_hub import snapshot_download


repo_id = "autonomousX/Instinct-1-0.5B"

# Download entire repository
local_path = snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir="TPU_500m",
    local_dir_use_symlinks=False
)

print("Download complete!")
print("Saved to:", local_path)

# =========================
# FAST 500M INFERENCE CELL
# =========================

import os
import math
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax
from transformers import AutoTokenizer

# ---------------- CONFIG ----------------
SEQ_LEN = 1024
VOCAB_SIZE = 50304

N_LAYERS = 32
D_MODEL = 1024
N_HEADS = 16
D_HEAD = 64
D_FF = 4096
ROTARY_PCT = 0.25

CKPT_PATH = os.path.abspath("TPU_500m/checkpoint_0")

# ---------------- RoPE ----------------
def build_rope_cache(seq_len, head_dim, rotary_pct):
    dim = int(head_dim * rotary_pct)
    freqs = 1.0 / (10000 ** (jnp.arange(0, dim, 2) / dim))
    pos = jnp.arange(seq_len)
    angles = jnp.einsum("i,j->ij", pos, freqs)
    return jnp.sin(angles), jnp.cos(angles)

ROPE_SIN, ROPE_COS = build_rope_cache(SEQ_LEN, D_HEAD, ROTARY_PCT)

def apply_rope(q, k):
    dim = int(D_HEAD * ROTARY_PCT)
    T = q.shape[1]

    sin = ROPE_SIN[:T][None, :, None, :]
    cos = ROPE_COS[:T][None, :, None, :]

    q_rot, q_pass = q[..., :dim], q[..., dim:]
    k_rot, k_pass = k[..., :dim], k[..., dim:]

    q1, q2 = q_rot[..., ::2], q_rot[..., 1::2]
    k1, k2 = k_rot[..., ::2], k_rot[..., 1::2]

    q_rot = jnp.concatenate(
        [q1 * cos - q2 * sin,
         q1 * sin + q2 * cos],
        axis=-1
    )

    k_rot = jnp.concatenate(
        [k1 * cos - k2 * sin,
         k1 * sin + k2 * cos],
        axis=-1
    )

    return (
        jnp.concatenate([q_rot, q_pass], axis=-1),
        jnp.concatenate([k_rot, k_pass], axis=-1),
    )

# ---------------- MODEL ----------------
class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6
    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", nn.initializers.ones, (self.dim,))
        norm = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return x * (scale / norm)

class Attention(nn.Module):
    @nn.compact
    def __call__(self, x, mask):
        B, T, C = x.shape
        qkv = nn.Dense(3 * C, use_bias=False, dtype=jnp.bfloat16)(x)
        qkv = qkv.reshape(B, T, 3, N_HEADS, D_HEAD)

        q = qkv[:, :, 0]
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]

        q, k = apply_rope(q, k)

        att = jnp.einsum("bthd,bshd->bhts", q, k)
        att = att / math.sqrt(D_HEAD)

        mask = mask.astype(jnp.float32)
        mask = (1.0 - mask) * -1e10
        att = att + mask

        att = nn.softmax(att.astype(jnp.float32), axis=-1)
        att = att.astype(jnp.bfloat16)

        out = jnp.einsum("bhts,bshd->bthd", att, v)
        out = out.reshape(B, T, C)

        return nn.Dense(C, use_bias=False, dtype=jnp.bfloat16)(out)

class Block(nn.Module):
    @nn.compact
    def __call__(self, x, mask):
        h = RMSNorm(D_MODEL)(x)
        h = Attention()(h, mask)
        x = x + h

        h = RMSNorm(D_MODEL)(x)
        h = nn.Dense(D_FF, dtype=jnp.bfloat16)(h)
        h = nn.gelu(h)
        h = nn.Dense(D_MODEL, dtype=jnp.bfloat16)(h)

        return x + h

class GPT(nn.Module):
    @nn.compact
    def __call__(self, input_ids):
        batch, seq_len = input_ids.shape
        mask = nn.attention.make_causal_mask(
            jnp.ones((batch, seq_len), dtype=jnp.bool_)
        )

        x = nn.Embed(
            VOCAB_SIZE,
            D_MODEL,
            embedding_init=nn.initializers.normal(0.02),
            dtype=jnp.bfloat16,
        )(input_ids)

        RematBlock = nn.remat(Block)

        for _ in range(N_LAYERS):
            x = RematBlock()(x, mask)

        x = RMSNorm(D_MODEL)(x)

        return nn.Dense(
            VOCAB_SIZE,
            use_bias=False,
            dtype=jnp.bfloat16
        )(x)
# ---------------- LOAD CHECKPOINT ----------------
def create_state():
    model = GPT()
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, jnp.ones((1, SEQ_LEN), dtype=jnp.int32))
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adamw(1e-4),
    )

state = create_state()
state = checkpoints.restore_checkpoint(CKPT_PATH, state)
params = state.params
model = GPT()

print("Checkpoint loaded.")

@jax.jit
def forward(params, input_ids):
    return model.apply(params, input_ids)

import jax.random as random

import jax.random as random

def generate(params, input_ids, max_new_tokens=30, temperature=0.9, top_k=40):
    rng = random.PRNGKey(0)

    for _ in range(max_new_tokens):

        logits = model.apply(params, input_ids)
        logits = logits[:, -1, :]
        logits = logits.astype(jnp.float32)

        logits = logits / temperature

        top_k_logits, top_k_indices = jax.lax.top_k(logits, top_k)
        probs = jax.nn.softmax(top_k_logits, axis=-1)

        rng, subkey = random.split(rng)
        next_token_idx = random.categorical(subkey, jnp.log(probs))

        next_token = jnp.take_along_axis(
            top_k_indices,
            next_token_idx[:, None],
            axis=-1
        )

        input_ids = jnp.concatenate([input_ids, next_token], axis=1)

    return input_ids
# ---------------- RUN ----------------
tokenizer = AutoTokenizer.from_pretrained("autonomousX/Instinct-1-0.5B")

prompt = "I am John,"
tokens = tokenizer(prompt, return_tensors="np")
input_ids = jnp.array(tokens["input_ids"], dtype=jnp.int32)

output_ids = generate(params, input_ids, 200)

print("\n=== GENERATED TEXT ===\n")
print(tokenizer.decode(output_ids[0].tolist()))

```

</div>

About AutonomousX
AutonomousX focuses on open-source contributions aimed at building Large Language Models from scratch using custom training pipelines. Our work explores different training configurations including optimizers, datasets, and scalable TPU training using JAX and pmap. The goal is to provide transparent and reproducible implementations so that researchers, students, and developers can understand how modern LLMs are trained end-to-end.

Due to the current scarcity of complete beginner-friendly guides for training LLMs on TPUs, especially using JAX, AutonomousX aims to bridge this gap by publishing full training pipelines, scripts, and documentation for the open-source community.

Maintained by: Rohit Yadav | B.Tech NIT Jalandhar | yrohit1825@gmail.com | [Hugging_Face](https://huggingface.co/autonomousX)
