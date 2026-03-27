import os
import gc
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import Dataset
from tqdm.auto import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import bitsandbytes as bnb

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Config

SEED = 42
DATASET_REPO = "emuuuu/rearc_tasks"
DATASET_FILE = "rearc_tasks.json"
MODEL_ID     = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"
OUTPUT_DIR   = Path("./Qwen3-4B_adapter")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
METRICS_CSV    = OUTPUT_DIR / "metrics.csv"

FRACTION     = 0.10
VAL_SPLIT    = 0.05
MAX_TOKENS   = 3072

BATCH_SIZE   = 8
ACCUM_STEPS  = 8
LEARNING_RATE = 5e-5
WEIGHT_DECAY  = 0.01
NUM_EPOCHS    = 1
WARMUP_RATIO  = 0.05
GRAD_CLIP_NORM= 1.0
CHECK_EVERY   = 100  
EARLY_STOP_PATIENCE = 5

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

DEBUG = False
DEBUG_N = 2

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def bf16_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8

ATTN_IMPL = "flash_attention_2"
try:
    import flash_attn
except Exception:
    ATTN_IMPL = "sdpa"

dtype = torch.bfloat16 if bf16_supported() else torch.float16
mprec = "bf16" if dtype is torch.bfloat16 else "fp16"

from huggingface_hub import hf_hub_download

print(f"Loading tokenizer from: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    enable_thinking=False,
)

# Pad token hygiene
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def get_assistant_prefix_ids(tok):
    txt = tok.apply_chat_template(
        [{"role":"assistant","content":"<PAYLOAD>"}],
        tokenize=False, add_generation_prompt=False
    )
    k = txt.find("<PAYLOAD>")
    prefix = txt[:k]
    return tok.encode(prefix, add_special_tokens=False)

ASSISTANT_PREFIX_IDS = get_assistant_prefix_ids(tokenizer)
IM_END_IDS = tokenizer.encode("<|im_end|>", add_special_tokens=False)

print(f"Downloading dataset file {DATASET_FILE} from: {DATASET_REPO}")
json_path = hf_hub_download(
    repo_id=DATASET_REPO,
    filename=DATASET_FILE,
    repo_type="dataset"
)
print(f"Local dataset path: {json_path}")

with open(json_path, "r") as f:
    raw = json.load(f)

if not raw:
    raise SystemExit("No samples loaded from dataset.")

items = list(raw.items()) if isinstance(raw, dict) else [(str(i), s) for i, s in enumerate(raw)]

# Deterministic sub-sample
if 0 < FRACTION < 1:
    k = max(1, int(len(items) * FRACTION))
    rng = random.Random(SEED)
    items = rng.sample(items, k)
    print(f"Using {len(items)} tasks ({FRACTION*100:.1f}%) of the dataset.")

SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a world-class puzzle solver with exceptional pattern recognition skills. "
        "You will be given several examples of input-output pairs of matrices which relate via some pattern, "
        "like in an IQ matrix puzzle. You should be able to find a rule/transformation that lets you find the output "
        "matrix given the input matrix of all the examples in this task. Then based on the pattern you find you will "
        "apply that pattern transformation on the test input matrix, and "
        "give your solution output matrix enclosed in the special token ```."
    ),
}
END_USER_PROMPT = {
    "role": "user",
    "content": (
        "Now try to see the pattern, and apply the transformation on this input matrix. "
        "Don't explain your thought process, just return the output matrix enclosed in the special token ```."
    ),
}

def grid_str(mat):
    return "\n".join("".join(str(int(c)) for c in row) for row in mat)

processed = []
for task_id, task in items:
    train_pairs = task.get("train", [])
    test_pairs  = task.get("test", [])
    if not train_pairs or not test_pairs:
        continue

    parts = []
    for i, io in enumerate(train_pairs, 1):
        parts.append(
            f"Example {i}\n"
            f"Input:\n{grid_str(io['input'])}\n\n"
            f"Output:\n{grid_str(io['output'])}\n"
        )
    test_in  = test_pairs[0]["input"]
    test_out = test_pairs[0]["output"]

    parts.append("\nHere is the input grid for the test example:\nInput:\n" + grid_str(test_in))
    user_block = "\n\n".join(parts)

    msgs = [
        SYSTEM_PROMPT,
        {"role": "user", "content": user_block},
        END_USER_PROMPT,
        {"role": "assistant", "content": f"```\n{grid_str(test_out)}\n```"},
    ]

    processed.append({"id": task_id, "messages": msgs})

examples = []
for s_idx, s in enumerate(processed):
    try:
        text = tokenizer.apply_chat_template(
            s["messages"], tokenize=False, add_generation_prompt=False
        )
        examples.append({"text": text, "id": s.get("id", f"sample_{s_idx}")})
    except Exception as e:
        print(f"Error applying chat template for sample {s_idx}: {e}")

print(f"Created {len(examples)} text examples.")

raw_ds = Dataset.from_list(examples)

# Filter by token length
raw_ds = raw_ds.filter(
    lambda x: len(tokenizer.encode(x["text"], truncation=False, add_special_tokens=False)) <= MAX_TOKENS
)
print(f"Dataset size after MAX_TOKENS filter: {len(raw_ds)}")

# Split
if VAL_SPLIT > 0 and len(raw_ds) > 1:
    split = raw_ds.train_test_split(test_size=VAL_SPLIT, seed=SEED)
    train_ds = split["train"]
    val_ds   = split["test"]
else:
    train_ds = raw_ds
    val_ds = None

print(f"Train: {len(train_ds)} | Val: {len(val_ds) if val_ds else 0}")

# Label maskign

def _match_at(seq, pos, pat):
    L = len(pat)
    return L and pos + L <= len(seq) and seq[pos:pos+L] == pat

def _find_last(seq, pat):
    L = len(pat)
    if not L: return -1
    for i in range(len(seq)-L, -1, -1):
        if _match_at(seq, i, pat):
            return i
    return -1

def _maybe_debug_print(enc, labels):
    # Print only the actively attended text
    if not DEBUG:
        return
    ids, attn = enc["input_ids"], enc["attention_mask"]
    active_ids = [tid for tid, m in zip(ids, attn) if m == 1]
    text = tokenizer.decode(active_ids, skip_special_tokens=False)
    toks = tokenizer.convert_ids_to_tokens(ids)
    labv = ["PRED" if l != -100 else "HIDDEN" for l in labels]
    print("\n" + "="*80)
    print(text)
    print("-"*80)
    print(" ".join(f"[{t_ok}|{b}]" for t, b in zip(toks, labv)
               for t_ok in [t.replace("\n", "\\n")]))
    print("="*80 + "\n")


def tokenize_and_prepare_labels(ex):
    enc = tokenizer(
        ex["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_TOKENS,
        add_special_tokens=False,
    )
    ids = enc["input_ids"]
    attn= enc["attention_mask"]
    labels = [-100] * len(ids)

    a_start = _find_last(ids, ASSISTANT_PREFIX_IDS)
    if a_start != -1:
        j = a_start + len(ASSISTANT_PREFIX_IDS)
        # find following IM_END or pad
        a_stop = None
        for pos in range(j, len(ids) - len(IM_END_IDS) + 1):
            if _match_at(ids, pos, IM_END_IDS):
                a_stop = pos
                break
        if a_stop is None:
            # until padding
            try:
                a_stop = next(k for k in range(j, len(ids)) if ids[k] == tokenizer.pad_token_id)
            except StopIteration:
                a_stop = len(ids)

        for k in range(j, min(a_stop, len(ids))):
            if attn[k] == 1 and ids[k] != tokenizer.pad_token_id:
                labels[k] = ids[k]

    enc["labels"] = labels
    _maybe_debug_print(enc, labels)
    return enc

# Map
remove_cols = [c for c in train_ds.column_names if c not in ("text", "id")]
train_ds = train_ds.remove_columns(remove_cols)
train_ds = train_ds.map(tokenize_and_prepare_labels, batched=False, remove_columns=["text", "id"], desc="Tokenize train")

if val_ds:
    remove_cols = [c for c in val_ds.column_names if c not in ("text", "id")]
    val_ds = val_ds.remove_columns(remove_cols)
    val_ds = val_ds.map(tokenize_and_prepare_labels, batched=False, remove_columns=["text", "id"], desc="Tokenize val")

# Dataloaders
from torch.utils.data import DataLoader

def collate_fn(batch):
    keys = ["input_ids", "attention_mask", "labels"]
    out = {k: torch.tensor([b[k] for b in batch], dtype=torch.long) for k in keys}
    return out

num_workers = max(1, (os.cpu_count() or 2) - 1)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn,
                          num_workers=num_workers, pin_memory=True, drop_last=True)
val_loader = None
if val_ds:
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn,
                            num_workers=num_workers, pin_memory=True)

print(f"Train iters/epoch: {len(train_loader)} | Val iters: {len(val_loader) if val_loader else 0}")

# Model and LoRA
print(f"Loading base model from: {MODEL_ID}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation=ATTN_IMPL,
    dtype=dtype,
    low_cpu_mem_usage=True,
)

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()
model.config.use_cache = False

# Print trainable params
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params     = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable_params/1e6:.2f}M / Total: {total_params/1e6:.2f}M ({100*trainable_params/total_params:.2f}%)")

optimizer = bnb.optim.AdamW8bit(
    [
        {"params": [p for n,p in model.named_parameters() if p.requires_grad], "weight_decay": WEIGHT_DECAY},
    ],
    lr=LEARNING_RATE,
)

# Compute total steps
steps_per_epoch = math.ceil(len(train_loader) / ACCUM_STEPS)
t_total = steps_per_epoch * NUM_EPOCHS
warmup_steps = max(1, int(t_total * WARMUP_RATIO))

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=t_total,
)

accelerator = Accelerator(mixed_precision=mprec, gradient_accumulation_steps=ACCUM_STEPS, log_with=None)
model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
if val_loader:
    val_loader = accelerator.prepare(val_loader)

# Checkpoints
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

state = {
    "global_step": 0,
    "best_val": float("inf"),
    "no_improve": 0,
}

last_ckpt = CHECKPOINT_DIR / "last.pt"
if last_ckpt.exists():
    ck = torch.load(last_ckpt, map_location="cpu")
    accelerator.print("Resuming from last checkpoint…")
    state.update({k: ck.get(k, state[k]) for k in state})
    try:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.load_adapter(OUTPUT_DIR)  # load LoRA adapter if present
        accelerator.print("Loaded previous adapter weights.")
    except Exception:
        pass
    try:
        optimizer.load_state_dict(ck["optim"])  # may fail cross-version; best effort
        scheduler.load_state_dict(ck["sched"])  # best effort
    except Exception:
        accelerator.print("Warning: failed to load optimizer/scheduler state; continuing.")

# Metrics

def compute_token_accuracy(logits, labels):
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        mask = labels != -100
        correct = (preds[mask] == labels[mask]).sum().item()
        total   = mask.sum().item()
        return (correct / total) if total > 0 else 0.0

# Training
train_losses = []
val_points, val_losses, val_accs = [], [], []

accelerator.print(f"Starting training for {NUM_EPOCHS} epoch(s) | total opt steps: {t_total}")
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    model.train()
    running = 0.0
    progress = tqdm(train_loader, disable=not accelerator.is_local_main_process, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

    for step, batch in enumerate(progress, start=1):
        outputs = model(**batch)
        loss = outputs.loss
        running += loss.item()

        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        state["global_step"] += 1

        if accelerator.is_local_main_process:
            train_losses.append(loss.item())
            if step % 20 == 0:
                progress.set_postfix(loss=f"{running/step:.4f}")

        # ---- Validation ----
        if val_loader and state["global_step"] % CHECK_EVERY == 0:
            model.eval()
            tot_loss, tot_acc, n_batches = 0.0, 0.0, 0
            with torch.no_grad():
                for vb in val_loader:
                    vo = model(**vb)
                    tot_loss += vo.loss.item()
                    tot_acc  += compute_token_accuracy(vo.logits, vb["labels"])
                    n_batches += 1
            avg_vloss = tot_loss / max(1, n_batches)
            avg_vacc  = tot_acc / max(1, n_batches)

            if accelerator.is_local_main_process:
                val_points.append(state["global_step"])
                val_losses.append(avg_vloss)
                val_accs.append(avg_vacc)
                print(f"\nStep {state['global_step']}: val_loss={avg_vloss:.4f} | val_token_acc={avg_vacc:.4f}")

                improved = avg_vloss < state["best_val"] - 1e-6
                if improved:
                    state["best_val"] = avg_vloss
                    state["no_improve"] = 0
                    # Save best adapter
                    unwrapped = accelerator.unwrap_model(model)
                    unwrapped.save_pretrained(OUTPUT_DIR)
                    tokenizer.save_pretrained(OUTPUT_DIR)
                    print(f"Best adapter saved to {OUTPUT_DIR}")
                else:
                    state["no_improve"] += 1

                # Save last checkpoint (lightweight)
                torch.save({
                    **state,
                    "optim": optimizer.state_dict(),
                    "sched": scheduler.state_dict(),
                }, last_ckpt)

                if state["no_improve"] >= EARLY_STOP_PATIENCE:
                    print("Early stopping triggered.")
                    break
            model.train()
    if state["no_improve"] >= EARLY_STOP_PATIENCE:
        break

elapsed = time.time() - start_time
accelerator.print(f"Training finished in {elapsed/60:.1f} min. Best val loss: {state['best_val']:.4f}")

# ===================== Save final & plots =====================
if accelerator.is_local_main_process and state["global_step"] > 0:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Final adapter saved to {OUTPUT_DIR}")

    # Save metrics CSV
    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_CSV, "w") as f:
        f.write("global_step,train_loss\n")
        for i, v in enumerate(train_losses, start=1):
            f.write(f"{i},{v}\n")

    # Plot losses
    plt.figure(figsize=(9, 6))
    plt.plot(train_losses, label="Train step loss", alpha=0.6)
    if val_points and val_losses:
        plt.plot(val_points, val_losses, "o-", label=f"Val loss (every {CHECK_EVERY})")
    plt.title("LoRA SFT training losses")
    plt.xlabel("Global step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    fig_path = OUTPUT_DIR / "training_validation_loss.png"
    plt.savefig(fig_path)
    print(f"Loss plot saved to {fig_path}")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()