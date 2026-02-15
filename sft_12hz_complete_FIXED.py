# coding=utf-8
# Complete training script: Duration Head + Text Weighting + Early Stopping
import argparse
import json
import os
import random
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from transformers import AutoConfig
from huggingface_hub import snapshot_download
from tqdm import tqdm

target_speaker_embedding = None


# ============================================================================
# TEXT WEIGHTING UTILITIES
# ============================================================================

def compute_text_weights(data_source, weight_power=0.5, max_weight=4.0, min_weight=1.0):
    """
    Compute inverse-frequency weights for text samples.

    Args:
        data_source: Either a path to a JSONL file (str/Path) or a list of dicts
                     that already have a 'text' key.  Pass the training split list
                     (not the full file) so validation texts don't contaminate the
                     frequency counts.
        weight_power: Power to apply (0.5=sqrt, 1.0=linear)
        max_weight: Maximum weight cap (prevents gradient explosion)
        min_weight: Minimum weight floor
    
    Returns:
        tuple: (text_to_weight_dict, text_counts)
    """
    text_counts = Counter()

    # Accept either a file path or an already-loaded list of dicts.
    if isinstance(data_source, (str, Path)):
        with open(data_source, 'r', encoding='utf-8') as f:
            records = (json.loads(line) for line in f if line.strip())
    else:
        records = iter(data_source)

    for data in records:
        try:
            text = data.get('text', '').strip()
            if text:
                text_counts[text] += 1
        except Exception:
            continue
    
    if not text_counts:
        return {}, {}
    
    total_samples = sum(text_counts.values())
    text_weights = {}
    
    for text, count in text_counts.items():
        frequency = count / total_samples
        raw_weight = (1.0 / frequency) ** weight_power
        text_weights[text] = raw_weight
    
    # Normalize to mean=1.0
    mean_weight = sum(text_weights.values()) / len(text_weights)
    text_weights = {text: w / mean_weight for text, w in text_weights.items()}
    
    # Apply caps
    text_weights = {
        text: max(min_weight, min(max_weight, w)) 
        for text, w in text_weights.items()
    }
    
    return text_weights, text_counts


# ============================================================================
# DURATION PREDICTION HEAD
# ============================================================================

class DurationPredictionHead(nn.Module):
    """Duration prediction head for audio length consistency"""
    def __init__(self, hidden_size=2048, dropout=0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, hidden_states, codec_mask=None):
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            codec_mask:    [batch_size, seq_len] bool — True for audio frame positions.
                           When provided, only those positions are averaged so the
                           pooled vector reflects audio content, not text prompt length.
                           Pass codec_mask[:, 1:] to align with the T-1 hidden states.
        Returns:
            duration_pred: [batch_size] - predicted duration in seconds
        """
        if codec_mask is not None:
            # Masked mean-pool: sum over audio positions, divide by frame count.
            # Avoids mixing text-prompt hidden states (which carry no duration signal)
            # into the pooled vector.  Clamping to ≥1 prevents div-by-zero on any
            # edge-case empty sample that would otherwise produce NaN gradients.
            mask_f = codec_mask.unsqueeze(-1).float()           # [B, T, 1]
            counts = mask_f.sum(dim=1).clamp(min=1)             # [B, 1]
            pooled = (hidden_states * mask_f).sum(dim=1) / counts  # [B, H]
        else:
            pooled = hidden_states.mean(dim=1)
        duration_logits = self.projection(pooled)
        duration_pred = F.softplus(duration_logits.squeeze(-1))
        return duration_pred


class Qwen3TTSWithDuration(nn.Module):
    """Wrapper adding duration prediction to Qwen3TTS"""
    def __init__(self, base_model, hidden_size=2048):
        super().__init__()
        self.base_model = base_model
        self.duration_head = DurationPredictionHead(hidden_size=hidden_size)
        
    def forward(self, hidden_states, target_duration=None, codec_mask=None):
        duration_pred = self.duration_head(hidden_states, codec_mask=codec_mask)
        
        duration_loss = None
        if target_duration is not None:
            duration_loss = F.mse_loss(duration_pred, target_duration)
        
        return duration_pred, duration_loss


def extract_audio_duration(codec_mask, fps=12.5):
    """
    Extract duration from codec mask.
    Args:
        codec_mask: [batch_size, seq_len] bool — True for codec token positions
                    (use batch['codec_mask'] from the collate_fn, NOT codec_ids).
        fps: frames per second (12.5 for 12Hz tokenizer)
    Returns:
        durations: [batch_size] in seconds

    ⚠️  Do NOT pass codec_ids[:,:,0] != 0 here: codec token 0 is a valid
    codebook entry, so the != 0 test silently under-counts frames whenever
    the first codebook stream emits token 0, producing wrong duration targets.
    codec_mask is the ground-truth boolean mask written by collate_fn.
    """
    seq_lengths = codec_mask.sum(dim=1).float()
    durations = seq_lengths / fps
    return durations


# ============================================================================
# WEIGHTED DATASET
# ============================================================================

class WeightedTTSDataset(TTSDataset):
    """Extended dataset with sample weights"""
    def __init__(self, data, processor, config, text_weights=None):
        super().__init__(data, processor, config)
        self.text_weights = text_weights or {}
        self.data = data
        
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        
        # Add weight
        text = self.data[idx].get('text', '').strip()
        item['weight'] = self.text_weights.get(text, 1.0)
        item['sample_idx'] = idx
        
        return item

    def collate_fn(self, batch):
        # Collect per-sample weights BEFORE calling super(), which only
        # packs the fixed TTS keys and silently drops everything else.
        weights = torch.tensor([item['weight'] for item in batch], dtype=torch.float32)
        result = super().collate_fn(batch)
        result['weight'] = weights          # ← now available as batch['weight']
        return result


# ============================================================================
# EARLY STOPPING
# ============================================================================

class EarlyStopping:
    """Early stopping handler"""
    def __init__(self, patience=5, min_delta=0.001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        
    def __call__(self, val_loss, accelerator):
        score = -val_loss if self.mode == 'min' else val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if accelerator.is_main_process:
                accelerator.print(
                    f'⚠️  Early Stopping counter: {self.counter}/{self.patience} '
                    f'(best val_loss: {self.val_loss_min:.4f})'
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0
            
        return self.early_stop


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train():
    global target_speaker_embedding

    parser = argparse.ArgumentParser()
    
    # Basic args
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=250)
    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    
    # Duration prediction
    parser.add_argument("--duration_loss_weight", type=float, default=0.1,
                       help="Weight for duration prediction loss")
    parser.add_argument("--duration_hidden_size", type=int, default=2048,
                       help="Hidden size for duration head")
    
    # Text weighting
    parser.add_argument("--use_text_weighting", action="store_true",
                       help="Enable text-based loss weighting")
    parser.add_argument("--weight_power", type=float, default=0.5,
                       help="Power for inverse frequency (0.5=sqrt, 1.0=linear)")
    parser.add_argument("--max_sample_weight", type=float, default=3.5,
                       help="Maximum weight for rare samples")
    parser.add_argument("--min_sample_weight", type=float, default=1.0,
                       help="Minimum weight")
    
    # Early stopping
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.001)
    
    args = parser.parse_args()

    # Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with="tensorboard",
        project_dir=args.output_model_path
    )

    # ⚠️  init_trackers() MUST be called before any accelerator.log() calls.
    # log_with="tensorboard" in the constructor only declares WHICH tracker to
    # use; it does NOT initialise it.  Without this call every accelerator.log()
    # is a silent no-op: TensorBoard receives nothing, and check_progress.py
    # will never see any data no matter how long training runs.
    accelerator.init_trackers(
        project_name=os.path.basename(args.output_model_path),
        config=vars(args)          # logs all hyperparameters to TensorBoard
    )

    MODEL_PATH = args.init_model_path
    
    # Compute text weights if enabled.
    # ⚠️  MUST run on ALL processes — not just main — because every GPU
    # independently constructs its own dataset.  If only the main process
    # computed text_weights, the other three processes would keep text_weights
    # as None, fall back to plain TTSDataset, and silently train without any
    # text-frequency weighting — completely defeating the feature.
    text_weights = None
    text_counts = None
    # text_weights are computed AFTER the data split below, from train_data[:train_size] only,
    # so that validation samples do not affect the frequency table.

    # Log configuration
    if accelerator.is_main_process:
        num_gpus = accelerator.num_processes
        effective_batch = args.batch_size * num_gpus * args.gradient_accumulation_steps
        accelerator.print(f"{'='*60}")
        accelerator.print(f"🚀 Training Configuration:")
        accelerator.print(f"  Model: {MODEL_PATH}")
        accelerator.print(f"  GPUs: {num_gpus}")
        accelerator.print(f"  Batch per GPU: {args.batch_size}")
        accelerator.print(f"  Gradient Accumulation: {args.gradient_accumulation_steps}")
        accelerator.print(f"  Effective Batch: {effective_batch}")
        accelerator.print(f"  Learning Rate: {args.lr}")
        accelerator.print(f"  Epochs: {args.num_epochs}")
        accelerator.print(f"  Duration Loss Weight: {args.duration_loss_weight}")
        accelerator.print(f"  Text Weighting: {'✓ Enabled' if args.use_text_weighting else '✗ Disabled'}")
        if args.use_text_weighting:
            accelerator.print(f"    Power: {args.weight_power}")
            accelerator.print(f"    Range: {args.min_sample_weight:.1f}x - {args.max_sample_weight:.1f}x")
        accelerator.print(f"  Early Stopping: {'✓ Enabled' if args.early_stopping else '✗ Disabled'}")
        if args.early_stopping:
            accelerator.print(f"    Patience: {args.early_stopping_patience}")
            accelerator.print(f"    Min Delta: {args.early_stopping_min_delta}")
        accelerator.print(f"{'='*60}\n")

    # Load model
    # Use flash_attention_2 if available (not present in piplist.txt — falls back to sdpa).
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"
    accelerator.print(f"Loading model from {MODEL_PATH} (attn={attn_impl})...")
    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    
    # qwen3tts.model is a custom wrapper (not a HuggingFace PretrainedModel).
    # The actual transformer — the one that runs forward passes, has attention
    # layers, and generates hidden states — lives at qwen3tts.model.talker.
    #
    # ❌  WRONG (original):
    #   hasattr(qwen3tts.model, "gradient_checkpointing_enable") → False on a
    #   custom class → gradient checkpointing silently never enabled → OOM risk.
    #   qwen3tts.model.config.use_cache = False → sets it on the wrapper's config;
    #   the transformer's own config is unchanged, KV cache grows during training.
    #
    # ✅  CORRECT: target the talker, which IS the HuggingFace LM.
    qwen3tts.model.talker.gradient_checkpointing_enable()
    qwen3tts.model.talker.config.use_cache = False

    config = AutoConfig.from_pretrained(MODEL_PATH)
    
    # Get actual path
    if os.path.isdir(MODEL_PATH):
        actual_model_path = MODEL_PATH
    else:
        actual_model_path = snapshot_download(MODEL_PATH)

    # Wrap with duration head
    duration_wrapper = Qwen3TTSWithDuration(
        qwen3tts.model, 
        hidden_size=args.duration_hidden_size
    )
    accelerator.print("✓ Duration prediction head initialized")

    # Restore duration head weights from the init checkpoint if present.
    #
    # save_checkpoint() writes duration_head.* keys into model.safetensors
    # alongside talker.* and speaker_encoder.*.  Qwen3TTSModel.from_pretrained()
    # doesn't know about the duration head, so those keys are treated as
    # "unexpected" and silently discarded — leaving the head randomly initialised.
    # Without this block every training round re-learns the duration head from
    # scratch, discarding the previous round's progress entirely.
    try:
        from safetensors.torch import load_file as _load_sf
        _sf_path = Path(actual_model_path) / "model.safetensors"
        if _sf_path.exists():
            _ckpt = _load_sf(str(_sf_path), device="cpu")
            _dur_sd = {
                k[len("duration_head."):]: v
                for k, v in _ckpt.items()
                if k.startswith("duration_head.")
            }
            if _dur_sd:
                duration_wrapper.duration_head.load_state_dict(_dur_sd, strict=True)
                accelerator.print("✓ Restored duration head weights from checkpoint")
            else:
                accelerator.print("ℹ️  No duration_head.* keys found — duration head initialised fresh")
        else:
            accelerator.print("ℹ️  No model.safetensors found — duration head initialised fresh")
    except Exception as _e:
        accelerator.print(f"⚠️  Could not restore duration head ({_e}) — using fresh initialisation")

    # Load dataset
    train_data = open(args.train_jsonl).readlines()
    train_data = [json.loads(line) for line in train_data]

    # Shuffle before splitting so the validation set is a representative random
    # sample rather than the last N lines of the JSONL (which may be ordered by
    # speaker, recording session, or audio length).  Fixed seed keeps the split
    # identical across all 4 processes and reproducible across runs.
    random.seed(42)
    random.shuffle(train_data)
    
    # Split
    val_size = int(len(train_data) * args.val_split)
    train_size = len(train_data) - val_size

    # Compute text weights from the TRAINING split only.
    # ⚠️  Must run on ALL processes (not just main) because every GPU builds its own
    # dataset independently.  Reading the list is cheap; this avoids a broadcast.
    # ⚠️  Must run AFTER the split so that val samples don't skew the frequency table.
    if args.use_text_weighting:
        text_weights, text_counts = compute_text_weights(
            train_data[:train_size],          # ← training split only, not full JSONL
            weight_power=args.weight_power,
            max_weight=args.max_sample_weight,
            min_weight=args.min_sample_weight
        )

        if accelerator.is_main_process:
            accelerator.print(f"\n{'='*60}")
            accelerator.print("📊 Text-based loss weights computed (training split only)...")
            weights_list = list(text_weights.values())
            accelerator.print(f"  Unique texts: {len(text_weights)}")
            accelerator.print(f"  Weight range: {min(weights_list):.2f}x - {max(weights_list):.2f}x")
            accelerator.print(f"  Mean weight: {sum(weights_list)/len(weights_list):.2f}x")

            sorted_by_weight = sorted(text_weights.items(), key=lambda x: x[1], reverse=True)
            accelerator.print(f"\n  🔝 Top 5 weighted (rarest texts):")
            for text, weight in sorted_by_weight[:5]:
                count = text_counts.get(text, 0)
                preview = text[:50] + "..." if len(text) > 50 else text
                accelerator.print(f"    {weight:.2f}x - \"{preview}\" ({count} samples)")

            accelerator.print(f"\n  📊 Bottom 5 weighted (most common):")
            for text, weight in sorted_by_weight[-5:]:
                count = text_counts.get(text, 0)
                preview = text[:50] + "..." if len(text) > 50 else text
                accelerator.print(f"    {weight:.2f}x - \"{preview}\" ({count} samples)")
            accelerator.print(f"{'='*60}\n")
    
        train_dataset = WeightedTTSDataset(
            train_data[:train_size], 
            qwen3tts.processor, 
            config, 
            text_weights=text_weights
        )
        val_dataset = WeightedTTSDataset(
            train_data[train_size:], 
            qwen3tts.processor, 
            config, 
            text_weights=text_weights
        ) if val_size > 0 else None
        accelerator.print("✓ Using weighted dataset")
    else:
        train_dataset = TTSDataset(train_data[:train_size], qwen3tts.processor, config)
        val_dataset = TTSDataset(train_data[train_size:], qwen3tts.processor, config) if val_size > 0 else None
    
    accelerator.print(f"Dataset: {train_size} train, {val_size} validation")

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_dataloader = None
    if val_dataset:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=val_dataset.collate_fn,
            num_workers=2,
            pin_memory=True
        )

    # Optimizer - includes duration head parameters
    all_parameters = list(qwen3tts.model.parameters()) + list(duration_wrapper.duration_head.parameters())
    optimizer = AdamW(all_parameters, lr=args.lr, weight_decay=args.weight_decay)
    accelerator.print("✓ Optimizer initialized (full precision)")

    # Scheduler steps are counted in optimizer-step units (after gradient accumulation).
    # Two corrections are required:
    #   1. Divide by gradient_accumulation_steps: each step only updates the optimizer
    #      every N micro-steps, so the cosine period must be in optimizer-step units.
    #   2. Divide by num_processes: accelerator.prepare() wraps the DataLoader with a
    #      DistributedSampler that splits data across GPUs.  len(train_dataloader) is
    #      measured BEFORE prepare() and returns the FULL dataset length.  After prepare
    #      each process only iterates 1/num_processes of those batches, so without this
    #      correction num_training_steps is num_processes (4×) too large — the LR would
    #      only decay from 3e-6 to ~2.56e-6 (85% of peak) instead of reaching ~0.
    num_training_steps = (
        (len(train_dataloader) * args.num_epochs)
        // (args.gradient_accumulation_steps * accelerator.num_processes)
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Prepare
    # ⚠️  Only wrap duration_wrapper in DDP — NOT qwen3tts.model separately.
    # duration_wrapper.base_model IS qwen3tts.model.  If both are passed to
    # prepare(), Accelerate creates two separate DDP wrappers that both register
    # gradient-reduction hooks on the same shared parameter tensors.  Every
    # backward pass would all-reduce each base-model gradient TWICE, silently
    # corrupting the gradient scale relative to the duration head.
    duration_wrapper, optimizer, train_dataloader, scheduler = accelerator.prepare(
        duration_wrapper, optimizer, train_dataloader, scheduler
    )
    model = duration_wrapper  # alias; keeps all train/eval toggling and accumulate calls working
    
    if val_dataloader:
        val_dataloader = accelerator.prepare(val_dataloader)

    unwrapped_duration = accelerator.unwrap_model(duration_wrapper)
    unwrapped_model = unwrapped_duration.base_model  # the actual Qwen3TTS model
    
    # Speaker encoder check
    has_speaker_encoder = hasattr(unwrapped_model, 'speaker_encoder') and unwrapped_model.speaker_encoder is not None
    
    if has_speaker_encoder:
        accelerator.print("✓ Model has speaker_encoder")
    else:
        accelerator.print("⚠️  Using fixed speaker embedding")
        try:
            fixed_embedding = unwrapped_model.talker.model.codec_embedding.weight[3000].detach()
            target_speaker_embedding = fixed_embedding.unsqueeze(0)
            accelerator.print("  ✓ Loaded fixed speaker embedding")
        except Exception as e:
            accelerator.print(f"  ❌ Error: {e}")
            raise

    # Early stopping
    early_stopping = None
    if args.early_stopping:
        early_stopping = EarlyStopping(
            patience=args.early_stopping_patience,
            min_delta=args.early_stopping_min_delta,
            mode='min'
        )
        accelerator.print("✓ Early stopping initialized")

    # Training loop
    accelerator.print(f"\n{'='*60}")
    accelerator.print("🎯 Starting training...")
    accelerator.print(f"{'='*60}\n")
    
    model.train()
    duration_wrapper.train()
    global_step = 0
    best_val_loss = float('inf')
    training_stopped_early = False

    for epoch in range(args.num_epochs):
        epoch_loss = 0
        epoch_acoustic_loss = 0
        epoch_duration_loss = 0
        epoch_high_weight_samples = 0
        
        progress_bar = tqdm(
            train_dataloader, 
            disable=not accelerator.is_main_process,
            desc=f"Epoch {epoch+1}/{args.num_epochs}"
        )
        
        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                # Extract sample weights
                sample_weights = batch.get('weight', torch.ones(batch['input_ids'].size(0)))
                sample_weights = sample_weights.to(unwrapped_model.device)
                
                # Standard TTS forward pass
                input_ids = batch['input_ids']
                codec_ids = batch['codec_ids']
                ref_mels = batch['ref_mels']
                text_embedding_mask = batch['text_embedding_mask']
                codec_embedding_mask = batch['codec_embedding_mask']
                attention_mask = batch['attention_mask']
                codec_0_labels = batch['codec_0_labels']
                codec_mask = batch['codec_mask']

                # Train speaker encoder for better voice adaptation (Option B)
                # Removed .detach() to allow gradients to flow to speaker_encoder
                speaker_embedding = unwrapped_model.speaker_encoder(
                    ref_mels.to(unwrapped_model.device).to(unwrapped_model.dtype)
                ) if has_speaker_encoder else target_speaker_embedding
                
                if target_speaker_embedding is None and has_speaker_encoder:
                    target_speaker_embedding = speaker_embedding

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = unwrapped_model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
                input_codec_embedding = unwrapped_model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                # Out-of-place speaker injection at position 6.
                # In-place [:, 6, :] = ... on a non-leaf tensor (product of an embedding
                # lookup and a mask) corrupts its autograd version counter.  That is safe
                # today only because no backward formula currently saves input_codec_embedding
                # as an input — but any future change to the talker or the codec_embedding
                # module could silently break training with a hard-to-diagnose RuntimeError.
                speaker_injection = torch.zeros_like(input_codec_embedding)
                speaker_injection[:, 6, :] = speaker_embedding
                input_codec_embedding = input_codec_embedding + speaker_injection

                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    codec_i_embedding = unwrapped_model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                outputs = unwrapped_model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True
                )

                # outputs.hidden_states is a tuple of (num_layers+1) tensors,
                # each shape [B, T, H].  [-1] selects the final transformer layer.
                # [0][-1] would be embedding_layer[last_sample_in_batch] = [T, H],
                # silently dropping the batch dim and corrupting all downstream ops.
                hidden_states = outputs.hidden_states[-1]   # [B, T-1, H] (shifted for next-token prediction)
                
                # FIX: Use consistent shifted codec_mask for both extractions to ensure alignment.
                # hidden_states[i] predicts codec_ids[i+1], so both need the shifted mask.
                shifted_codec_mask = codec_mask[:, 1:]      # [B, T-1] aligned with hidden_states
                talker_hidden_states = hidden_states[shifted_codec_mask]  # [N, H]
                talker_codec_ids = codec_ids[:, 1:][shifted_codec_mask]   # [N, 16] - properly aligned

                sub_talker_logits, sub_talker_loss = unwrapped_model.talker.forward_sub_talker_finetune(
                    talker_codec_ids, talker_hidden_states
                )

                # --- Acoustic loss ---
                if args.use_text_weighting:
                    # Recompute the talker loss per-sample from logits so each
                    # sample can be weighted individually.  Multiplying the already
                    # batch-averaged scalar by sample_weights.mean() would apply the
                    # same scale to every sample and defeat the purpose.
                    logits = outputs.logits                       # [B, T, vocab]
                    labels_shifted = codec_0_labels[:, 1:]        # [B, T]
                    per_token_loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        labels_shifted.reshape(-1),
                        ignore_index=-100,
                        reduction='none'
                    ).reshape(logits.size(0), -1)                 # [B, T]
                    valid_mask = (labels_shifted != -100).float()
                    tokens_per_sample = valid_mask.sum(dim=-1).clamp(min=1)
                    per_sample_main_loss = (per_token_loss * valid_mask).sum(dim=-1) / tokens_per_sample  # [B]

                    # Normalise weights so their mean equals 1.0 — this preserves
                    # the same overall gradient magnitude as the unweighted case.
                    norm_weights = sample_weights * (sample_weights.numel() / sample_weights.sum())
                    weighted_main_loss = (per_sample_main_loss * norm_weights).mean()
                    # sub_talker_loss is already a scalar (no batch dim); scale by
                    # the batch-mean normalised weight as the best approximation.
                    acoustic_loss = weighted_main_loss + 0.3 * sub_talker_loss * norm_weights.mean()
                    weighted_acoustic_loss = acoustic_loss
                else:
                    acoustic_loss = outputs.loss + 0.3 * sub_talker_loss
                    weighted_acoustic_loss = acoustic_loss
                    norm_weights = torch.ones(input_ids.size(0), device=input_ids.device)

                # --- Duration prediction ---
                target_duration = extract_audio_duration(codec_mask, fps=12.5).to(hidden_states.device)
                # Don't pass target_duration into the wrapper — compute the loss here
                # so we can apply per-sample weights before reducing.
                # codec_mask[:, 1:] aligns with the T-1 hidden_states (input was sliced [:,:-1,:]).
                duration_pred, _ = unwrapped_duration(hidden_states, None, codec_mask=codec_mask[:, 1:])
                per_sample_duration_loss = (duration_pred - target_duration).pow(2)  # [B]
                duration_loss = (per_sample_duration_loss * norm_weights).mean()
                weighted_duration_loss = duration_loss
                
                # Combined loss
                total_loss = weighted_acoustic_loss + args.duration_loss_weight * weighted_duration_loss

                # Backward
                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    # Clip once on the superset: duration_wrapper.parameters()
                    # already includes all base-model weights (duration_wrapper.base_model
                    # IS model), so clipping model.parameters() first would clip the
                    # shared transformer weights twice and halve their effective grad norm.
                    accelerator.clip_grad_norm_(duration_wrapper.parameters(), args.max_grad_norm)
                    # ⚠️  scheduler.step() MUST be inside this guard.
                    # accelerator.accumulate() wraps optimizer.step()/zero_grad() as
                    # no-ops on non-sync micro-steps, but it does NOT wrap the scheduler.
                    # Calling scheduler.step() every micro-step would advance the cosine
                    # curve gradient_accumulation_steps (3×) faster than the optimizer,
                    # causing LR to reach ~0 after just 1/9 of actual training.
                    scheduler.step()

                optimizer.step()
                optimizer.zero_grad()

            # Stats
            epoch_loss += total_loss.item()
            epoch_acoustic_loss += acoustic_loss.item()
            epoch_duration_loss += duration_loss.item()
            if args.use_text_weighting:
                epoch_high_weight_samples += (sample_weights > 1.5).sum().item()
            # global_step counts dataloader iterations (micro-steps), NOT optimizer
            # steps.  With gradient_accumulation_steps=3 the effective optimizer step
            # is global_step // 3.  eval_steps / save_steps therefore also refer to
            # dataloader iterations: e.g. eval_steps=500 → validation every ~167
            # optimizer steps.  This is intentional; do not divide here.
            global_step += 1

            # Logging
            if global_step % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                
                log_dict = {
                    'loss': f'{total_loss.item():.4f}',
                    'ac': f'{acoustic_loss.item():.4f}',
                    'dur': f'{duration_loss.item():.4f}',
                    'lr': f'{current_lr:.2e}'
                }
                
                if args.use_text_weighting:
                    log_dict['wgt'] = f'{sample_weights.mean().item():.2f}'
                
                progress_bar.set_postfix(log_dict)
                
                if accelerator.is_main_process:
                    log_metrics = {
                        "train/loss_total": total_loss.item(),
                        "train/loss_acoustic": acoustic_loss.item(),
                        "train/loss_duration": duration_loss.item(),
                        "train/learning_rate": current_lr,
                        "train/epoch": epoch
                    }
                    if args.use_text_weighting:
                        log_metrics["train/avg_sample_weight"] = sample_weights.mean().item()
                        log_metrics["train/max_sample_weight"] = sample_weights.max().item()
                    
                    accelerator.log(log_metrics, step=global_step)

            # Validation
            if val_dataloader and global_step % args.eval_steps == 0:
                val_metrics = evaluate(
                    model, duration_wrapper, val_dataloader, accelerator, 
                    has_speaker_encoder, target_speaker_embedding, args
                )
                
                if accelerator.is_main_process:
                    accelerator.log({
                        "eval/loss_total": val_metrics['total_loss'],
                        "eval/loss_acoustic": val_metrics['acoustic_loss'],
                        "eval/loss_duration": val_metrics['duration_loss'],
                        "eval/duration_mae": val_metrics['duration_mae']
                    }, step=global_step)
                    
                    accelerator.print(
                        f"\n📊 Validation - "
                        f"Total: {val_metrics['total_loss']:.4f}, "
                        f"Acoustic: {val_metrics['acoustic_loss']:.4f}, "
                        f"Duration MAE: {val_metrics['duration_mae']:.2f}s"
                    )
                    
                    # Save best
                    val_loss = val_metrics['total_loss']
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(
                            model, duration_wrapper, config, target_speaker_embedding, args, 
                            actual_model_path, "checkpoint-best", accelerator
                        )
                    
                    # Early stopping decision made on main process; break deferred
                    # until all processes have been notified (see broadcast below).
                    if early_stopping is not None:
                        if early_stopping(val_loss, accelerator):
                            accelerator.print(f"\n🛑 Early Stopping triggered!")
                            accelerator.print(f"   Best validation loss: {early_stopping.val_loss_min:.4f}")
                            training_stopped_early = True
                
                # Broadcast the stop decision to ALL processes BEFORE breaking.
                # If only process 0 calls `break`, the other three remain stuck
                # in the DDP all-reduce barrier waiting for it → deadlock / hang.
                stop_signal = torch.tensor([int(training_stopped_early)], device=accelerator.device)
                accelerator.broadcast(stop_signal)
                if stop_signal.item():
                    training_stopped_early = True
                    break
                
                model.train()
                duration_wrapper.train()

            # Periodic checkpoint
            if global_step % args.save_steps == 0 and accelerator.is_main_process:
                save_checkpoint(
                    model, duration_wrapper, config, target_speaker_embedding, args,
                    actual_model_path, f"checkpoint-step-{global_step}", accelerator
                )
        
        if training_stopped_early:
            break

        # End of epoch
        if accelerator.is_main_process:
            num_steps = len(train_dataloader)
            avg_epoch_loss     = epoch_loss          / max(num_steps, 1)
            avg_acoustic_loss  = epoch_acoustic_loss / max(num_steps, 1)
            avg_duration_loss  = epoch_duration_loss / max(num_steps, 1)

            accelerator.print(
                f"\n📉 Epoch {epoch+1} summary — "
                f"loss: {avg_epoch_loss:.4f}  "
                f"acoustic: {avg_acoustic_loss:.4f}  "
                f"duration: {avg_duration_loss:.4f}"
            )
            accelerator.log({
                "train/epoch_loss":          avg_epoch_loss,
                "train/epoch_acoustic_loss": avg_acoustic_loss,
                "train/epoch_duration_loss": avg_duration_loss,
            }, step=global_step)

            if args.use_text_weighting:
                total_samples = len(train_dataloader) * args.batch_size
                pct_weighted = (epoch_high_weight_samples / total_samples) * 100
                accelerator.print(f"📈 Epoch {epoch+1}: {pct_weighted:.1f}% samples had weight > 1.5x")
            
            save_checkpoint(
                model, duration_wrapper, config, target_speaker_embedding, args,
                actual_model_path, f"checkpoint-epoch-{epoch}", accelerator
            )
    
    # Final summary
    if accelerator.is_main_process:
        status = "stopped early" if training_stopped_early else "completed all epochs"
        accelerator.print(f"\n{'='*60}")
        accelerator.print(f"✅ Training {status}")
        accelerator.print(f"   Best validation loss: {best_val_loss:.4f}")
        accelerator.print(f"{'='*60}\n")

    # Flush and close all trackers (TensorBoard, etc.).
    # Without this call the tracker's internal write buffer may not be flushed
    # to disk before the process exits — the last epoch's metrics can be silently
    # lost and never appear in check_progress.py or the TensorBoard UI.
    accelerator.end_training()


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate(model, duration_wrapper, dataloader, accelerator, has_speaker_encoder, target_speaker_embedding, args):
    """Evaluation loop"""
    model.eval()
    duration_wrapper.eval()
    # model == duration_wrapper after the single-prepare fix.  unwrap_model()
    # returns Qwen3TTSWithDuration; .base_model gives the actual Qwen3TTS model
    # that has .talker, .speaker_encoder, etc.
    unwrapped_duration = accelerator.unwrap_model(duration_wrapper)
    unwrapped_model = unwrapped_duration.base_model
    
    total_loss = 0
    total_acoustic_loss = 0
    total_duration_loss = 0
    total_duration_mae = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids']
            codec_ids = batch['codec_ids']
            ref_mels = batch['ref_mels']
            text_embedding_mask = batch['text_embedding_mask']
            codec_embedding_mask = batch['codec_embedding_mask']
            attention_mask = batch['attention_mask']
            codec_0_labels = batch['codec_0_labels']
            codec_mask = batch['codec_mask']

            # Speaker encoder call (already in torch.no_grad() context, so no gradients anyway)
            speaker_embedding = unwrapped_model.speaker_encoder(
                ref_mels.to(unwrapped_model.device).to(unwrapped_model.dtype)
            ) if has_speaker_encoder else target_speaker_embedding

            input_text_ids = input_ids[:, :, 0]
            input_codec_ids = input_ids[:, :, 1]

            input_text_embedding = unwrapped_model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
            input_codec_embedding = unwrapped_model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
            speaker_injection = torch.zeros_like(input_codec_embedding)
            speaker_injection[:, 6, :] = speaker_embedding
            input_codec_embedding = input_codec_embedding + speaker_injection

            input_embeddings = input_text_embedding + input_codec_embedding

            for i in range(1, 16):
                codec_i_embedding = unwrapped_model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                input_embeddings = input_embeddings + codec_i_embedding

            outputs = unwrapped_model.talker(
                inputs_embeds=input_embeddings[:, :-1, :],
                attention_mask=attention_mask[:, :-1],
                labels=codec_0_labels[:, 1:],
                output_hidden_states=True
            )

            hidden_states = outputs.hidden_states[-1]   # [B, T-1, H] (shifted for next-token prediction)
            
            # FIX: Use consistent shifted codec_mask for both extractions to ensure alignment.
            # hidden_states[i] predicts codec_ids[i+1], so both need the shifted mask.
            shifted_codec_mask = codec_mask[:, 1:]      # [B, T-1] aligned with hidden_states
            talker_hidden_states = hidden_states[shifted_codec_mask]  # [N, H]
            talker_codec_ids = codec_ids[:, 1:][shifted_codec_mask]   # [N, 16] - properly aligned

            sub_talker_logits, sub_talker_loss = unwrapped_model.talker.forward_sub_talker_finetune(
                talker_codec_ids, talker_hidden_states
            )

            acoustic_loss = outputs.loss + 0.3 * sub_talker_loss
            
            # Duration
            target_duration = extract_audio_duration(codec_mask, fps=12.5)
            duration_pred, duration_loss = unwrapped_duration(
                hidden_states, 
                target_duration.to(hidden_states.device),
                codec_mask=codec_mask[:, 1:]
            )
            
            duration_mae = (duration_pred - target_duration.to(duration_pred.device)).abs().mean()
            
            total_loss += (acoustic_loss + args.duration_loss_weight * duration_loss).item()
            total_acoustic_loss += acoustic_loss.item()
            total_duration_loss += duration_loss.item()
            total_duration_mae += duration_mae.item()
            num_batches += 1
    
    # Gather metrics from all processes.
    # Each process ran evaluate() on its own shard of the validation set
    # (DistributedSampler splits it 4 ways).  Without a reduce, main process
    # only sees ~25% of validation data — making best-checkpoint selection
    # and early stopping decisions 4× noisier than necessary.
    loss_tensor = torch.tensor(
        [total_loss, total_acoustic_loss, total_duration_loss, total_duration_mae, float(num_batches)],
        device=accelerator.device
    )
    loss_tensor = accelerator.reduce(loss_tensor, reduction='sum')
    n = loss_tensor[4].item()

    return {
        'total_loss':    loss_tensor[0].item() / n,
        'acoustic_loss': loss_tensor[1].item() / n,
        'duration_loss': loss_tensor[2].item() / n,
        'duration_mae':  loss_tensor[3].item() / n,
    }


# ============================================================================
# CHECKPOINT SAVING
# ============================================================================

def save_checkpoint(model, duration_wrapper, config, target_speaker_embedding, args, 
                   model_path, checkpoint_name, accelerator):
    """Save checkpoint including duration head"""
    # After the single-prepare fix, 'model' IS 'duration_wrapper' (an alias).
    # unwrap_model() on either gives Qwen3TTSWithDuration.
    # We need .base_model to reach the actual Qwen3TTS that has .talker / .speaker_encoder.
    unwrapped_duration = accelerator.unwrap_model(duration_wrapper)
    unwrapped_model = unwrapped_duration.base_model
    
    checkpoint_dir = Path(args.output_model_path) / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    state_dict = {}
    
    # Save base model
    for name, param in unwrapped_model.talker.named_parameters():
        state_dict[f"talker.{name}"] = param.cpu()
    
    if hasattr(unwrapped_model, 'speaker_encoder') and unwrapped_model.speaker_encoder is not None:
        for name, param in unwrapped_model.speaker_encoder.named_parameters():
            state_dict[f"speaker_encoder.{name}"] = param.cpu()
    
    # Save duration head
    for name, param in unwrapped_duration.duration_head.named_parameters():
        state_dict[f"duration_head.{name}"] = param.cpu()
    
    # ⚠️  DO NOT "anchor" the speaker embedding into the model weights.
    #    Doing so would break voice_clone (ref_audio) at inference time because
    #    the model would expect a fixed speaker ID instead of a dynamic embedding.
    #
    # ❌  NEVER do this:
    # weight = state_dict['talker.model.codec_embedding.weight']
    # state_dict['talker.model.codec_embedding.weight'][3000] = (
    #     target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
    # )
    #
    # The speaker identity lives in the transformer layers, NOT in a fixed slot.
    # The embedding at position 3000 is only an "anchor" used by custom_voice mode;
    # voice_clone mode reads the embedding from ref_audio at every call.

    save_file(state_dict, checkpoint_dir / "model.safetensors")

    # ⚠️  Keep tts_model_type as "voice_clone" so inference uses ref_audio.
    #    Setting it to "custom_voice" would force the model to look for a fixed
    #    spk_id slot and ignore the reference audio passed at generation time.
    config_dict = config.to_dict()
    if config_dict.get("tts_model_type") == "custom_voice":
        accelerator.print(
            "⚠️  config.tts_model_type was 'custom_voice' — forcing back to 'voice_clone'"
        )
    # ❌  NEVER do this:
    # config_dict["tts_model_type"] = "custom_voice"
    config_dict["tts_model_type"] = "voice_clone"   # ✅ always keep voice_clone
    config.__dict__.update(config_dict)
    config.save_pretrained(checkpoint_dir)
    
    if target_speaker_embedding is not None:
        torch.save(target_speaker_embedding.cpu(), checkpoint_dir / "speaker_embedding.pt")
    
    with open(checkpoint_dir / "training_args.json", 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    accelerator.print(f"✓ Checkpoint saved: {checkpoint_name}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    train()
