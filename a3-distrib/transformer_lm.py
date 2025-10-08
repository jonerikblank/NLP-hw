# models.py - Optimized for perplexity < 7 with autograder constraints

import numpy as np
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import PositionalEncoding

# ---------------- Base API ----------------
class LanguageModel(object):
    def get_next_char_log_probs(self, context) -> np.ndarray: 
        raise Exception("Only implemented in subclasses")
    
    def get_log_prob_sequence(self, next_chars, context) -> float: 
        raise Exception("Only implemented in subclasses")

class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size): 
        self.voc_size = voc_size
    
    def get_next_char_log_probs(self, context): 
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)
    
    def get_log_prob_sequence(self, next_chars, context): 
        return np.log(1.0/self.voc_size) * len(next_chars)

def _oov_to_space_idx(vocab_index, ch: str) -> int:
    idx = vocab_index.index_of(ch)
    if idx is None or idx < 0: 
        idx = vocab_index.index_of(' ')
    return idx

# ---------------- Transformer LM ----------------
class CharTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 4096,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=None)
        self.pos = PositionalEncoding(d_model, num_positions=max_len, batched=True)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.emb_norm = nn.LayerNorm(d_model)
        self.in_drop = nn.Dropout(p=dropout)

        self.proj = nn.Linear(d_model, vocab_size, bias=False)
        # Tie weights
        self.proj.weight = self.embed.weight
        self.norm = nn.LayerNorm(d_model)

        self._mask_cache = {}

    def forward(self, x_ids: torch.Tensor) -> torch.Tensor:
        """
        Input: [B, T] token IDs
        Output: [B, T, V] logits
        """
        B, T = x_ids.shape
        x = self.embed(x_ids) * math.sqrt(self.d_model)  # [B, T, d_model]
        x = self.pos(x)  # [B, T, d_model]
        x = self.emb_norm(x)
        x = self.in_drop(x)

        # Causal mask - prevent attending to future tokens
        key = (T, x_ids.device)
        if key not in self._mask_cache:
            self._mask_cache[key] = torch.triu(
                torch.ones((T, T), dtype=torch.bool, device=x_ids.device), 
                diagonal=1
            )
        mask = self._mask_cache[key]

        x = self.encoder(x, mask=mask)  # [B, T, d_model]
        x = self.norm(x)
        logits = self.proj(x)  # [B, T, V]
        return logits

# ---------------- Neural Language Model API ----------------
class NeuralLanguageModel(LanguageModel):
    def __init__(
        self, 
        model: CharTransformerLM, 
        vocab_index, 
        device: torch.device, 
        max_seq_len: int, 
        sos_char: str = ' '
    ):
        self.model = model.to(device)
        self.vocab_index = vocab_index
        self.device = device
        self.max_seq_len = max_seq_len
        self.sos_id = _oov_to_space_idx(vocab_index, sos_char)
        self.V = len(vocab_index)
    
    def _encode(self, text: str) -> List[int]:
        return [_oov_to_space_idx(self.vocab_index, c) for c in text]

    @torch.no_grad()
    def get_next_char_log_probs(self, context):
        self.model.eval()
        ids = self._encode(context)
        in_ids = [self.sos_id] + ids
        if len(in_ids) > self.max_seq_len:
            in_ids = in_ids[-self.max_seq_len:]
        x = torch.tensor(in_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        logits = self.model(x)
        last = logits[0, -1, :]
        log_probs = F.log_softmax(last, dim=-1)
        return log_probs.cpu().numpy()

    @torch.no_grad()
    def get_log_prob_sequence(self, next_chars, context):
        self.model.eval()
        ctx_ids = self._encode(context)
        nxt_ids = self._encode(next_chars)
        total = 0.0
        prefix = ctx_ids[:]
        for yi in nxt_ids:
            in_ids = [self.sos_id] + prefix
            if len(in_ids) > self.max_seq_len:
                in_ids = in_ids[-self.max_seq_len:]
            x = torch.tensor(in_ids, dtype=torch.long, device=self.device).unsqueeze(0)
            logits = self.model(x)
            last = logits[0, -1, :]
            log_probs = F.log_softmax(last, dim=-1)
            total += float(log_probs[yi].item())
            prefix.append(yi)
        return total

# ---------------- Training Utilities ----------------
def stream_with_left_context(
    ids: List[int],
    tgt_len: int,
    batch_size: int,
    sos_id: int,
    stride: int,
    shuffle: bool = True
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Creates training batches with proper input-target alignment.
    
    For a chunk of length tgt_len:
    - Input: [SOS, tok0, tok1, ..., tok(n-2)]  (length = tgt_len)
    - Target: [tok0, tok1, tok2, ..., tok(n-1)] (length = tgt_len)
    
    This ensures model predicts next token at each position.
    """
    X, Y = [], []
    N = len(ids)
    
    # Create overlapping chunks
    num_chunks = 0
    for start in range(0, N - tgt_len + 1, stride):
        chunk = ids[start:start + tgt_len]
        if len(chunk) != tgt_len:
            continue
        
        # CRITICAL: Input is [SOS] + chunk[:-1], target is chunk
        # This means position i in input predicts position i in target
        x = [sos_id] + chunk[:-1]  # length = tgt_len
        y = chunk                   # length = tgt_len
        
        X.append(torch.tensor(x, dtype=torch.long))
        Y.append(torch.tensor(y, dtype=torch.long))
        num_chunks += 1
    
    if shuffle:
        perm = np.random.permutation(len(X))
        X = [X[i] for i in perm]
        Y = [Y[i] for i in perm]
    
    # Create batches
    batches = []
    for i in range(0, len(X), batch_size):
        batch_x = X[i:i+batch_size]
        batch_y = Y[i:i+batch_size]
        if len(batch_x) > 0:
            xb = torch.stack(batch_x, 0)
            yb = torch.stack(batch_y, 0)
            batches.append((xb, yb))
    
    return batches

def _make_optimizer(model, lr, weight_decay):
    """Create optimizer with proper weight decay groups."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: 
            continue
        if n.endswith(".bias") or "norm" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr, betas=(0.9, 0.98), eps=1e-8
    )

# ---------------- Training Function ----------------
def train_lm(args, train_text, dev_text, vocab_index):
    """Train a character-level transformer language model."""
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cuda':
        try: 
            torch.set_float32_matmul_precision('high')
        except Exception: 
            pass
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    V = len(vocab_index)
    
    # Hyperparameters - best config that got 7.90 (memory-safe for autograder)
    d_model      = getattr(args, 'd_model', 192)
    nhead        = getattr(args, 'nhead', 6)
    num_layers   = getattr(args, 'num_layers', 6)
    dim_ff       = getattr(args, 'dim_ff', 768)
    dropout      = getattr(args, 'dropout', 0.15)
    seq_len      = getattr(args, 'seq_len', 128)
    batch_size   = getattr(args, 'batch_size', 64)
    lr           = getattr(args, 'lr', 1e-3)
    weight_decay = getattr(args, 'weight_decay', 0.1)
    max_epochs   = getattr(args, 'max_epochs', 80)
    max_len      = getattr(args, 'max_len', 1024)
    stride       = getattr(args, 'stride', 32)

    # Encode data
    train_ids = [_oov_to_space_idx(vocab_index, c) for c in train_text]
    dev_ids   = [_oov_to_space_idx(vocab_index, c) for c in dev_text]
    sos_id    = _oov_to_space_idx(vocab_index, ' ')

    print(f"\n{'='*60}")
    print(f"Training Configuration")
    print(f"{'='*60}")
    print(f"Data: {len(train_ids)} train chars, {len(dev_ids)} dev chars")
    print(f"Model: d_model={d_model}, nhead={nhead}, layers={num_layers}")
    print(f"       dim_ff={dim_ff}, dropout={dropout}")
    print(f"Training: seq_len={seq_len}, batch_size={batch_size}")
    print(f"          lr={lr}, weight_decay={weight_decay}")
    print(f"          stride={stride}, max_epochs={max_epochs}")
    print(f"Device: {device}")

    # Model
    model = CharTransformerLM(
        vocab_size=V, 
        d_model=d_model, 
        nhead=nhead, 
        num_layers=num_layers,
        dim_feedforward=dim_ff, 
        dropout=dropout, 
        max_len=max_len
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    optimizer = _make_optimizer(model, lr, weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # Create initial batches to determine schedule
    first_batches = stream_with_left_context(
        train_ids, tgt_len=seq_len,
        batch_size=batch_size, sos_id=sos_id, stride=stride, shuffle=True
    )
    steps_per_epoch = len(first_batches)
    total_updates = steps_per_epoch * max_epochs
    warmup = min(400, total_updates // 20)

    print(f"Batches per epoch: {steps_per_epoch}")
    print(f"Total updates: {total_updates}, warmup steps: {warmup}")
    print(f"{'='*60}\n")

    # Learning rate scheduler
    def lr_lambda(step):
        if step < warmup: 
            return step / float(max(1, warmup))
        progress = (step - warmup) / float(max(1, total_updates - warmup))
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_ppl = float('inf')
    best_state = None
    patience = 15
    bad_epochs = 0

    # Training loop
    for epoch in range(1, max_epochs + 1):
        # Create fresh batches each epoch (except first)
        if epoch > 1:
            batches = stream_with_left_context(
                train_ids, tgt_len=seq_len,
                batch_size=batch_size, sos_id=sos_id, stride=stride, shuffle=True
            )
        else:
            batches = first_batches
        
        # Training phase
        model.train()
        total_loss = 0.0
        for xb, yb in batches:
            xb = xb.to(device)  # [B, seq_len]
            yb = yb.to(device)  # [B, seq_len]
            
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)  # [B, seq_len, V]
            
            # Compute loss over all positions
            loss = criterion(logits.reshape(-1, V), yb.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(batches)
        
        # Evaluation phase
        model.eval()
        with torch.no_grad():
            # Prepare dev data: input = [SOS] + dev[:-1], target = dev
            x_ids = [sos_id] + dev_ids[:-1]
            x = torch.tensor(x_ids, dtype=torch.long, device=device).unsqueeze(0)
            y = torch.tensor(dev_ids, dtype=torch.long, device=device).unsqueeze(0)
            
            logits = model(x)  # [1, T, V]
            log_probs = F.log_softmax(logits, dim=-1)[0]  # [T, V]
            
            # Get log probability of gold tokens
            ll = log_probs.gather(-1, y[0].unsqueeze(-1)).squeeze(-1).sum().item()
        
        avg_logp = ll / len(dev_ids)
        ppl = math.exp(-avg_logp)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch:2d} | loss={avg_train_loss:.4f} | ppl={ppl:6.3f} | lr={current_lr:.6f}", end="")
        
        # Track best model
        if ppl < best_ppl:
            best_ppl = ppl
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f" *** BEST ***")
        else:
            bad_epochs += 1
            print()
            if bad_epochs >= patience:
                print(f"\nEarly stopping after {patience} epochs without improvement")
                break

    # Restore best model
    if best_state is not None:
        print(f"\nRestoring best model (perplexity: {best_ppl:.3f})")
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()}, strict=True)

    return NeuralLanguageModel(
        model=model, 
        vocab_index=vocab_index, 
        device=device, 
        max_seq_len=max_len, 
        sos_char=' '
    )