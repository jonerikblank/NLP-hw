# models.py

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
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
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
        self.in_drop = nn.Dropout(p=0.2)

        self.proj = nn.Linear(d_model, vocab_size, bias=False)
        # Tie weights
        self.proj.weight = self.embed.weight
        self.norm = nn.LayerNorm(d_model)

        self._mask_cache = {}

    def forward(self, x_ids: torch.Tensor) -> torch.Tensor:
        B, T = x_ids.shape
        x = self.embed(x_ids) * math.sqrt(self.d_model)
        x = self.pos(x)
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

        x = self.encoder(x, mask=mask)
        x = self.norm(x)
        logits = self.proj(x)
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
    left_ctx: int,
    batch_size: int,
    sos_id: int,
    stride: int,
    shuffle: bool = True
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    X, Y = [], []
    N = len(ids)
    last_start = N - tgt_len
    for start in range(0, last_start, stride):
        y = ids[start:start + tgt_len]
        ctx_start = max(0, start - left_ctx)
        x_core = ids[ctx_start:start + tgt_len - 1]
        need = left_ctx + tgt_len - 1 - len(x_core)
        if need > 0: 
            x_core = [sos_id] * need + x_core
        x = [sos_id] + x_core
        X.append(torch.tensor(x, dtype=torch.long))
        Y.append(torch.tensor(y, dtype=torch.long))
    
    if shuffle:
        perm = np.random.permutation(len(X))
        X = [X[i] for i in perm]
        Y = [Y[i] for i in perm]
    
    batches = []
    for i in range(0, len(X), batch_size):
        xb = torch.stack(X[i:i+batch_size], 0)
        yb = torch.stack(Y[i:i+batch_size], 0)
        batches.append((xb, yb))
    return batches

@torch.no_grad()
def _quick_eval_print(model, dev_ids, sos_id, device):
    if len(dev_ids) < 2: 
        return
    was_training = model.training
    model.eval()
    x_ids = [sos_id] + dev_ids[:-1]
    x = torch.tensor(x_ids, dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(dev_ids, dtype=torch.long, device=device).unsqueeze(0)
    logits = model(x)
    log_probs = F.log_softmax(logits, dim=-1)[0]
    gold = y[0]
    ll = log_probs.gather(-1, gold.unsqueeze(-1)).squeeze(-1).sum().item()
    avg = ll / len(dev_ids)
    print(f"   dev_avg_logp={avg:.4f} | dev_ppl={math.exp(-avg):.3f}")
    if was_training: 
        model.train()

def _make_optimizer(model, lr, weight_decay):
    # No weight decay for bias and LayerNorm weights
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: 
            continue
        if n.endswith(".bias") or "norm" in n or "emb_norm" in n:
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr, betas=(0.9, 0.98)
    )

# ---------------- Training Function ----------------
def train_lm(args, train_text, dev_text, vocab_index):
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
    
    # Hyperparameters - back to basics, match reference approach
    d_model      = getattr(args, 'd_model', 256)
    nhead        = getattr(args, 'nhead', 8)
    num_layers   = getattr(args, 'num_layers', 4)
    dim_ff       = getattr(args, 'dim_ff', 1024)
    dropout      = getattr(args, 'dropout', 0.1)
    seq_len      = getattr(args, 'seq_len', 128)
    batch_size   = getattr(args, 'batch_size', 128 if device.type == 'cuda' else 64)
    lr           = getattr(args, 'lr', 3e-3)
    weight_decay = getattr(args, 'weight_decay', 0.01)
    max_epochs   = getattr(args, 'max_epochs', 30)
    max_len      = getattr(args, 'max_len', 4096)
    stride       = getattr(args, 'stride', 1)  # Dense overlapping
    left_ctx     = getattr(args, 'left_ctx', 20)  # Small context window

    # Encode data
    train_ids = [_oov_to_space_idx(vocab_index, c) for c in train_text]
    dev_ids   = [_oov_to_space_idx(vocab_index, c) for c in dev_text]
    sos_id    = _oov_to_space_idx(vocab_index, ' ')

    # Model and optimizer
    model = CharTransformerLM(
        vocab_size=V, 
        d_model=d_model, 
        nhead=nhead, 
        num_layers=num_layers,
        dim_feedforward=dim_ff, 
        dropout=dropout, 
        max_len=max_len
    ).to(device)
    
    optimizer = _make_optimizer(model, lr, weight_decay)
    criterion = nn.CrossEntropyLoss()  # No smoothing

    # Build batches and scheduler
    first_batches = stream_with_left_context(
        train_ids, tgt_len=seq_len, left_ctx=left_ctx,
        batch_size=batch_size, sos_id=sos_id, stride=stride, shuffle=True
    )
    steps_per_epoch = max(1, len(first_batches))
    total_updates = max(1, steps_per_epoch * max_epochs)
    warmup = max(200, total_updates // 20)

    def lr_lambda(step):
        if step < warmup: 
            return step / float(max(1, warmup))
        progress = (step - warmup) / float(max(1, total_updates - warmup))
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_ppl = float('inf')
    best_state = None
    patience = 6
    bad = 0

    # Training loop
    for epoch in range(1, max_epochs + 1):
        batches = first_batches if epoch == 1 else stream_with_left_context(
            train_ids, tgt_len=seq_len, left_ctx=left_ctx,
            batch_size=batch_size, sos_id=sos_id, stride=stride, shuffle=True
        )
        
        # Training
        model.train()
        total_loss = 0.0
        for xb, yb in batches:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            logits_tgt = logits[:, -yb.shape[1]:, :]
            loss = criterion(logits_tgt.reshape(-1, V), yb.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        print(f"Epoch {epoch:02d} | train_loss={total_loss/len(batches):.4f}")
        
        # Quick eval
        _quick_eval_print(model, dev_ids, sos_id, device)

        # Full dev evaluation
        model.eval()
        with torch.no_grad():
            x_ids = [sos_id] + dev_ids[:-1]
            x = torch.tensor(x_ids, dtype=torch.long, device=device).unsqueeze(0)
            y = torch.tensor(dev_ids, dtype=torch.long, device=device).unsqueeze(0)
            logits = model(x)
            lp = F.log_softmax(logits, dim=-1)[0]
            ll = lp.gather(-1, y[0].unsqueeze(-1)).squeeze(-1).sum().item()
        
        avg = ll / len(dev_ids)
        ppl = math.exp(-avg)
        print(f"   dev_ppl={ppl:.3f} | lr={optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if ppl < best_ppl:
            best_ppl = ppl
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"   *** New best: {best_ppl:.3f}")
        else:
            bad += 1
            if bad > patience: 
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()}, strict=True)

    return NeuralLanguageModel(
        model=model, 
        vocab_index=vocab_index, 
        device=device, 
        max_seq_len=max_len, 
        sos_char=' '
    )