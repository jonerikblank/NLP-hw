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
# ---------------- Simple Sinusoidal Positional Encoding ----------------
class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding - no learning required!"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, moves with model to device)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        Returns:
            Tensor of shape [batch_size, seq_len, d_model]
        """
        # x.shape[1] is seq_len
        return x + self.pe[:x.shape[1], :].unsqueeze(0)


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
        use_pos_encoding: bool = True,  # Default to TRUE now!
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.use_pos_encoding = use_pos_encoding

        # Simple embedding layer
        self.embed = nn.Embedding(vocab_size, d_model)
        
        # Use our simple sinusoidal positional encoding
        if use_pos_encoding:
            self.pos = SinusoidalPositionalEncoding(d_model, max_len)
        else:
            self.pos = None

        # Use PyTorch's built-in TransformerEncoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=False  # Standard post-norm
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        
        # Output projection
        self.output = nn.Linear(d_model, vocab_size)
        
        # Tie input and output embeddings (standard practice)
        self.output.weight = self.embed.weight
        
        self._mask_cache = {}

    def forward(self, x_ids: torch.Tensor) -> torch.Tensor:
        """
        Input: [B, T] token IDs
        Output: [B, T, V] logits
        """
        B, T = x_ids.shape
        
        # Embedding
        x = self.embed(x_ids)  # [B, T, d_model]
        
        # Scale embeddings (standard practice for transformers)
        x = x * math.sqrt(self.d_model)
        
        # Add positional encoding if enabled
        if self.use_pos_encoding and self.pos is not None:
            x = self.pos(x)  # [B, T, d_model]
        
        # Create causal mask
        key = (T, x_ids.device)
        if key not in self._mask_cache:
            # PyTorch convention: True = masked out
            self._mask_cache[key] = torch.triu(
                torch.ones((T, T), dtype=torch.bool, device=x_ids.device), 
                diagonal=1
            )
        mask = self._mask_cache[key]
        
        # Pass through transformer
        x = self.encoder(x, mask=mask, is_causal=True)  # [B, T, d_model]
        
        # Project to vocabulary
        logits = self.output(x)  # [B, T, V]
        
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
        
        # Truncate if needed
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
    for start in range(0, N - tgt_len + 1, stride):
        chunk = ids[start:start + tgt_len]
        if len(chunk) != tgt_len:
            continue
        
        # Input is [SOS] + chunk[:-1], target is chunk
        x = [sos_id] + chunk[:-1]
        y = chunk
        
        X.append(torch.tensor(x, dtype=torch.long))
        Y.append(torch.tensor(y, dtype=torch.long))
    
    # Use PyTorch-based shuffling
    if shuffle and len(X) > 0:
        indices = torch.randperm(len(X))
        X = [X[i] for i in indices]
        Y = [Y[i] for i in indices]
    
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
    
    # Hyperparameters - Clean implementation with sinusoidal encoding
    d_model      = getattr(args, 'd_model', 64)
    nhead        = getattr(args, 'nhead', 8)
    num_layers   = getattr(args, 'num_layers', 4)
    dim_ff       = getattr(args, 'dim_ff', 256)
    dropout      = getattr(args, 'dropout', 0.1)
    seq_len      = getattr(args, 'seq_len', 40)
    batch_size   = getattr(args, 'batch_size', 64)
    lr           = getattr(args, 'lr', 5e-3)
    weight_decay = getattr(args, 'weight_decay', 0.01)
    label_smooth = getattr(args, 'label_smoothing', 0.0)
    max_epochs   = getattr(args, 'max_epochs', 50)
    max_len      = getattr(args, 'max_len', 2048)
    stride       = getattr(args, 'stride', 20)
    patience     = getattr(args, 'patience', 15)
    use_warmup   = getattr(args, 'use_warmup', False)
    use_pos_enc  = getattr(args, 'use_pos_encoding', True)  # Enable sinusoidal!
    use_warmup   = getattr(args, 'use_warmup', False)  # Warmup disabled by default

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
    print(f"          lr={lr} (CONSTANT), weight_decay={weight_decay}")
    print(f"          warmup={'enabled' if use_warmup else 'disabled'}")
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
        max_len=max_len,
        use_pos_encoding=use_pos_enc  # Pass the flag!
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    optimizer = _make_optimizer(model, lr, weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Create initial batches to determine schedule
    first_batches = stream_with_left_context(
        train_ids, tgt_len=seq_len,
        batch_size=batch_size, sos_id=sos_id, stride=stride, shuffle=True
    )
    steps_per_epoch = len(first_batches)
    total_updates = steps_per_epoch * max_epochs
    warmup = steps_per_epoch if use_warmup else 0

    print(f"Batches per epoch: {steps_per_epoch}")
    print(f"Total updates: {total_updates}, warmup steps: {warmup}")
    
    # SANITY CHECK: Print a few input/output examples
    print(f"\n{'='*60}")
    print("SANITY CHECK - First 3 training examples:")
    print(f"{'='*60}")
    for i in range(min(3, len(first_batches))):
        xb, yb = first_batches[i]
        for j in range(min(2, len(xb))):
            x_ids = xb[j].tolist()
            y_ids = yb[j].tolist()
            x_chars = ''.join([vocab_index.get_object(idx) if idx < V else '?' for idx in x_ids])
            y_chars = ''.join([vocab_index.get_object(idx) if idx < V else '?' for idx in y_ids])
            print(f"Example {i*2+j+1}:")
            print(f"  Input:  [{' '.join(str(id) for id in x_ids[:10])}...] = '{x_chars[:30]}...'")
            print(f"  Target: [{' '.join(str(id) for id in y_ids[:10])}...] = '{y_chars[:30]}...'")
    print(f"{'='*60}\n")

    # Constant LR scheduler (with optional warmup)
    if use_warmup and warmup > 0:
        def lr_lambda(step):
            if step < warmup: 
                return step / float(max(1, warmup))
            return 1.0  # Constant after warmup
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        # No scheduler - completely constant LR
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)

    best_ppl = float('inf')
    best_state = None
    patience_counter = 0

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
        
        # Evaluation phase - matches print_evaluation() format
        temp_lm = NeuralLanguageModel(
            model=model, 
            vocab_index=vocab_index, 
            device=device, 
            max_seq_len=max_len, 
            sos_char=' '
        )
        
        # Get total log probability
        log_prob = temp_lm.get_log_prob_sequence(dev_text, "")
        
        # Get last character's log probability as sanity check
        if len(dev_text) > 0:
            context = dev_text[:-1] if len(dev_text) > 1 else ""
            last_char = dev_text[-1]
            last_char_log_probs = temp_lm.get_next_char_log_probs(context)
            last_char_idx = _oov_to_space_idx(vocab_index, last_char)
            last_char_loss = float(last_char_log_probs[last_char_idx])
        else:
            last_char_loss = 0.0
        
        # Calculate perplexity
        ppl = np.exp(-log_prob / len(dev_text))
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch #{epoch}, Loss: {log_prob:.3f}, last char loss: {last_char_loss}; Perplexity: {ppl:.2f}", end="")
        
        # Track best model with early stopping
        if ppl < best_ppl:
            best_ppl = ppl
            patience_counter = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f" *** BEST ***")
        else:
            patience_counter += 1
            print()
            if patience_counter >= patience:
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