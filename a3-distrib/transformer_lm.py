# models.py

import numpy as np
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import PositionalEncoding

class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)

def _oov_to_space_idx(vocab_index, ch: str) -> int:
    """
    Maps unknowns to the index of space ' ' (SOS per assignment).
    utils.Indexer.index_of returns -1 if missing.
    """
    idx = vocab_index.index_of(ch)
    if idx is None or idx < 0:
        idx = vocab_index.index_of(' ')
    return idx


# =========================
# Transformer LM (batch-first)
# =========================
class CharTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 4096,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

        self.embed = nn.Embedding(vocab_size, d_model)
        # Reuse learned positional embeddings from Part 1 (batched=True path)
        self.pos = PositionalEncoding(d_model, num_positions=max_len, batched=True)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,      # expect [B, T, D]
            activation='gelu',
            norm_first = True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size, bias = False)
        self.proj.weight = self.embed.weight
        self.emb_norm = nn.LayerNorm(d_model)
        self._mask_cache = {}  # cache by (T, device)
        

    def forward(self, x_ids: torch.Tensor) -> torch.Tensor:
        """
        x_ids: [B, T] (Long)
        returns logits: [B, T, V]
        """
        B, T = x_ids.shape

        x = self.embed(x_ids) * math.sqrt(self.d_model)
        x = self.emb_norm(x)
        x = self.pos(x)

        # cached boolean causal mask (upper triangle = True -> disallowed)
        # cached float mask: 0 on allowed, -inf on future (upper triangle)
        key = (T, x_ids.device)
        if key not in self._mask_cache:
            m = torch.full((T, T), float('-inf'), device=x_ids.device)
            m = torch.triu(m, diagonal=1)  # upper triangle -inf; diag/lower 0
            self._mask_cache[key] = m
        mask = self._mask_cache[key]


        x = self.encoder(x, mask=mask)   # no is_causal
        x = self.norm(x)
        logits = self.proj(x)                              # [B, T, V]
        return logits

class NeuralLanguageModel(LanguageModel):
    def __init__(self, model: CharTransformerLM, vocab_index, device: torch.device, max_seq_len: int, sos_char: str = ' '):
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

        x = torch.tensor(in_ids, dtype=torch.long, device=self.device).unsqueeze(0)  # [1, T]
        logits = self.model(x)                     # [1, T, V]
        last = logits[0, -1, :]                    # [V]
        log_probs = F.log_softmax(last, dim=-1)
        return log_probs.cpu().numpy()

    @torch.no_grad()
    def get_log_prob_sequence(self, next_chars, context):
        self.model.eval()
        # encode once
        ctx_ids = self._encode(context)
        nxt_ids = self._encode(next_chars)

        total = 0.0
        prefix = ctx_ids[:]  # grow this as we consume next_chars
        for yi in nxt_ids:
            # build input: [SOS] + prefix  (truncate to keep within max_seq_len)
            in_ids = [self.sos_id] + prefix
            if len(in_ids) > self.max_seq_len:
                in_ids = in_ids[-self.max_seq_len:]

            x = torch.tensor(in_ids, dtype=torch.long, device=self.device).unsqueeze(0)  # [1, T]
            logits = self.model(x)                     # [1, T, V]
            last = logits[0, -1, :]                    # [V]
            log_probs = F.log_softmax(last, dim=-1)
            total += float(log_probs[yi].item())

            # teacher-forcing prefix grows with the gold next token
            prefix.append(yi)

        return total

def _stream_to_batches_batchfirst(
    ids: List[int],
    seq_len: int,
    batch_size: int,
    sos_id: int,
    stride: int,
    shuffle: bool = True
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Slice the long stream into non-overlapping chunks of length seq_len.
    For each target chunk y[0:T], build input x = [SOS] + y[0:T-1].
    Returns list of (x_batch, y_batch), each [B, T].
    """
    X, Y = [], []
    N = len(ids)
    last_start = N - seq_len
    if last_start < 0:
        return []  # not enough data; caller should guard/adjust seq_len

    for start in range(0, last_start, stride):
        y_chunk = ids[start:start + seq_len]
        if len(y_chunk) < seq_len:
            break
        x_chunk = [sos_id] + y_chunk[:-1]
        X.append(torch.tensor(x_chunk, dtype=torch.long))
        Y.append(torch.tensor(y_chunk, dtype=torch.long))

    if shuffle:
        idxs = np.random.permutation(len(X)).tolist()
        X = [X[i] for i in idxs]
        Y = [Y[i] for i in idxs]

    batches = []
    for i in range(0, len(X), batch_size):
        xb = torch.stack(X[i:i + batch_size], dim=0)  # [B, T]
        yb = torch.stack(Y[i:i + batch_size], dim=0)  # [B, T]
        batches.append((xb, yb))
    return batches

def _sample_random_batch(
    ids: List[int],
    seq_len: int,
    batch_size: int,
    sos_id: int,
    device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly sample batch_size windows from the stream.
    Returns xb, yb with shape [B, T] on device.
    """
    N = len(ids)
    hi = max(1, N - seq_len)
    starts = np.random.randint(0, hi, size=batch_size)

    xs, ys = [], []
    arr = ids
    for s in starts:
        y = arr[s:s+seq_len]
        if len(y) < seq_len:
            y = y + [arr[-1]] * (seq_len - len(y))
        x = [sos_id] + y[:-1]
        ys.append(torch.tensor(y, dtype=torch.long))
        xs.append(torch.tensor(x, dtype=torch.long))

    xb = torch.stack(xs, 0).to(device)
    yb = torch.stack(ys, 0).to(device)
    return xb, yb

@torch.no_grad()
def _quick_eval_print(model: CharTransformerLM, dev_ids: List[int], sos_id: int, device):
    """
    Lightweight progress print for visibility (avg logp & perplexity).
    lm.py does final checks; this is optional but handy during training.
    """
    if len(dev_ids) < 2:
        return
    x_ids = [sos_id] + dev_ids[:-1]
    x = torch.tensor(x_ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, T]
    y = torch.tensor(dev_ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, T]
    logits = model(x)                                # [1, T, V]
    log_probs = F.log_softmax(logits, dim=-1)[0]     # [T, V]
    gold = y[0]                                      # [T]
    ll = log_probs.gather(-1, gold.unsqueeze(-1)).squeeze(-1).sum().item()
    avg = ll / len(dev_ids)
    ppl = math.exp(-avg)
    print(f"   dev_avg_logp={avg:.4f} | dev_ppl={ppl:.3f}")

def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        try:
            torch.set_float32_matmul_precision('high')  # allow TF32 matmuls
        except Exception:
            pass
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Prefer flash/mem-efficient SDPA if available
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)
        except Exception:
            pass
    V = len(vocab_index)

    # ---- hyperparameters (safe defaults; override with CLI if desired) ----
    d_model      = getattr(args, 'd_model', 288)
    nhead        = getattr(args, 'nhead', 8)
    num_layers   = getattr(args, 'num_layers', 4)
    dim_ff       = getattr(args, 'dim_ff', 1024)
    dropout      = getattr(args, 'dropout', 0.1)
    seq_len      = getattr(args, 'seq_len', 320)
    batch_size   = getattr(args, 'batch_size', 128 if device.type == 'cuda' else 16)
    lr           = getattr(args, 'lr', 1.5e-3)
    weight_decay = getattr(args, 'weight_decay', 0.0)
    max_epochs   = getattr(args, 'max_epochs', 24)
    max_len      = getattr(args, 'max_len', 4096)
    stride       = getattr(args, 'stride', 8)
    steps_per_epoch= getattr(args, 'steps_per_epoch', 600 if device.type == 'cuda' else 400)
    burn_in        = getattr(args, 'burn_in', 64)

    # ---- encode text ----
    train_ids = [_oov_to_space_idx(vocab_index, c) for c in train_text]
    dev_ids   = [_oov_to_space_idx(vocab_index, c) for c in dev_text]
    sos_id    = _oov_to_space_idx(vocab_index, ' ')

    # ---- build model/optim ----
    model = CharTransformerLM(
        vocab_size=V,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_ff,
        dropout=dropout,
        max_len=max_len
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
    criterion = nn.CrossEntropyLoss(ignore_index = -100)

    for epoch in range(1, max_epochs + 1):
        # Rebuild/reshuffle overlapping windows each epoch
        batches = _stream_to_batches_batchfirst(
            train_ids, seq_len=seq_len, batch_size=batch_size,
            sos_id=sos_id, shuffle=True, stride=stride
        )
        if len(batches) == 0:
            seq_len = max(8, min(len(train_ids) - 1, 64))
            batches = _stream_to_batches_batchfirst(
                train_ids, seq_len=seq_len, batch_size=batch_size,
                sos_id=sos_id, shuffle=True, stride=stride
            )

        model.train()
        total_loss = 0.0
        for xb, yb in batches:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)

            if burn_in > 0:
                yb_masked = yb.clone()
                yb_masked[:, :burn_in] = -100
            else:
                yb_masked = yb

            logits = model(xb)  # [B, T, V]
            loss = criterion(logits.view(-1, V), yb_masked.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        print(f"Epoch {epoch:02d} | train_loss={total_loss/len(batches):.4f}")
        _quick_eval_print(model, dev_ids, sos_id, device)

    # ---- wrap in assignment API ----
    return NeuralLanguageModel(model=model, vocab_index=vocab_index, device=device, max_seq_len=max_len, sos_char=' ')

