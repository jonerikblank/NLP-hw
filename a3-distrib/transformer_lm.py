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
def _generate_causal_mask(seq_len: int, device) -> torch.Tensor:
    """
    mask[i, j] = -inf if j > i else 0; shape [T, T]
    """
    mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
    return torch.triu(mask, diagonal=1)  # upper triangle is -inf; diag & lower are 0


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
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x_ids: torch.Tensor) -> torch.Tensor:
        """
        x_ids: [B, T] (Long)
        returns logits: [B, T, V]
        """
        B, T = x_ids.shape
        device = x_ids.device
        src_mask = _generate_causal_mask(T, device)   # [T, T]

        x = self.embed(x_ids) * math.sqrt(self.d_model)  # [B, T, D]
        x = self.pos(x)                                   # learned positions
        x = self.encoder(x, mask=src_mask)                # [B, T, D]
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
        ctx_ids = self._encode(context)
        nxt_ids = self._encode(next_chars)
        targets = ctx_ids + nxt_ids

        inputs = [self.sos_id] + targets[:-1]

        # Truncate tail if too long; keep input/target aligned
        if len(inputs) > self.max_seq_len:
            inputs  = inputs[-self.max_seq_len:]
            targets = targets[-self.max_seq_len:]

        x = torch.tensor(inputs,  dtype=torch.long, device=self.device).unsqueeze(0)  # [1, T]
        y = torch.tensor(targets, dtype=torch.long, device=self.device).unsqueeze(0)  # [1, T]

        logits = self.model(x)                 # [1, T, V]
        log_probs = F.log_softmax(logits, dim=-1)  # [1, T, V]

        # Gather log p of the gold tokens
        gold = y  # [1, T]
        selected = log_probs.gather(-1, gold.unsqueeze(-1)).squeeze(-1)  # [1, T]

        # Sum only over next_chars positions
        ctx_len = len(ctx_ids)
        return float(selected[0, ctx_len:].sum().item())

def _stream_to_batches_batchfirst(
    ids: List[int],
    seq_len: int,
    batch_size: int,
    sos_id: int,
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

    for start in range(0, last_start, seq_len):
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
    V = len(vocab_index)

    # ---- hyperparameters (safe defaults; override with CLI if desired) ----
    d_model      = getattr(args, 'd_model', 256)
    nhead        = getattr(args, 'nhead', 8)
    num_layers   = getattr(args, 'num_layers', 3)
    dim_ff       = getattr(args, 'dim_ff', 512)
    dropout      = getattr(args, 'dropout', 0.1)
    seq_len      = getattr(args, 'seq_len', 256)
    batch_size   = getattr(args, 'batch_size', 64 if device.type == 'cuda' else 16)
    lr           = getattr(args, 'lr', 1e-3)
    weight_decay = getattr(args, 'weight_decay', 0.01)
    max_epochs   = getattr(args, 'max_epochs', 3)
    max_len      = getattr(args, 'max_len', 4096)

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
    criterion = nn.CrossEntropyLoss()

    # ---- make batches ----
    batches = _stream_to_batches_batchfirst(train_ids, seq_len=seq_len, batch_size=batch_size, sos_id=sos_id, shuffle=True)
    if len(batches) == 0:
        # Fallback: shrink seq_len if train is tiny (shouldn't happen in the assignment)
        seq_len = max(8, min(len(train_ids) - 1, 64))
        batches = _stream_to_batches_batchfirst(train_ids, seq_len=seq_len, batch_size=batch_size, sos_id=sos_id, shuffle=True)

    # ---- train ----
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in batches:
            xb = xb.to(device)  # [B, T]
            yb = yb.to(device)  # [B, T]

            optimizer.zero_grad()
            logits = model(xb)                     # [B, T, V]
            loss = criterion(logits.view(-1, V), yb.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch:02d} | train_loss={total_loss/len(batches):.4f}")
        _quick_eval_print(model, dev_ids, sos_id, device)

    # ---- wrap in assignment API ----
    return NeuralLanguageModel(model=model, vocab_index=vocab_index, device=device, max_seq_len=max_len, sos_char=' ')
