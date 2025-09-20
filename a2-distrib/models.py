# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *
import copy


# ---- Prefix Embeddings (k-mer) ----
class PrefixEmbeddings:
    """
    Embeddings over first-k-character prefixes.
    - indexer: Indexer mapping prefix -> id (PAD=0, UNK_PREFIX=1)
    - vectors: np.ndarray [num_prefixes, d]
    """
    def __init__(self, prefix_indexer, vectors: np.ndarray, k: int = 3):
        self.prefix_indexer = prefix_indexer
        self.vectors = vectors
        self.k = k

    def get_initialized_embedding_layer(self, frozen: bool = False, padding_idx: int = 0, sparse: bool = True):
        num_prefixes, d = self.vectors.shape
        emb = nn.Embedding(num_prefixes, d, padding_idx=padding_idx, sparse=sparse)
        emb.weight.data.copy_(torch.from_numpy(self.vectors))
        emb.weight.requires_grad = not frozen
        return emb


    def get_embedding_length(self):
        return self.vectors.shape[1]

    def prefix_of(self, word: str) -> str:
        if len(word) >= self.k:
            return word[:self.k]
        # Very short tokens can just use themselves; you could also route them to UNK
        return word

    def index_of_prefix(self, prefix: str) -> int:
        idx = self.prefix_indexer.index_of(prefix)
        return idx if idx is not None and idx >= 0 else self.prefix_indexer.index_of("UNK_PREFIX")


def build_prefix_embeddings(word_embeddings: WordEmbeddings, k: int = 3) -> PrefixEmbeddings:
    """
    Build PrefixEmbeddings by averaging word vectors for each k-prefix present in the vocab.
    - PAD=0 (zeros), UNK_PREFIX=1 (zeros)
    - Other prefixes initialized as mean of all words starting with that prefix
    """
    # 1) Collect all words in the word vocab (skip PAD/UNK)
    word_indexer = word_embeddings.word_indexer
    V = len(word_indexer)

    # Determine embedding dim
    d = word_embeddings.get_embedding_length()

    # 2) Build prefix indexer with PAD and UNK_PREFIX
    prefix_indexer = Indexer()
    prefix_indexer.add_and_get_index("PAD")         # 0
    prefix_indexer.add_and_get_index("UNK_PREFIX")  # 1

    # 3) First pass: collect sums & counts per prefix
    # Use dict prefix -> (sum_vec, count)
    sums = {}
    counts = {}

    for wid in range(V):
        w = word_indexer.get_object(wid)
        if w in ("PAD", "UNK") or w is None:
            continue
        pref = w[:k] if len(w) >= k else w
        if pref not in sums:
            sums[pref] = np.zeros(d, dtype=np.float32)
            counts[pref] = 0
        sums[pref] += word_embeddings.vectors[wid]   # numpy vector
        counts[pref] += 1

    # 4) Build vectors array: [num_prefixes, d]
    # Start with PAD and UNK rows as zeros
    vectors = []
    vectors.append(np.zeros(d, dtype=np.float32))  # PAD
    vectors.append(np.zeros(d, dtype=np.float32))  # UNK_PREFIX

    # Add prefixes in a stable order
    for pref in sorted(sums.keys()):
        prefix_indexer.add_and_get_index(pref)
        avg = sums[pref] / max(1, counts[pref])
        vectors.append(avg.astype(np.float32))

    vectors = np.stack(vectors, axis=0)  # shape [num_prefixes, d]
    return PrefixEmbeddings(prefix_indexer, vectors, k=k)


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise. If you do
        spelling correction, this parameter allows you to only use your method for the appropriate dev eval in Q3
        and not otherwise
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1

class DANNet(nn.Module):
    def __init__(self, embedding_layer: nn.Embedding, hidden_size: int, out_size: int = 2, dropout_p: float = 0.2):
        super().__init__()
        self.embedding = embedding_layer
        embed_dim = embedding_layer.embedding_dim

        self.ln = nn.LayerNorm(embed_dim)          # <- NEW

        self.fc1 = nn.Linear(embed_dim, hidden_size)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_size, out_size)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.dim() == 1:
            indices = indices.unsqueeze(0)
        E = self.embedding(indices)                 # [B, T, d]
        mask = (indices != 0).unsqueeze(-1).float()
        sum_vec = (E * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1.0)
        avg = sum_vec / lengths                     # [B, d]
        avg = self.ln(avg)                          # <- NEW

        h = self.fc1(avg)
        h = self.act(h)
        h = self.drop(h)
        logits = self.fc2(h)
        return logits


class NeuralSentimentClassifier(SentimentClassifier):
    def __init__(self, word_embeddings: WordEmbeddings, frozen_embeddings=True,
                 hidden_size=200, dropout_p=0.0, device=torch.device("cpu"),
                 use_prefix_embeddings: bool = False, prefix_k: int = 3):
        self.indexer = word_embeddings.word_indexer
        self.device = device
        self.use_prefix = use_prefix_embeddings
        self.prefix_k = prefix_k
    
        if self.use_prefix:
            # ---- Build prefix embeddings from word embeddings (not frozen) ----
            self.prefix_embs = build_prefix_embeddings(word_embeddings, k=self.prefix_k)
            self.indexer = self.prefix_embs.prefix_indexer
            self.embedding = self.prefix_embs.get_initialized_embedding_layer(
                frozen=False,
                padding_idx=self.indexer.index_of("PAD"),
                sparse=True                      # << important
            )
            self.UNK_IDX = self.indexer.index_of("UNK_PREFIX")
        else:
            # ---- Original word embeddings path ----
            self.indexer = word_embeddings.word_indexer
            self.embedding = word_embeddings.get_initialized_embedding_layer(
                frozen=frozen_embeddings,
                padding_idx=self.indexer.index_of("PAD")
            )
            self.UNK_IDX = self.indexer.index_of("UNK")
        embed_dim = self.embedding.embedding_dim
        self.net = DANNet(self.embedding, hidden_size, out_size=2, dropout_p=dropout_p).to(self.device)
        self.embedding.to(self.device)
    
    def _word_to_idx(self, w: str) -> int:
        if self.use_prefix:
            pref = w[:self.prefix_k] if len(w) >= self.prefix_k else w
            idx = self.indexer.index_of(pref)
            return self.UNK_IDX if (idx is None or idx < 0) else idx
        else:
            idx = self.indexer.index_of(w)
            return self.UNK_IDX if (idx is None or idx < 0) else idx

    def _words_to_indices(self, words: List[str]) -> torch.Tensor:
        ids = [self._word_to_idx(w) for w in words]
        if not ids:  # safety
            ids = [self.UNK_IDX]
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def logits_from_words(self, words: List[str]) -> torch.Tensor:
        indices = self._words_to_indices(words)
        return self.net(indices)

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        self.net.eval()
        with torch.no_grad():
            logits = self.logits_from_words(ex_words)
            return int(torch.argmax(logits, dim=-1).item())


    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool, batch_size: int = 64) -> List[int]:
        """
        Batched prediction for speed. 
        - Converts words -> indices
        - Pads to the longest sequence in the batch with PAD=0
        - Runs the network once per batch and argmaxes logits
        
        Assumes:
        - self._word_to_idx maps OOV -> UNK (index 1)
        - PAD index is 0 (embedding was built with padding_idx=0)
        - DANNet.forward(masking) averages only non-PAD tokens
        """
        self.net.eval()
        preds: List[int] = []
        device = self.device

        with torch.no_grad():
            
            for start in range(0, len(all_ex_words), batch_size):
                chunk = all_ex_words[start:start + batch_size]

               
                id_lists = []
                for words in chunk:
                    ids = [self._word_to_idx(w) for w in words]
                    if not ids: 
                        ids = [self.UNK_IDX]
                    id_lists.append(ids)

                T = max(len(x) for x in id_lists)  
                B = len(id_lists)
                padded = torch.zeros(B, T, dtype=torch.long, device=device)  

                for i, ids in enumerate(id_lists):
                    L = len(ids)
                    padded[i, :L] = torch.tensor(ids, dtype=torch.long, device=device)

                logits = self.net(padded)
                batch_preds = torch.argmax(logits, dim=1)
                preds.extend(batch_preds.tolist())


        return preds

def collate_fn_train(ex_list: List[SentimentExample],
                     token_to_idx,               
                     pad_idx: int = 0,
                     max_len: int = None,
                     device: torch.device = torch.device("cpu")):
    seqs, labels = [], []
    for ex in ex_list:
        ids = [token_to_idx(w) for w in ex.words]  
        if not ids:
            ids = [token_to_idx("UNK")]             
        seqs.append(ids)
        labels.append(ex.label)

    T = (max_len if max_len is not None else max(len(s) for s in seqs))
    B = len(seqs)
    indices = torch.full((B, T), fill_value=pad_idx, dtype=torch.long, device=device)
    for i, ids in enumerate(seqs):
        if max_len is not None:
            ids = ids[:max_len]
        L = min(len(ids), T)
        indices[i, :L] = torch.tensor(ids[:L], dtype=torch.long, device=device)

    labels = torch.tensor(labels, dtype=torch.long, device=device)
    return indices, labels





def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    lr = args.lr
    num_epochs = args.num_epochs
    hidden_size = args.hidden_size
    batch_size = max(args.batch_size, 64)
    # at the top of train_deep_averaging_network (once)
    torch.manual_seed(1337)
    np.random.seed(1337)
    random.seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1337)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    if train_model_for_typo_setting:
        use_prefix = True
        frozen_embeddings = False          # trainable prefix embeddings
    else:
        use_prefix = False
        frozen_embeddings = True           # freeze for non-typo path (faster)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clf = NeuralSentimentClassifier(
        word_embeddings=word_embeddings,
        frozen_embeddings=frozen_embeddings,
        hidden_size=256,
        dropout_p=0.2,
        device=device,
        use_prefix_embeddings=use_prefix,
        prefix_k=3
    )

    criterion = nn.CrossEntropyLoss()

    # --- OPTIMIZERS: define ONCE ---
    if train_model_for_typo_setting:
        # SparseAdam only for the sparse embedding; Adam for the dense MLP
        emb_param = [clf.net.embedding.weight]
        other_params = [p for n, p in clf.net.named_parameters() if n != "embedding.weight"]
        opt_emb   = optim.SparseAdam(emb_param, lr=.003)                   # no weight_decay
        opt_other = optim.Adam(other_params, lr=.001, weight_decay=0.0)
    else:
        optimizer = optim.Adam(clf.net.parameters(), lr=.001, weight_decay=0.0)

    best_dev_acc, best_state = -1.0, None
    patience, no_improve = 4, 0

    for epoch in range(1, num_epochs + 1):
        clf.net.train()
        random.shuffle(train_exs)
        total_loss = 0.0

        for start in range(0, len(train_exs), batch_size):
            batch = train_exs[start:start + batch_size]
            batch_indices, batch_labels = collate_fn_train(
                batch,
                token_to_idx=clf._word_to_idx,
                pad_idx=0,
                max_len=60,     # keep as-is for Step 1; cap later for more speed
                device=device
            )

            logits = clf.net(batch_indices)
            loss = criterion(logits, batch_labels)

            if train_model_for_typo_setting:
                opt_other.zero_grad(set_to_none=True)
                opt_emb.zero_grad(set_to_none=True)
                loss.backward()
                # clip only dense params; sparse grads cannot be clipped
                torch.nn.utils.clip_grad_norm_(other_params, 5.0)
                opt_other.step()
                opt_emb.step()
            else:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(clf.net.parameters(), 5.0)
                optimizer.step()

            # inside the batch loop
            total_loss += loss.item() * batch_indices.size(0)

       # --- Dev eval (same path the harness uses) ---
        clf.net.eval()
        with torch.no_grad():
            preds = clf.predict_all([ex.words for ex in dev_exs], has_typos=False, batch_size=256)
            correct = sum(int(p == ex.label) for p, ex in zip(preds, dev_exs))
        dev_acc = correct / max(1, len(dev_exs))
        avg_loss = total_loss / len(train_exs)
        print(f"Epoch {epoch:02d} | train_loss={avg_loss:.4f} | dev_acc={dev_acc:.4f}")




        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_state = copy.deepcopy(clf.net.state_dict())   # <- deep copy the whole thing
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    if best_state is not None:
        clf.net.load_state_dict(best_state)
    clf.net.eval()
    with torch.no_grad():
        preds = clf.predict_all([ex.words for ex in dev_exs], has_typos=False, batch_size=256)
        correct = sum(int(p == ex.label) for p, ex in zip(preds, dev_exs))
    restored_dev_acc = correct / max(1, len(dev_exs))
    print(f"best dev_acc seen: {best_dev_acc:.4f} | restored dev_acc: {restored_dev_acc:.4f}")

    return clf
