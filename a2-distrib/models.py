# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *


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
    """
    Deep Averaging Network: takes word indices, looks up embeddings,
    averages them, then applies an MLP for classification.
    """
    def __init__(self, embedding_layer: nn.Embedding, hidden_size: int, out_size: int = 2, dropout_p: float = 0.0):
        super().__init__()
        self.embedding = embedding_layer
        embed_dim = embedding_layer.embedding_dim

        self.fc1 = nn.Linear(embed_dim, hidden_size)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_size, out_size)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        indices: tensor of shape [n_tokens] (or [batch, n_tokens] if you add batching later)
        returns: logits [2] (or [batch, 2])
        """
        E = self.embedding(indices)   # [n_tokens, d]
        avg = E.mean(dim=0)           # [d]
        h = self.fc1(avg)
        h = self.act(h)
        h = self.drop(h)
        return self.fc2(h)            # [2]

class NeuralSentimentClassifier(SentimentClassifier):
    def __init__(self, word_embeddings: WordEmbeddings, frozen_embeddings=True,
                 hidden_size=200, dropout_p=0.0, device=torch.device("cpu")):
        self.word_embeddings = word_embeddings
        self.indexer = word_embeddings.word_indexer
        self.device = device

        embedding_layer = word_embeddings.get_initialized_embedding_layer(
          frozen=frozen_embeddings,
          padding_idx=0   # PAD is index 0 per the assignment; this keeps PAD as the zero vector
          )

        self.net = DANNet(embedding_layer, hidden_size, out_size=2, dropout_p=dropout_p).to(self.device)

        self.UNK_IDX = self.indexer.index_of("UNK")


    def _words_to_indices(self, words: List[str]) -> torch.Tensor:
        ids = [self.indexer.index_of(w) or self.UNK_IDX for w in words]
        if not ids:
            ids = [self.UNK_IDX]
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def logits_from_words(self, words: List[str]) -> torch.Tensor:
        indices = self._words_to_indices(words)
        return self.net(indices)

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        self.net.eval()
        with torch.no_grad():
            logits = self.logits_from_words(ex_words)
            return int(torch.argmax(logits).item())




def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :param train_model_for_typo_setting: True if we should train the model for the typo setting, False otherwise
    :return: A trained NeuralSentimentClassifier model. Note: you can create an additional subclass of SentimentClassifier
    and return an instance of that for the typo setting if you want; you're allowed to return two different model types
    for the two settings.
    """
    lr = args.lr
    num_epochs = args.num_epochs
    hidden_size = args.hidden_size

    frozen_embeddings = not train_model_for_typo_setting

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clf = NeuralSentimentClassifier(
        word_embeddings=word_embeddings,
        frozen_embeddings=frozen_embeddings,
        hidden_size=hidden_size,
        dropout_p=0.0,   # add a --dropout arg later if you want
        device=device
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(clf.net.parameters(), lr=lr)

    best_dev_acc = -1.0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        clf.net.train()
        random.shuffle(train_exs)
        total_loss = 0.0

        for ex in train_exs:
            # words -> indices -> logits
            logits = clf.logits_from_words(ex.words)  # <--- clean API
            gold = torch.tensor(ex.label, dtype=torch.long, device=device)

            loss = criterion(logits, gold)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Dev eval
        clf.net.eval()
        correct = 0
        with torch.no_grad():
            for ex in dev_exs:
                pred = clf.predict(ex.words, has_typos=False)
                correct += int(pred == ex.label)
        dev_acc = correct / max(1, len(dev_exs))

        print(f"Epoch {epoch:02d} | train_loss={total_loss/len(train_exs):.4f} | dev_acc={dev_acc:.4f}")

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_state = {k: v.detach().cpu() for k, v in clf.net.state_dict().items()}

    if best_state is not None:
        clf.net.load_state_dict(best_state)

    return clf
