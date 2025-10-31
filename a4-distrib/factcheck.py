# factcheck.py

import torch
import torch.nn.functional as F
from typing import List
import numpy as np
import spacy
import gc
import re
from nltk.corpus import stopwords
from heapq import nlargest


class FactExample(object):
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text". "title" denotes the title of the
    Wikipedia page it was taken from; you generally don't need to use this. "text" is a chunk of text, which may or
    may not align with sensible paragraph or sentence boundaries
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant. Note that we will ignore the Irrelevant
    label for prediction, so your model should just predict S or NS, but we leave it here so you can look at the
    raw data.
    """
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))

class EntailmentModel(object):
    def __init__(self, model, tokenizer, cuda=False):
        self.model = model
        self.tokenizer = tokenizer
        self.cuda = cuda

    def check_entailment(self, premise: str, hypothesis: str):
        """Return (p_ent, p_neu, p_contra) for a single pair."""
        with torch.no_grad():
            inputs = self.tokenizer(
                premise, hypothesis,
                return_tensors='pt',
                truncation=True, padding=True,
                max_length=192  # tighter for speed+memory
            )
            if self.cuda:
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1).squeeze(0)
            p_ent, p_neu, p_contra = float(probs[0]), float(probs[1]), float(probs[2])
        del inputs, outputs, logits, probs
        if self.cuda:
            torch.cuda.empty_cache()
        gc.collect()
        return p_ent, p_neu, p_contra

    def check_entailment_batch(self, premises: list, hypothesis: str, batch_size: int = 16):
        """
        Batched inference: returns list of (p_ent, p_neu, p_contra) aligned to 'premises'.
        Much faster than per-sentence calls.
        """
        results = []
        with torch.no_grad():
            for i in range(0, len(premises), batch_size):
                chunk = premises[i:i+batch_size]
                inputs = self.tokenizer(
                    chunk, [hypothesis]*len(chunk),
                    return_tensors='pt',
                    truncation=True, padding=True,
                    max_length=192
                )
                if self.cuda:
                    inputs = {k: v.to('cuda') for k, v in inputs.items()}
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)  # [B,3]
                probs = probs.detach().cpu()
                for row in probs:
                    results.append((float(row[0]), float(row[1]), float(row[2])))
                del inputs, outputs, probs
                if self.cuda:
                    torch.cuda.empty_cache()
                gc.collect()
        return results


class FactChecker(object):
    """
    Fact checker base type
    """

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence
        :param fact: same as FactExample
        :param passages: same as FactExample
        :return: "S" (supported) or "NS" (not supported)
        """
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


class WordRecallThresholdFactChecker(FactChecker):
    def __init__(self, threshold: float = 0.65, remove_stopwords: bool = True):
        self.threshold = threshold
        self.remove_stopwords = remove_stopwords
        self.word_re = re.compile(r"\w+")
        # Initialize stopword list (you may need: nltk.download('stopwords'))
        self.stopwords = set(stopwords.words('english')) if remove_stopwords else set()

    def _tokens(self, text: str):
        toks = [t.lower() for t in self.word_re.findall(text)]
        if self.remove_stopwords:
            toks = [t for t in toks if t not in self.stopwords]
        return set(toks)

    def _fact_recall(self, fact_tokens, passage_tokens):
        if not fact_tokens:
            return 0.0
        overlap = len(fact_tokens & passage_tokens)
        return overlap / len(fact_tokens)

    def predict(self, fact: str, passages: List[dict]) -> str:
        fact_tokens = self._tokens(fact)
        best_recall = 0.0
        for p in passages:
            passage_tokens = self._tokens(p["text"])
            recall = self._fact_recall(fact_tokens, passage_tokens)
            best_recall = max(best_recall, recall)
        return "S" if best_recall >= self.threshold else "NS"

class EntailmentFactChecker(FactChecker):
    WORD_RE = re.compile(r"\w+")
    SENT_SPLIT_RE = re.compile(r"</s>|(?<=[.!?;:])\s+|\n+")

    def __init__(
        self,
        ent_model,
        entail_threshold: float = 0.45,       # keeps S recall healthy
        entail_high_threshold: float = 0.70,  # confident shortcut
        margin_threshold: float = 0.06,       # ↑ a touch to shave FPs
        prune_overlap_threshold: float = 0.05,# light gate for speed
        max_sentences_per_passage: int = 60,  # safety cap
        top_m_per_passage: int = 8,           # local cap
        top_k_candidates: int = 28,           # global cap (keeps runtime ~½–⅓)
        hybrid_overlap_fallback: float = 0.74 # lexical backstop
    ):
        self.ent_model = ent_model
        self.entail_threshold = entail_threshold
        self.entail_high_threshold = entail_high_threshold
        self.margin_threshold = margin_threshold
        self.prune_overlap_threshold = prune_overlap_threshold
        self.max_sentences_per_passage = max_sentences_per_passage
        self.top_m_per_passage = top_m_per_passage
        self.top_k_candidates = top_k_candidates
        self.hybrid_overlap_fallback = hybrid_overlap_fallback

    def _tokens(self, s: str):
        return set(t.lower() for t in self.WORD_RE.findall(s))

    def _fact_recall(self, fact_tokens, text_tokens) -> float:
        if not fact_tokens:
            return 0.0
        return len(fact_tokens & text_tokens) / len(fact_tokens)

    def _sentences(self, passage_text: str):
        txt = passage_text.replace("<s>", " ").strip()
        parts = [s.strip() for s in self.SENT_SPLIT_RE.split(txt) if s.strip()]
        if len(parts) > self.max_sentences_per_passage:
            parts = parts[:self.max_sentences_per_passage]
        return parts

    def predict(self, fact: str, passages: List[dict]) -> str:
        fact_toks = self._tokens(fact)

        # Early global prune: if no sentence in any passage reaches tiny overlap, call NS (very fast path)
        global_best_overlap = 0.0

        # 1) Per-passage candidate selection (keep top-m by recall)
        per_passage_candidates = []
        for p in passages:
            title = p.get("title", "").strip()
            sents = self._sentences(p["text"])
            if not sents:
                continue

            scored = []
            for s in sents:
                rec = self._fact_recall(fact_toks, self._tokens(s))
                if rec >= self.prune_overlap_threshold:
                    scored.append((rec, s, title))
                global_best_overlap = max(global_best_overlap, rec)

            if not scored:
                # keep best one anyway
                best = max(
                    ((self._fact_recall(fact_toks, self._tokens(s)), s, title) for s in sents),
                    key=lambda x: x[0],
                    default=None
                )
                if best:
                    scored = [best]
                    global_best_overlap = max(global_best_overlap, scored[0][0])

            if scored:
                # local top-m
                per_passage_candidates.extend(nlargest(self.top_m_per_passage, scored, key=lambda x: x[0]))

        if not per_passage_candidates:
            return "NS"

        # Global top-k
        topk = nlargest(self.top_k_candidates, per_passage_candidates, key=lambda x: x[0])

        # If even the best lexical overlap is extremely high, remember it for the fallback
        best_overlap = topk[0][0] if topk else global_best_overlap

        # 2) Build single-premise strings with title context (no double scoring)
        premises = []
        for _, sent, title in topk:
            prem = f"{title}. {sent}" if title else sent
            premises.append(prem)

        # 3) One batched entailment call
        probs = self.ent_model.check_entailment_batch(premises, fact, batch_size=16)

        # 4) Aggregate: best p_ent and best margin
        best_p_ent = 0.0
        best_margin = -1.0
        best_p_contra = 0.0
        for (p_ent, _p_neu, p_contra) in probs:
            if p_ent > best_p_ent:
                best_p_ent = p_ent
            margin = p_ent - p_contra
            if margin > best_margin:
                best_margin = margin
            if p_contra > best_p_contra:
                best_p_contra = p_contra

        # ---- Decision rule (precision-leaning) ----
        # thresholds: slightly stricter to shave FPs
        ENTAIL_HIGH = 0.70       # was 0.70
        ENTAIL_MAIN = 0.42       # was 0.46
        MARGIN = 0.03           # was 0.06
        HYBRID_OVERLAP = 0.68    # was 0.72
        CONTRA_VETO = 0.55       # new: if contradiction is strong, force NS
        # Dynamic entail threshold: lower it when lexical overlap is strong
        # Scale: if best_overlap ∈ [0.0, 0.9+], reduce threshold by up to 0.06
        effective_entail = ENTAIL_MAIN - 0.06 * min(1.0, best_overlap / 0.90)


        # Contradiction veto for borderline cases
        if best_p_contra >= CONTRA_VETO and best_margin < 0.04:
            return "NS"

        if best_p_ent >= ENTAIL_HIGH:
            return "S"
        if (best_p_ent >= effective_entail) and (best_margin >= MARGIN):
            return "S"
        if (best_p_ent >= effective_entail - 0.05) and (best_overlap >= HYBRID_OVERLAP):
            return "S"
        if (best_p_ent >= 0.50) and (best_p_contra <= 0.20) and (best_margin >= 0.02):
            return "S"
        if (best_p_ent >= 0.52) and (best_p_contra <= 0.18):
            return "S"


        return "NS"




# OPTIONAL
class DependencyRecallThresholdFactChecker(FactChecker):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")

    def get_dependencies(self, sent: str):
        """
        Returns a set of relevant dependencies from sent
        :param sent: The sentence to extract dependencies from
        :param nlp: The spaCy model to run
        :return: A set of dependency relations as tuples (head, label, child) where the head and child are lemmatized
        if they are verbs. This is filtered from the entire set of dependencies to reflect ones that are most
        semantically meaningful for this kind of fact-checking
        """
        # Runs the spaCy tagger
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            # Simplify the relation to its basic form (root verb form for verbs)
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations
