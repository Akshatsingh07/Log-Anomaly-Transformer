"""
invariant_detector.py
=====================
Three-detector ensemble:
  1. Transformer autoencoder  — novel/short sequences
  2. N-gram (bigram+trigram)  — invalid local transitions
  3. Invariant miner          — wrong global structure

Fixes vs previous version
--------------------------
- NgramAnomalyDetector defined inline (no pickle import issue)
- Ordering invariants use min_support=1.0 (must hold in 100% of sequences)
  Previously 0.99 learned 212 pairs that flagged everything
- Count ranges use wider percentile band (0.5/99.5 instead of 1/99)
- Co-occurrence uses min_support=1.0 to avoid false positives

Usage:
    python invariant_detector.py
"""

import numpy as np
import pickle
from collections import defaultdict, Counter
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from data_pipeline import get_data
import tensorflow as tf


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

print("Loading data ...")
train_ds, val_ds, X_val, X_all, y_all, log_vectorizer = get_data(
    trace_path='Event_traces.csv',
    label_path='anomaly_label.csv',
    max_len=50,
    batch_size=64,
    val_split=0.15,
)

X_np        = X_all.numpy() if hasattr(X_all, 'numpy') else np.array(X_all)
normal_mask = (y_all == 0)
vocab       = log_vectorizer.get_vocabulary()

print(f"Vocab: {len(vocab)} tokens  |  Anomaly rate: {y_all.mean()*100:.1f}%")
print(f"Normal sequences for training: {normal_mask.sum():,}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. N-GRAM DETECTOR  (defined inline — avoids pickle class import issue)
# ─────────────────────────────────────────────────────────────────────────────

class NgramAnomalyDetector:
    def __init__(self, n_values=(2, 3)):
        self.n_values     = n_values
        self.valid_ngrams = {n: set() for n in n_values}

    def _ngrams(self, seq, n):
        tokens = [int(t) for t in seq if t != 0]
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def fit(self, X_normal):
        print(f"Fitting n-gram model on {len(X_normal):,} sequences ...")
        for seq in X_normal:
            for n in self.n_values:
                for ng in self._ngrams(seq, n):
                    self.valid_ngrams[n].add(ng)
        for n in self.n_values:
            print(f"   Valid {n}-grams: {len(self.valid_ngrams[n]):,}")

    def score(self, X):
        scores = np.zeros(len(X))
        for i, seq in enumerate(X):
            unseen = total = 0
            for n in self.n_values:
                ngs = self._ngrams(seq, n)
                total  += len(ngs)
                unseen += sum(1 for ng in ngs if ng not in self.valid_ngrams[n])
            scores[i] = unseen / max(total, 1)
        return scores

    def predict(self, X, threshold=0.0):
        scores = self.score(X)
        return (scores > threshold).astype(int), scores


# ─────────────────────────────────────────────────────────────────────────────
# 3. INVARIANT DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class InvariantDetector:
    """
    Mines structural invariants from normal sequences.

    Invariants (all require 100% support to avoid false positives):
      a. count_ranges     — min/max occurrences of each token
      b. always_together  — tokens that ALWAYS co-occur
      c. always_before    — strict ordering that ALWAYS holds
      d. n-grams          — valid local transitions (bigram+trigram)

    Support thresholds
    ------------------
    Using 1.0 (100%) for ordering/co-occurrence to be strict.
    Count ranges use 0.5/99.5 percentile band (very wide tolerance).
    This prevents flagging normal sequences due to minor variations.
    """

    def __init__(self):
        self.count_ranges    = {}
        self.always_together = defaultdict(set)
        self.always_before   = defaultdict(set)
        self.valid_ngrams    = {2: set(), 3: set()}

    def _tokens(self, seq):
        return [int(t) for t in seq if t != 0]

    def fit(self, X_normal, order_support=1.0, cooc_support=1.0):
        print(f"Mining invariants from {len(X_normal):,} normal sequences ...")
        N = len(X_normal)

        # ── a. count ranges ───────────────────────────────────────────────────
        token_counts = defaultdict(list)
        for seq in X_normal:
            tokens = self._tokens(seq)
            c = Counter(tokens)
            for tok in set(tokens):
                token_counts[tok].append(c[tok])

        for tok, counts in token_counts.items():
            freq = len(counts) / N
            if freq >= 0.05:     # token appears in ≥5% of sequences
                self.count_ranges[tok] = (
                    int(np.percentile(counts, 0.5)),    # very wide lower bound
                    int(np.percentile(counts, 99.5)),   # very wide upper bound
                )
        print(f"   Count range invariants : {len(self.count_ranges):,} tokens")

        # ── b. co-occurrence (always_together) ────────────────────────────────
        # Token A always appears with token B if every sequence containing A
        # also contains B (and vice versa). Support must be 100%.
        token_seqs = defaultdict(set)
        for i, seq in enumerate(X_normal):
            for tok in set(self._tokens(seq)):
                token_seqs[tok].add(i)

        for tok_a, seqs_a in token_seqs.items():
            for tok_b, seqs_b in token_seqs.items():
                if tok_a >= tok_b:
                    continue
                # Both must imply each other
                if (len(seqs_a & seqs_b) / len(seqs_a) >= cooc_support and
                        len(seqs_a & seqs_b) / len(seqs_b) >= cooc_support):
                    self.always_together[tok_a].add(tok_b)
                    self.always_together[tok_b].add(tok_a)

        n_pairs = sum(len(v) for v in self.always_together.values()) // 2
        print(f"   Co-occurrence invariants: {n_pairs:,} pairs")

        # ── c. ordering (always_before) ───────────────────────────────────────
        # "A always before B" means: in EVERY sequence containing both A and B,
        # the FIRST occurrence of A comes before the FIRST occurrence of B.
        # Using 100% support — if even one normal sequence violates it, drop it.
        pair_counts   = defaultdict(int)   # (a,b): times a first seen before b
        pair_together = defaultdict(int)   # (a,b): times both appear

        for seq in X_normal:
            tokens = self._tokens(seq)
            first_pos = {}
            for i, tok in enumerate(tokens):
                if tok not in first_pos:
                    first_pos[tok] = i

            toks = list(first_pos.keys())
            for i, a in enumerate(toks):
                for b in toks[i+1:]:
                    pair_together[(a, b)] += 1
                    pair_together[(b, a)] += 1
                    if first_pos[a] < first_pos[b]:
                        pair_counts[(a, b)] += 1   # a before b
                    else:
                        pair_counts[(b, a)] += 1   # b before a

        for (a, b), cnt in pair_counts.items():
            total = pair_together[(a, b)]
            if total > 0 and cnt / total >= order_support:
                self.always_before[b].add(a)    # a always before b

        n_order = sum(len(v) for v in self.always_before.values())
        print(f"   Ordering invariants    : {n_order:,} pairs  "
              f"(support={order_support:.0%})")

        # ── d. n-grams ────────────────────────────────────────────────────────
        for seq in X_normal:
            tokens = self._tokens(seq)
            for n in (2, 3):
                for i in range(len(tokens) - n + 1):
                    self.valid_ngrams[n].add(tuple(tokens[i:i+n]))

        print(f"   Valid bigrams          : {len(self.valid_ngrams[2]):,}")
        print(f"   Valid trigrams         : {len(self.valid_ngrams[3]):,}")

    def score_sequence(self, seq):
        tokens  = self._tokens(seq)
        if not tokens:
            return 0
        tok_set   = set(tokens)
        tok_count = Counter(tokens)
        violations = 0

        # a. count range violations
        for tok in tok_set:
            if tok in self.count_ranges:
                lo, hi = self.count_ranges[tok]
                c = tok_count[tok]
                if c < lo or c > hi:
                    violations += 1

        # b. co-occurrence violations
        for tok in tok_set:
            for required in self.always_together.get(tok, set()):
                if required not in tok_set:
                    violations += 1

        # c. ordering violations
        first_pos = {}
        for i, tok in enumerate(tokens):
            if tok not in first_pos:
                first_pos[tok] = i

        for tok in tok_set:
            for must_before in self.always_before.get(tok, set()):
                if must_before in tok_set:
                    if first_pos.get(must_before, 999) > first_pos.get(tok, 0):
                        violations += 1

        # d. n-gram violations
        for n in (2, 3):
            for i in range(len(tokens) - n + 1):
                if tuple(tokens[i:i+n]) not in self.valid_ngrams[n]:
                    violations += 1

        return violations

    def predict(self, X):
        scores = np.array([self.score_sequence(seq) for seq in X])
        return (scores > 0).astype(int), scores


# ─────────────────────────────────────────────────────────────────────────────
# 4. FIT ALL DETECTORS
# ─────────────────────────────────────────────────────────────────────────────

X_normal = X_np[normal_mask]

print("\n── N-gram detector ─────────────────────────────────────────────")
ngram = NgramAnomalyDetector(n_values=(2, 3))
ngram.fit(X_normal)
ngram_preds, ngram_scores = ngram.predict(X_np)

print("\n── Invariant detector ──────────────────────────────────────────")
inv = InvariantDetector()
inv.fit(X_normal, order_support=1.0, cooc_support=1.0)
inv_preds, inv_scores = inv.predict(X_np)

# ── individual results ────────────────────────────────────────────────────────
for label, preds, scores in [
    ("N-gram",     ngram_preds, ngram_scores),
    ("Invariant",  inv_preds,   inv_scores),
]:
    print(f"\n── {label} detector ─────────────────────────────────────────────")
    print(f"   Flagged: {preds.sum():,} / {len(preds):,}")
    print(classification_report(y_all, preds, target_names=['Normal','Anomaly'],
                                zero_division=0))
    try:
        print(f"   ROC-AUC: {roc_auc_score(y_all, scores):.4f}")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 5. TRANSFORMER SCORES
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Loading Transformer ...")
model = tf.keras.models.load_model('best_model.keras', compile=False)
probs = model.predict(X_all, batch_size=256, verbose=1)

all_losses = []
for i in range(0, len(X_np), 256):
    bx = X_np[i:i+256]
    bp = probs[i:i+256]
    tlp = np.log(
        bp[np.arange(len(bx))[:, None],
           np.arange(bx.shape[1])[None, :],
           bx] + 1e-9
    )
    all_losses.append(-tlp)

token_losses       = np.concatenate(all_losses)
transformer_scores = token_losses.max(axis=1)
transformer_preds  = (transformer_scores > 1e-9).astype(int)

print(f"\n── Transformer ─────────────────────────────────────────────────")
print(f"   Flagged: {transformer_preds.sum():,} / {len(transformer_preds):,}")
print(classification_report(y_all, transformer_preds,
                            target_names=['Normal','Anomaly'], zero_division=0))


# ─────────────────────────────────────────────────────────────────────────────
# 6. ENSEMBLE — OR ACROSS ALL THREE
# ─────────────────────────────────────────────────────────────────────────────

combined = np.maximum(transformer_preds,
           np.maximum(ngram_preds, inv_preds)).astype(int)

# Normalised combined score for AUC
t_norm = transformer_scores / (transformer_scores.max() + 1e-9)
i_norm = inv_scores / (inv_scores.max() + 1e-9)
combined_scores = np.maximum(t_norm, np.maximum(ngram_scores, i_norm))

print("\n" + "=" * 60)
print("ENSEMBLE: Transformer + N-gram + Invariant")
print("=" * 60)
print(f"   Flagged: {combined.sum():,} / {len(combined):,}")
print()
print(classification_report(y_all, combined,
                            target_names=['Normal','Anomaly'], zero_division=0))
try:
    auc = roc_auc_score(y_all, combined_scores)
    print(f"   ROC-AUC: {auc:.4f}")
except Exception:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# 7. BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────

anomaly_mask = (y_all == 1)
missed       = ~combined.astype(bool) & anomaly_mask

print(f"\n── Breakdown ({anomaly_mask.sum():,} total anomalies) ───────────────────")
print(f"   Transformer catches : {(transformer_preds & anomaly_mask).sum():,}")
print(f"   N-gram catches      : {(ngram_preds       & anomaly_mask).sum():,}")
print(f"   Invariant catches   : {(inv_preds         & anomaly_mask).sum():,}")
print(f"   Combined catches    : {(combined          & anomaly_mask).sum():,}")
print(f"   Still missed        : {missed.sum():,}")
print(f"\n   Final recall        : "
      f"{(combined & anomaly_mask).sum() / anomaly_mask.sum() * 100:.1f}%")

# ── save ──────────────────────────────────────────────────────────────────────
with open('invariant_detector.pkl', 'wb') as f:
    pickle.dump(inv, f)
with open('ngram_detector.pkl', 'wb') as f:
    pickle.dump(ngram, f)
print("\nSaved → invariant_detector.pkl")
print("Saved → ngram_detector.pkl")
print("Done.")