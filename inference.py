# import numpy as np
# import tensorflow as tf
# import pickle
# from data_pipeline import get_data
# from sklearn.metrics import classification_report, f1_score, roc_auc_score

# print("\nLoading saved model.")
# model=tf.keras.models.load_model('best_model.keras',compile=False)
# print("Model loaded successfully.")
# model.summary()

# print("\nLoading data for inference.")
# train_ds,val_ds,X_val,X_all,y_all,log_vectorizer=get_data(
#     trace_path='Event_traces.csv',
#     label_path='anomaly_label.csv',
#     max_len=50,
#     batch_size=64,
#     val_split=0.15,
# )

# normal_ratio  = (y_all == 0).mean()
# anomaly_ratio = (y_all == 1).mean()

# print(f"X_all  : {X_all.shape}")
# print(f"X_val  : {X_val.shape}  (held-out normal sequences)")
# print(f"Normal : {normal_ratio*100:.1f}%  |  Anomaly: {anomaly_ratio*100:.1f}%")


# def compute_scores(model,X):
#     probs=model.predict(X,batch_size=256,verbose=1)
#     X_np=X.numpy() if hasattr(X,'numpy') else np.array(X)
#     losses=[]
    
#     for i in range(0,len(X_np),256):
#         batch_x=X_np[i:i+256]
#         batch_prob=probs[i:i+256]
        
#         true_log_probs=np.log(
#             batch_prob[np.arange(len(batch_x))[:,None],
#                        np.arange(batch_x.shape[1])[None, :],
#                        batch_x]+1e-9
#         )
#         losses.append(-true_log_probs.max(axis=-1))
#     return np.concatenate(losses)

# print("\n" + "=" * 60)
# print("Scoring all sequences ...")
# all_scores = compute_scores(model, X_all)

# # ── score distribution diagnostics ───────────────────────────────────────────
# print("\n── Score distribution by class ─────────────────────────────────")
# normal_scores  = all_scores[y_all == 0]
# anomaly_scores = all_scores[y_all == 1]

# print(f"\n   Normal sequences  ({len(normal_scores):,}):")
# print(f"      mean  : {normal_scores.mean():.8f}")
# print(f"      std   : {normal_scores.std():.8f}")
# print(f"      max   : {normal_scores.max():.8f}")
# print(f"      > 1e-6: {(normal_scores > 1e-6).sum():,} sequences")

# print(f"\n   Anomaly sequences ({len(anomaly_scores):,}):")
# print(f"      mean  : {anomaly_scores.mean():.8f}")
# print(f"      std   : {anomaly_scores.std():.8f}")
# print(f"      min   : {anomaly_scores.min():.8f}")
# print(f"      > 1e-6: {(anomaly_scores > 1e-6).sum():,} sequences")

# print(f"\n   Separation (anomaly mean - normal mean): "
#       f"{anomaly_scores.mean() - normal_scores.mean():.8f}")
# print("   (larger = better — model clearly distinguishes the two)")

# # All-data percentiles — shows where the natural gap sits
# print(f"\n── All-data score percentiles ──────────────────────────────────")
# for p in [50, 90, 95, 97, 98, 99, 99.5, 99.9]:
#     val = np.percentile(all_scores, p)
#     print(f"   p{p:5.1f} = {val:.8f}")

# # ROC-AUC — threshold-independent quality metric
# print(f"\n── ROC-AUC ─────────────────────────────────────────────────────")
# try:
#     auc = roc_auc_score(y_all, all_scores)
#     print(f"   {auc:.4f}   (0.5 = random  |  1.0 = perfect)")
#     print(f"   This measures raw score separation regardless of threshold.")
# except Exception as e:
#     print(f"   Could not compute: {e}")


# # ─────────────────────────────────────────────────────────────────────────────
# # 5A. APPROACH 1 — ABSOLUTE THRESHOLD
# # ─────────────────────────────────────────────────────────────────────────────
# # Normal scores are numerically zero (floating point underflow).
# # Anomaly scores are 0.1+.
# # So any small positive epsilon cleanly separates them.

# print("\n" + "=" * 60)
# print("APPROACH 1 — Absolute threshold (exploits zero vs non-zero gap)")
# print("=" * 60)

# absolute_thresholds = [1e-9, 1e-6, 1e-4, 0.01, 0.05, 0.1]

# best_f1_abs       = 0
# best_thresh_abs   = 1e-6

# for thresh in absolute_thresholds:
#     predictions = (all_scores > thresh).astype(int)
#     f1 = f1_score(y_all, predictions, zero_division=0)

#     print(f"\n   threshold = {thresh:.2e}  |  "
#           f"flagged = {predictions.sum():,} / {len(predictions):,}")
#     print(classification_report(
#         y_all, predictions,
#         target_names=['Normal', 'Anomaly'],
#         zero_division=0,
#     ))

#     if f1 > best_f1_abs:
#         best_f1_abs     = f1
#         best_thresh_abs = thresh

# print(f"\n   ✓ Best absolute threshold : {best_thresh_abs:.2e}  "
#       f"(anomaly F1 = {best_f1_abs:.4f})")


# # ─────────────────────────────────────────────────────────────────────────────
# # 5B. APPROACH 2 — NORMAL-RATIO PERCENTILE THRESHOLD
# # ─────────────────────────────────────────────────────────────────────────────
# # Your dataset is 97.1% normal and 2.9% anomalous.
# # The bottom 97.1% of all scores should be the normal cluster.
# # Setting the threshold at p(97.1) of all scores sits exactly at the
# # boundary between the zero cluster and the anomaly cluster.
# # This is the most principled approach for imbalanced datasets.

# print("\n" + "=" * 60)
# print("APPROACH 2 — Normal-ratio percentile threshold")
# print("=" * 60)

# boundary_percentile = normal_ratio * 100   # e.g. 97.1
# threshold_ratio     = np.percentile(all_scores, boundary_percentile)
# predictions_ratio   = (all_scores > threshold_ratio).astype(int)
# f1_ratio            = f1_score(y_all, predictions_ratio, zero_division=0)

# print(f"\n   Normal ratio      : {normal_ratio*100:.1f}%")
# print(f"   Boundary percentile: p{boundary_percentile:.1f}")
# print(f"   Threshold          : {threshold_ratio:.8f}")
# print(f"   Flagged            : {predictions_ratio.sum():,} / {len(predictions_ratio):,}")
# print()
# print(classification_report(
#     y_all, predictions_ratio,
#     target_names=['Normal', 'Anomaly'],
#     zero_division=0,
# ))
# print(f"   Anomaly F1 : {f1_ratio:.4f}")

# # Also try nearby percentiles to see the precision/recall trade-off
# print("\n   Nearby percentiles for trade-off analysis:")
# print(f"   {'percentile':>12}  {'threshold':>12}  {'flagged':>8}  "
#       f"{'precision':>10}  {'recall':>8}  {'F1':>6}")

# from sklearn.metrics import precision_score, recall_score

# for p in [95, 96, 97, 97.5, 98, 99]:
#     t    = np.percentile(all_scores, p)
#     pred = (all_scores > t).astype(int)
#     prec = precision_score(y_all, pred, zero_division=0)
#     rec  = recall_score(y_all, pred, zero_division=0)
#     f1   = f1_score(y_all, pred, zero_division=0)
#     print(f"   p{p:>10.1f}  {t:>12.8f}  {pred.sum():>8,}  "
#           f"{prec:>10.4f}  {rec:>8.4f}  {f1:>6.4f}")


# # ─────────────────────────────────────────────────────────────────────────────
# # 6. PICK AND SAVE BEST THRESHOLD
# # ─────────────────────────────────────────────────────────────────────────────

# print("\n" + "=" * 60)
# print("FINAL SUMMARY")
# print("=" * 60)

# # Pick the better of the two approaches
# if f1_ratio >= best_f1_abs:
#     final_threshold = threshold_ratio
#     final_f1        = f1_ratio
#     final_method    = f"normal-ratio percentile (p{boundary_percentile:.1f})"
# else:
#     final_threshold = best_thresh_abs
#     final_f1        = best_f1_abs
#     final_method    = f"absolute threshold ({best_thresh_abs:.2e})"

# print(f"\n   Method    : {final_method}")
# print(f"   Threshold : {final_threshold:.8f}")
# print(f"   Anomaly F1: {final_f1:.4f}")
# print(f"   ROC-AUC   : {auc:.4f}")

# final_predictions = (all_scores > final_threshold).astype(int)
# print(f"\n   Final classification report:")
# print(classification_report(
#     y_all, final_predictions,
#     target_names=['Normal', 'Anomaly'],
#     zero_division=0,
# ))

# with open('anomaly_threshold.txt', 'w') as f:
#     f.write(str(final_threshold))

# print("Saved → anomaly_threshold.txt")
# print("Done. No retraining was needed.")

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score,
    precision_score, recall_score,
)
from data_pipeline import get_data
 
print("=" * 60)
print("Loading saved model ...")
model = tf.keras.models.load_model('best_model.keras', compile=False)
print("Model loaded.\n")
 
print("=" * 60)
print("Loading data ...")
train_ds, val_ds, X_val, X_all, y_all, log_vectorizer = get_data(
    trace_path='Event_traces.csv',
    label_path='anomaly_label.csv',
    max_len=50,
    batch_size=64,
    val_split=0.15,
)
 
normal_ratio  = (y_all == 0).mean()
anomaly_ratio = (y_all == 1).mean()
 
print(f"X_all  : {X_all.shape}")
print(f"X_val  : {X_val.shape}")
print(f"Normal : {normal_ratio*100:.1f}%  |  Anomaly: {anomaly_ratio*100:.1f}%")
 
def compute_token_losses(model, X):
    X_np  = X.numpy() if hasattr(X, 'numpy') else np.array(X)
    probs = model.predict(X, batch_size=256, verbose=1)
    all_token_losses = []
 
    for i in range(0, len(X_np), 256):
        bx = X_np[i:i+256]
        bp = probs[i:i+256]
 
        true_log_p = np.log(
            bp[
                np.arange(len(bx))[:, None],
                np.arange(bx.shape[1])[None, :],
                bx
            ] + 1e-9
        )
        all_token_losses.append(-true_log_p)
 
    return np.concatenate(all_token_losses)
 
def apply_strategy(token_losses, strategy='max', k=3):
    if strategy == 'mean':
        return token_losses.mean(axis=1)
    elif strategy == 'max':
        return token_losses.max(axis=1)
    elif strategy == 'top_k':
        sorted_losses = np.sort(token_losses, axis=1)[:, ::-1]
        return sorted_losses[:, :k].mean(axis=1)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
 
print("\n" + "=" * 60)
print("Computing per-token losses for all sequences ...")
token_losses = compute_token_losses(model, X_all)
 
print("\n" + "=" * 60)
print("Applying Aggressive 'MAX' Strategy")
print("=" * 60)
 
scores = apply_strategy(token_losses, strategy='max')

try:
    auc = roc_auc_score(y_all, scores)
except Exception:
    auc = 0.0

aggressive_threshold = np.percentile(scores, 95)
predictions = (scores > aggressive_threshold).astype(int)

prec = precision_score(y_all, predictions, zero_division=0)
rec  = recall_score(y_all, predictions, zero_division=0)
f1   = f1_score(y_all, predictions, zero_division=0)
 
print(f"\n── Strategy: MAX (Hyper-Sensitive)")
print(f"   ROC-AUC   : {auc:.4f}")
print(f"   Threshold : {aggressive_threshold:.2e}")
print(f"   Precision : {prec:.4f}  |  Recall: {rec:.4f}")
print(f"   F1 Score  : {f1:.4f}")
print(f"   Flagged   : {predictions.sum():,} / {len(predictions):,}")
print()
print(classification_report(
    y_all, predictions,
    target_names=['Normal', 'Anomaly'],
    zero_division=0,
))
 
print("=" * 60)
print("SAVING AGGRESSIVE CONFIGURATION")
print("=" * 60)

with open('anomaly_threshold.txt', 'w') as f:
    f.write(str(aggressive_threshold))
with open('anomaly_strategy.txt', 'w') as f:
    f.write('max')
 
print("Saved → anomaly_threshold.txt")
print("Saved → anomaly_strategy.txt")
print("\nDone.")