from data_pipeline import get_data
from transformer_model import build_log_transformer_model
import tensorflow as tf
import pickle
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

class WarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model=float(d_model)
        self.warmup_steps=warmup_steps
        
    def __call__(self, step):
        step=tf.cast(step,tf.float32)
        arg1=tf.math.rsqrt(step+1e-8)
        arg2=step*(self.warmup_steps**-1.5)
        return tf.math.rsqrt(self.d_model)*tf.math.minimum(arg1,arg2)
    
    def get_config(self):
        return {'d_model': self.d_model, 'warmup_steps': self.warmup_steps}
    
def score_anomalies(model, X, y_true,X_calib=None, n_sigma=2.0, use_entropy=False):
    print("\nScoring sequences ...")
    # probs = model.predict(X, batch_size=256, verbose=1)
    if X_calib is not None:
        calib_probs = model.predict(X_calib, batch_size=256, verbose=0)
        if use_entropy:
            lp  = tf.math.log(calib_probs + 1e-9)
            ent = -tf.reduce_sum(calib_probs * lp, axis=-1)
            calib_scores = tf.reduce_mean(ent, axis=-1).numpy()
        else:
            X_calib_np = X_calib.numpy() if hasattr(X_calib, 'numpy') else np.array(X_calib)
            c_losses = []
            for i in range(0, len(X_calib_np), 256):
                bx   = X_calib_np[i:i+256]
                bp   = calib_probs[i:i+256]
                tlp  = np.log(
                    bp[np.arange(len(bx))[:, None],
                       np.arange(bx.shape[1])[None, :],
                       bx] + 1e-9
                )
                c_losses.append(-tlp.mean(axis=1))
            calib_scores = np.concatenate(c_losses)
    else:
        calib_scores = scores[y_true == 0]

    threshold   = calib_scores.mean() + n_sigma * calib_scores.std()
    predictions = (scores > threshold).astype(int)

    print(f"\n── Anomaly Detection Results ──────────────────────")
    print(f"   Threshold  : {threshold:.6f}")
    print(f"   Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"   Flagged    : {predictions.sum()} / {len(predictions)} sequences\n")

    print(classification_report(y_true, predictions, target_names=['Normal', 'Anomaly']))

    try:
        auc = roc_auc_score(y_true, scores)
        print(f"   ROC-AUC    : {auc:.4f}")
    except Exception:
        pass

    return scores, predictions, threshold

Trace_path='Event_traces.csv'
Label_path='anomaly_label.csv'
EMBEDDING_DIM = 64
print("Loading data ...")
train_ds, val_ds,X_val, X_all, y_all, log_vectorizer = get_data(
    trace_path=Trace_path,
    label_path=Label_path,
    max_len=50,
    batch_size=64,
    val_split=0.15,
)
Vocab_size = len(log_vectorizer.get_vocabulary())
SEQ_LEN    = 50
print(f"Vocab size : {Vocab_size}")
print(f"Train batches: {len(train_ds)}  |  Val batches: {len(val_ds)}")

model = build_log_transformer_model(
    vocab_size=Vocab_size,
    sequence_length=SEQ_LEN,
    num_heads=4,
    ff_dim=128,
    embedding_dim=EMBEDDING_DIM,
    num_layers=2,
)

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=WarmupSchedule(d_model=EMBEDDING_DIM, warmup_steps=4000),
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9,
        clipnorm=1.0),
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy']
)
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=4,
        restore_best_weights=True,
        verbose=1,
    ),
]

print("Starting training...")

Epochs=30

history=model.fit(
    train_ds,
    epochs=Epochs,
    validation_data=val_ds,
    callbacks=callbacks,
    verbose=1
)

print("Training completed.")

scores, predictions, threshold = score_anomalies(
    model=model,
    X=X_all,
    y_true=y_all,
    X_calib=X_val,
    n_sigma=2.0,
    use_entropy=False,
)

print("\nSaving model")
model.save('saved_log_transformer_model.keras')

with open('vectorizer_vocab.pkl', 'wb') as f:
    pickle.dump({'vocabulary': log_vectorizer.get_vocabulary()}, f)

with open('anomaly_threshold.txt', 'w') as f:
    f.write(str(threshold))

print(f"Saved: saved_log_transformer_model.keras")
print(f"Saved: vectorizer_vocab.pkl")
print(f"Saved: anomaly_threshold.txt  (threshold={threshold:.6f})")
