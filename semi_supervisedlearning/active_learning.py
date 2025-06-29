from sklearn.metrics import log_loss
from scipy.stats import entropy
import numpy as np
import pandas as pd
from datetime import datetime
from .config import CONFIDENCE_THRESHOLD


def active_learning_cycle(model, X_labeled, y_labeled,
                          X_unlabeled, y_unlabeled,
                          X_val, y_val,
                          epoch, total_epochs,
                          use_dynamic_threshold=False,
                          tau_min=0.7, tau_max=0.95):

    if len(X_unlabeled) == 0:
        print(f"Epoch {epoch}: No unlabeled data remaining.")
        return model, X_labeled, y_labeled, 0.0, X_unlabeled, y_unlabeled

    def compute_dynamic_threshold(epoch, total_epochs, tau_min=0.7, tau_max=0.95):
        if total_epochs <= 1:
            return tau_max
        return tau_min + (tau_max - tau_min) * (epoch - 1) / (total_epochs - 1)

    if use_dynamic_threshold:
        threshold = compute_dynamic_threshold(epoch, total_epochs, tau_min, tau_max)
    else:
        threshold = CONFIDENCE_THRESHOLD

    # Predict on unlabeled set
    proba = model.predict_proba(X_unlabeled)
    predictions = np.argmax(proba, axis=1)

    # Confidence from entropy
    entropy_scores = entropy(proba.T)
    entropy_conf = 1 - entropy_scores / np.log(proba.shape[1])
    confidences = entropy_conf  # or (np.max(proba, axis=1) + entropy_conf) / 2

    high_conf_mask = confidences >= threshold
    high_conf_indices = np.where(high_conf_mask)[0]

    X_pseudo = X_unlabeled[high_conf_mask]
    y_pseudo = predictions[high_conf_mask]
    conf_pseudo = confidences[high_conf_mask]

    print(f"Epoch {epoch}: Pseudo-labels added: {len(X_pseudo)} at threshold {threshold:.4f}")

    if len(X_pseudo) == 0:
        return model, X_labeled, y_labeled, 0.0, X_unlabeled, y_unlabeled

    # Optional: sample export for inspection
    sample_size = min(100, len(X_pseudo))
    sample_indices = np.random.choice(len(X_pseudo), sample_size, replace=False)
    sample_indices_unlabeled = high_conf_indices[sample_indices]

    sample_data = pd.DataFrame({
        'features': list(X_unlabeled[sample_indices_unlabeled]),
        'predicted_label': y_pseudo[sample_indices],
        'confidence': conf_pseudo[sample_indices],
        'true_label': y_unlabeled[sample_indices_unlabeled]
    })
    # sample_data.to_csv("optional_export.csv", index=False)

    # Retrain
    X_combined = np.vstack([X_labeled, X_pseudo])
    y_combined = np.hstack([y_labeled, y_pseudo])
    model.fit(X_combined, y_combined)

    # Training metrics
    train_acc = model.score(X_combined, y_combined)
    try:
        proba_train = model.predict_proba(X_combined)
        train_loss = log_loss(y_combined, proba_train)
    except ValueError:
        train_loss = 1.0 - train_acc

    # Validation metrics
    val_acc = model.score(X_val, y_val)
    try:
        proba_val = model.predict_proba(X_val)
        val_loss = log_loss(y_val, proba_val)
    except ValueError:
        val_loss = 1.0 - val_acc

    print(f"Epoch {epoch} — Train Acc: {train_acc:.4f}, Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")

    # Update pool
    mask = np.ones(len(X_unlabeled), dtype=bool)
    mask[high_conf_indices] = False
    X_unlabeled_new = X_unlabeled[mask]
    y_unlabeled_new = y_unlabeled[mask]

    return model, X_combined, y_combined, train_loss, X_unlabeled_new, y_unlabeled_new
