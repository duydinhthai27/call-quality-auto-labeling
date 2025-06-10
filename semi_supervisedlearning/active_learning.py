# --- active_learning.py ---
import numpy as np
import pandas as pd
from datetime import datetime
from config import CONFIDENCE_THRESHOLD, SAMPLE_DIR
import torch
import numpy as np
import torch.nn.functional as F
def predict_with_confidence(model, X_unlabeled, threshold=CONFIDENCE_THRESHOLD):
    # Duy note: can be change the way we calculate the confidences, this version I use max, consider to use cross entropy 
    proba = model.predict_proba(X_unlabeled)
    predictions = np.argmax(proba, axis=1)
    confidences = np.max(proba, axis=1)
    high_conf_mask = confidences >= threshold
    return X_unlabeled[high_conf_mask], predictions[high_conf_mask], confidences[high_conf_mask],high_conf_mask

def pseudo_label_and_retrain(X_labeled, y_labeled, X_pseudo, y_pseudo, model):
    X_combined = np.vstack([X_labeled, X_pseudo])
    y_combined = np.hstack([y_labeled, y_pseudo])
    model.fit(X_combined, y_combined)
    return model, X_combined, y_combined
    
# Duy note: follow exactly current pipeline
def active_learning_cycle(model, X_labeled, y_labeled, X_unlabeled, y_unlabeled, threshold=0.9):
    X_pseudo , y_pseudo, conf_pseudo,high_conf_mask= predict_with_confidence(model, X_unlabeled, threshold)
    if len(X_pseudo) == 0:
        # No confident samples found
        return model, X_labeled, y_labeled, 0.0, X_unlabeled, y_unlabeled

    # Sample some pseudo-labels for CSV export
    sample_size = min(100, len(X_pseudo))
    sample_indices_in_pseudo = np.random.choice(len(X_pseudo), size=sample_size, replace=False)
    
    # Map sampled indices back to original unlabeled indices
    high_conf_indices = np.where(high_conf_mask)[0]
    sample_indices_in_unlabeled = high_conf_indices[sample_indices_in_pseudo]

    sample_data = pd.DataFrame({
        'features': list(X_unlabeled[sample_indices_in_unlabeled]),
        'predicted_label': y_pseudo[sample_indices_in_pseudo],
        'confidence': conf_pseudo[sample_indices_in_pseudo],
        'true_label': y_unlabeled[sample_indices_in_unlabeled]
    })
    sample_data.to_csv(f"{SAMPLE_DIR}/sample_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model}.csv")

    # Retrain on combined data
    model, X_combined, y_combined = pseudo_label_and_retrain(X_labeled, y_labeled, X_pseudo, y_pseudo, model)

    # Calculate training accuracy on combined data as proxy for loss
    train_acc = model.score(X_combined, y_combined)
    train_loss = 1.0 - train_acc

    # Remove confident pseudo-labeled samples from unlabeled dataset
    mask = np.ones(len(X_unlabeled), dtype=bool)
    mask[high_conf_indices] = False
    X_unlabeled_new = X_unlabeled[mask]
    y_unlabeled_new = y_unlabeled[mask]

    return model, X_combined, y_combined, train_loss, X_unlabeled_new, y_unlabeled_new

def predict_with_cross_entropy(model,X_unlabeled, threshold=CONFIDENCE_THRESHOLD):
    from scipy.stats import entropy
    proba = model.predict_proba(X_unlabeled)
    predictions = np.argmax(proba, axis=1)
    confidences = -entropy(proba.T)  # Cross-entropy as a measure of confidence
    high_conf_mask =confidences >= threshold
    return X_unlabeled[high_conf_mask], predictions[high_conf_mask], confidences[high_conf_mask],high_conf_mask
def pseudo_label_and_update_sets_for_cnn(model, X_unlabeled_current_np, y_unlabeled_current_true_labels, 
                                X_labeled_current_np, y_labeled_current_np, confidence_threshold, device):
   

    model.eval()
    X_unlabeled_tensor = torch.tensor(X_unlabeled_current_np, dtype=torch.float32).unsqueeze(1).to(device)
    with torch.no_grad():
        outputs = model(X_unlabeled_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidences, pseudo_labels = torch.max(probabilities, dim=1)
    confidences = confidences.cpu().numpy()
    pseudo_labels = pseudo_labels.cpu().numpy()

    # Select high-confidence pseudo-labels
    high_confidence_indices = np.where(confidences >= confidence_threshold)[0]
    num_new_pseudo_labels = len(high_confidence_indices)
    print(f"Found {num_new_pseudo_labels} new pseudo-labels with confidence >= {confidence_threshold}")

    if num_new_pseudo_labels > 0:
        # Add high-confidence pseudo-labeled data to labeled set
        X_labeled_current_np = np.concatenate([X_labeled_current_np, X_unlabeled_current_np[high_confidence_indices]], axis=0)
        y_labeled_current_np = np.concatenate([y_labeled_current_np, pseudo_labels[high_confidence_indices]], axis=0)
        # Remove them from unlabeled set
        mask = np.ones(X_unlabeled_current_np.shape[0], dtype=bool)
        mask[high_confidence_indices] = False
        X_unlabeled_current_np = X_unlabeled_current_np[mask]
        y_unlabeled_current_true_labels = y_unlabeled_current_true_labels[mask]

    return X_labeled_current_np, y_labeled_current_np, X_unlabeled_current_np, y_unlabeled_current_true_labels, num_new_pseudo_labels