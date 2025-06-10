import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
class CNN1D(nn.Module):
        def __init__(self, input_features, num_classes_model):
            super(CNN1D, self).__init__()
            self.conv1 = nn.Conv1d(1,32,kernel_size=3,padding=1); self.bn1=nn.BatchNorm1d(32); self.relu1=nn.ReLU(); self.pool1=nn.MaxPool1d(2,2); self.dropout1=nn.Dropout(0.25)
            self.conv2 = nn.Conv1d(32,64,kernel_size=3,padding=1); self.bn2=nn.BatchNorm1d(64); self.relu2=nn.ReLU(); self.pool2=nn.MaxPool1d(2,2); self.dropout2=nn.Dropout(0.25)
            bn1_original_mode=self.bn1.training; bn2_original_mode=self.bn2.training
            self.bn1.eval(); self.bn2.eval()
            with torch.no_grad():
                dummy_input = torch.randn(1,1,input_features)
                x = self.conv1(dummy_input); x=self.bn1(x); x=self.relu1(x); x=self.pool1(x)
                x = self.conv2(x); x=self.bn2(x); x=self.relu2(x); x=self.pool2(x)
                flattened_size = x.shape[1]*x.shape[2]
                if flattened_size <= 0:
                    self.bn1.train(bn1_original_mode); self.bn2.train(bn2_original_mode)
                    raise ValueError(f"Flattened size {flattened_size} invalid.")
            self.bn1.train(bn1_original_mode); self.bn2.train(bn2_original_mode)
            self.flatten=nn.Flatten(); self.fc1=nn.Linear(flattened_size,128)
            self.bn3=nn.BatchNorm1d(128); self.relu3=nn.ReLU(); self.dropout3=nn.Dropout(0.5)
            self.fc2=nn.Linear(128,num_classes_model)
        def forward(self,x):
            x=self.conv1(x); x=self.bn1(x); x=self.relu1(x); x=self.pool1(x); x=self.dropout1(x)
            x=self.conv2(x); x=self.bn2(x); x=self.relu2(x); x=self.pool2(x); x=self.dropout2(x)
            x=self.flatten(x)
            x=self.fc1(x); x=self.bn3(x); x=self.relu3(x); x=self.dropout3(x)
            x=self.fc2(x)
            return x
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