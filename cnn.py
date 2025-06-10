import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from semi_supervisedlearning.data_loader import load_call_quality_data
import numpy as np
from model_cnn import CNN1D
import torch.nn.functional as F
from model_cnn import pseudo_label_and_update_sets_for_cnn

confidence_threshold = 0.9
batch_size = 64
num_epochs = 50
num_self_training_iterations =5
FILE_PATH = "Filtered_Call_Records-cut (1).csv"  
# Load the data
target_col = "OverallCallQuality"

# Columns to ignore, Duy note: in case some columns are not in use but not remove all
cols_to_remove = [
     'StreamId', 'CallRecordId', 'Comment', 'SbcSessionId', 'CallerPhoneNumber',
        'CalleePhoneNumber', 'CallerIpAddress', 'CalleeIpAddress', 'CallerReflexiveIpAddress',
        'CalleeReflexiveIpAddress', 'CallerSubnet', 'CalleeSubnet',
        'SegmentFailedReason', 'SegmentFailureStage', 'CallerRelayIpAddress', 'CallerRelayPort',
        'CalleeRelayIpAddress', 'CalleeRelayPort', 'EstimatedGttCost', 'EstimatedSoftnetCost',
        'SbcEstimatedGttCost', 'SbcEstimatedSoftnetCost', 'CallStartTime', 'CallEndTime',
        'SbcSessionStartTime', 'SbcSessionEndTime',
        'SbcSessionStatus', 'Trunk', 'CallerPhoneNumberPrefix', 'CalleePhoneNumberPrefix', 'StreamQuality'
]

X, y, feature_names, original_df, scaler= load_call_quality_data(FILE_PATH, target_col, cols_to_remove)
num_input_features = X.shape[1]
num_classes_model = len(set(y))  # Assuming y contains class labels
print(f"Number of input features: {num_input_features}")
print(f"Number of classes: {num_classes_model}")
# -- call model and train --
## Split data for training and testing
X_temp_Ssl ,X_test_final, y_temp_Ssl, y_test_final = train_test_split(X, y, test_size=0.2, random_state=42)
X_unlabeled_current_np, X_labeled_initial_np, y_unlabeled_current_true_labels, y_labeled_initial_np = train_test_split(
            X_temp_Ssl, y_temp_Ssl,
            test_size=0.1,  # 10% of the data for initial labeled set
            random_state=42,
            stratify=y_temp_Ssl
        )
print(f"Initial Labeled Set (L) shape: X={X_labeled_initial_np.shape}, y={y_labeled_initial_np.shape}")
print(f"Initial Unlabeled Set (U) shape: X={X_unlabeled_current_np.shape} (true labels hidden for training)")
# Convert to numpy arrays if they're not already
X_labeled_current_np = X_labeled_initial_np.copy()
y_labeled_current_np = y_labeled_initial_np.copy()
#--Convert to torch tensors--
X_test_final_tensor = torch.tensor(X_test_final, dtype=torch.float32).unsqueeze(1)
y_test_final_tensor = torch.tensor(y_test_final, dtype=torch.long)
test_final_dataset = torch.utils.data.TensorDataset(X_test_final_tensor, y_test_final_tensor)
tes_final_loader = torch.utils.data.DataLoader(test_final_dataset, batch_size=64, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

final_model_for_evaluation = None

for iteration in range(num_self_training_iterations):
    print(f"\n--- Self-Training Iteration: {iteration + 1}/{num_self_training_iterations} ---")
    print(f"Current Labeled Set size: {X_labeled_current_np.shape[0]}")
    print(f"Current Unlabeled Set size: {X_unlabeled_current_np.shape[0]}")

    if X_labeled_current_np.shape[0] == 0 or X_unlabeled_current_np.shape[0] == 0:
        print("No labeled or unlabeled data. Stopping.")
        break

    # Optionally split labeled set into train/val
    if X_labeled_current_np.shape[0] > 1:
        val_size_iter = min(0.2, 500 / X_labeled_current_np.shape[0])
        val_size_iter = max(val_size_iter, 0.1) if X_labeled_current_np.shape[0] > 10 else 0
        if val_size_iter > 0 and X_labeled_current_np.shape[0] * val_size_iter >= num_classes_model:
            try:
                X_train_iter_np, X_val_iter_np, y_train_iter_np, y_val_iter_np = train_test_split(
                    X_labeled_current_np, y_labeled_current_np, test_size=val_size_iter, random_state=42+iteration, stratify=y_labeled_current_np)
            except ValueError:
                X_train_iter_np, X_val_iter_np, y_train_iter_np, y_val_iter_np = train_test_split(
                    X_labeled_current_np, y_labeled_current_np, test_size=val_size_iter, random_state=42+iteration)
        elif val_size_iter > 0:
            X_train_iter_np, X_val_iter_np, y_train_iter_np, y_val_iter_np = train_test_split(
                X_labeled_current_np, y_labeled_current_np, test_size=val_size_iter, random_state=42+iteration)
        else:
            X_train_iter_np, y_train_iter_np = X_labeled_current_np, y_labeled_current_np
            X_val_iter_np, y_val_iter_np = np.array([]).reshape(0, num_input_features), np.array([])
    else:
        X_train_iter_np, y_train_iter_np = X_labeled_current_np, y_labeled_current_np
        X_val_iter_np, y_val_iter_np = np.array([]).reshape(0, num_input_features), np.array([])

    # Prepare DataLoaders
    X_train_tensor = torch.tensor(X_train_iter_np, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train_iter_np, dtype=torch.long)
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = None
    if X_val_iter_np.shape[0] > 0:
        X_val_tensor = torch.tensor(X_val_iter_np, dtype=torch.float32).unsqueeze(1)
        y_val_tensor = torch.tensor(y_val_iter_np, dtype=torch.long)
        val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize a new model for this iteration
    model = CNN1D(num_input_features, num_classes_model).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if val_loader and (epoch % 5 == 0 or epoch == num_epochs - 1):
            model.eval()
            val_loss, val_acc, val_total = 0, 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, labels).item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    val_acc += (predicted == labels).sum().item()
                    val_total += labels.size(0)
            val_loss /= val_total
            val_acc /= val_total
            print(f"  Iter {iteration+1}, Epoch {epoch+1}: Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Pseudo-labeling and update sets
    X_labeled_current_np, y_labeled_current_np, X_unlabeled_current_np, y_unlabeled_current_true_labels, num_new_pseudo_labels = pseudo_label_and_update_sets_for_cnn(
        model, X_unlabeled_current_np, y_unlabeled_current_true_labels, 
        X_labeled_current_np, y_labeled_current_np, confidence_threshold, device
    )

    if num_new_pseudo_labels == 0 and iteration > 0:
        print("No new pseudo-labels. Stopping.")
        break

    final_model_for_evaluation = model

# Final evaluation on test set
if final_model_for_evaluation is not None:
    final_model_for_evaluation.eval()
    X_test_tensor = torch.tensor(X_test_final, dtype=torch.float32).unsqueeze(1).to(device)
    y_test_tensor = torch.tensor(y_test_final, dtype=torch.long).to(device)
    with torch.no_grad():
        outputs = final_model_for_evaluation(X_test_tensor)
        _, preds = torch.max(outputs, 1)
        accuracy = (preds == y_test_tensor).float().mean().item()
    print(f"Final Test Accuracy: {accuracy:.4f}")
else:
    print("No final model available for evaluation.")