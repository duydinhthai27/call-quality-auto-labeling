# --- evaluation_utils.py ---
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc

def plot_confidence_distribution(confidences, title="Confidence Histogram"):
    plt.figure(figsize=(8, 4))
    plt.hist(confidences, bins=20, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel("Confidence")
    plt.ylabel("Sample Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def visualize_embeddings(X, y, method='pca', title="2D Embedding"):
    if method == 'pca':
        reducer = PCA(n_components=2)
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)

    X_reduced = reducer.fit_transform(X)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=y, palette='tab10', s=30, alpha=0.8)
    plt.title(f"{title} using {method.upper()}")
    plt.legend(loc='best', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()


def evaluate_and_report(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

def plot_learning_curves(train_losses, val_accuracies):
    epochs = range(1, len(train_losses) + 1)
    
    fig, ax1 = plt.subplots(figsize=(8,5))

    # Plot training loss on left y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color=color)
    ax1.plot(epochs, train_losses, marker='o', color=color, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Create second y-axis for validation accuracy
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Validation Accuracy', color=color)
    ax2.plot(epochs, val_accuracies, marker='s', linestyle='--', color=color, label='Validation Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    # Title and legend
    plt.title('Learning Curves')
    fig.tight_layout()
    
    # Combine legends of both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='best')

    plt.show()


def plot_roc_curve(model, X_test, y_test):
    from sklearn.preprocessing import label_binarize

    y_score = model.predict_proba(X_test)
    classes = np.unique(y_test)
    y_test_binarized = label_binarize(y_test, classes=classes)
    plt.figure(figsize=(8,6))

    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {c} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Multi-class ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()