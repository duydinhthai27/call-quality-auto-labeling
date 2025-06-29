# --- multi_model_runner.py ---
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from pprint import pprint
import numpy as np
from evaluation_utils import evaluate_and_report, visualize_embeddings, plot_learning_curves, plot_roc_curve
from data_loader import load_call_quality_data
from config import MODEL_REGISTRY, LABELED_RATIO
from active_learning import active_learning_cycle
from checkpoint_manager import save_checkpoint

def sample_and_label(X, y, labeled_ratio=LABELED_RATIO):
    n_labeled = int(len(X) * labeled_ratio)
    indices = np.random.permutation(len(X))
    labeled_indices = indices[:n_labeled]
    unlabeled_indices = indices[n_labeled:]
    return X[labeled_indices], y[labeled_indices], X[unlabeled_indices], y[unlabeled_indices]

def train_model(X_train, y_train, model_class, model_params):
    model = model_class(**model_params)
    model.fit(X_train, y_train)
    return model

def run_all_models(use_dynamic_threshold=False):
    X, y, feature_names, original_df, scaler = load_call_quality_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}
    learning_curves = {}
    total_epochs = 100

    for name, (model_class, model_params) in MODEL_REGISTRY.items():
        print(f"\nTraining model: {name}")
        X_labeled, y_labeled, X_unlabeled, y_unlabeled = sample_and_label(X_train, y_train)
        model = train_model(X_labeled, y_labeled, model_class, model_params)

        losses = []
        for iteration in range(total_epochs):
            model, X_labeled, y_labeled, train_loss, X_unlabeled, y_unlabeled = active_learning_cycle(
                model, X_labeled, y_labeled, X_unlabeled, y_unlabeled,
                X_val=X_test, y_val=y_test,
                epoch=iteration + 1, total_epochs=total_epochs,
                use_dynamic_threshold=use_dynamic_threshold
            )
            losses.append(train_loss)  # collect training loss per iteration
            save_checkpoint(model, name, iteration)

        y_pred = model.predict(X_test)
        final_acc = accuracy_score(y_test, y_pred)

        evaluate_and_report(model, X_test, y_test)
        visualize_embeddings(X_test, y_pred, method='pca', title=f"{name} Prediction Embedding")
        plot_learning_curves(losses, [accuracy_score(y_test, y_pred)] * len(losses))
        plot_roc_curve(model, X_test, y_test)

        results[name] = {'final_acc': final_acc}
        learning_curves[name] = losses

    # Plot accuracy bar chart
    plt.figure(figsize=(12, 6))
    models = list(results.keys())
    accuracies = [results[m]['final_acc'] for m in models]
    plt.bar(models, accuracies)
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # New plot: Learning curves (training loss) per model
    plt.figure(figsize=(12, 6))
    for name, losses in learning_curves.items():
        plt.plot(range(1, len(losses)+1), losses, marker='o', label=name)
    plt.xlabel('Active Learning Iteration')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Across Active Learning Iterations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(results)