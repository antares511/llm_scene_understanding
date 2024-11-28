import numpy as np
import os

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support,
)


def compute_metrics_by_class(y_true, y_pred, class_labels, label_set, save_folder):
    precisions = []
    recalls = []
    f1s = []

    evals_folder = save_folder
    evals_file = os.path.join(evals_folder, f"metrics_{label_set}.txt")
    if not os.path.exists(evals_folder):
        os.makedirs(evals_folder)

    print("Metrics by Class:")
    for room_label_id in np.unique(y_true):
        # Filter predictions and true values for the current class
        y_true_cls = (y_true == room_label_id).astype(int)
        y_pred_cls = (y_pred == room_label_id).astype(int)

        # Skip classes with no true instances
        if y_true_cls.sum() == 0:
            continue

        number_of_true = y_true_cls.sum()

        # Calculate metrics for the current class
        precision = precision_score(y_true_cls, y_pred_cls, zero_division=0)
        recall = recall_score(y_true_cls, y_pred_cls, zero_division=0)
        f1 = f1_score(y_true_cls, y_pred_cls, zero_division=0)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        print(
            f"Class: {class_labels[room_label_id]} ({number_of_true}), Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}"
        )

        with open(evals_file, "a") as f:
            f.write(
                f"Class: {class_labels[room_label_id]} ({number_of_true}), Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}\n"
            )

    # Calculate the average metrics across all classes
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1s)

    print("\nUnweighted Average Metrics:")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1-Score: {avg_f1:.4f}")

    with open(evals_file, "a") as f:
        f.write(f"\nAverage Precision: {avg_precision:.4f}\n")
        f.write(f"Average Recall: {avg_recall:.4f}\n")
        f.write(f"Average F1-Score: {avg_f1:.4f}\n")

    weighted_avg_precision, weighted_avg_recall, weighted_avg_recall_f1, _ = (
        precision_recall_fscore_support(y_true, y_pred, average="weighted")
    )
    print("\nWeighted Average Metrics:")
    print(f"Weighted Average Precision: {weighted_avg_precision:.4f}")
    print(f"Weighted Average Recall: {weighted_avg_recall:.4f}")
    print(f"Weighted Average F1-Score: {weighted_avg_recall_f1:.4f}")

    with open(evals_file, "a") as f:
        f.write(f"\nWeighted Average Precision: {weighted_avg_precision:.4f}\n")
        f.write(f"Weighted Average Recall: {weighted_avg_recall:.4f}\n")
        f.write(f"Weighted Average F1-Score: {weighted_avg_recall_f1:.4f}\n")

    accuracy = (y_true == y_pred).mean()
    print(f"\nAccuracy: {accuracy:.4f}")

    with open(evals_file, "a") as f:
        f.write(f"\nAccuracy: {accuracy:.4f}\n")

    return
