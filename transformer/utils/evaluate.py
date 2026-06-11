import torch
import numpy as np
from transformers import VivitForVideoClassification
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

def evaluate_testset(test_dataset, model_path, abnormal=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the best model
    model = VivitForVideoClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    all_logits = []
    all_labels = []

    # Iterate through the dataset
    for sample in test_dataset:
        inputs = {
            k: v.unsqueeze(0).to(device) if k == "pixel_values" and v.ndim == 4 else v.to(device)
            for k, v in sample.items()
            if k != "labels"
        }
        label = sample["labels"]

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # Shape: [1, 2] for binary classification

        all_logits.append(logits.cpu().numpy())
        all_labels.append(label)

    # Stack outputs and labels
    logits = np.vstack(all_logits)                     # Shape: (num_samples, 2)
    labels = np.array(all_labels)                      # Shape: (num_samples,)
    pred_labels = np.argmax(logits, axis=1)            # Predicted classes

    # Compute metrics
    accuracy = accuracy_score(labels, pred_labels)
    precision = precision_score(labels, pred_labels, zero_division=0)
    recall = recall_score(labels, pred_labels)
    f1 = f1_score(labels, pred_labels)
    roc_auc = roc_auc_score(labels, logits[:, 1])      # For binary classification
    conf_matrix = confusion_matrix(labels, pred_labels)

    result = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": conf_matrix,
        "logits": logits,
        "pred_labels": pred_labels,
        "true_labels": labels
    }

    # save the results into a txt file
    with open(f"{model_path}/evaluation_{'abnormal_' if abnormal else ''}results.txt", "w") as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1 Score: {f1}\n")
        f.write(f"ROC AUC: {roc_auc}\n")
        f.write(f"Confusion Matrix:\n{conf_matrix}\n")
        f.write(f"Logits: {logits}\n")
        f.write(f"Predicted Labels: {pred_labels}\n")
        f.write(f"True Labels: {labels}\n")
    return result