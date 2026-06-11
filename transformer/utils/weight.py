import torch
from collections import Counter


def compute_weights(dataset):
    label_counts = Counter([sample['labels'] for sample in dataset])
    print("Label counts:", label_counts)
    num_classes = len(label_counts)
    total = sum(label_counts.values())
    weights = [total / (num_classes * label_counts[i]) for i in range(num_classes)]
    weights_tensor = torch.tensor(weights, dtype=torch.float)
    return weights_tensor