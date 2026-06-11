import numpy as np
from transformers import VivitConfig,VivitForVideoClassification
import torch
import evaluate

accuracy = evaluate.load("accuracy", trust_remote_code=True)
f1 = evaluate.load("f1", trust_remote_code=True)
recall = evaluate.load("recall", trust_remote_code=True)
precision = evaluate.load("precision", trust_remote_code=True)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids

    acc = accuracy.compute(predictions=preds, references=labels)
    f1_score = f1.compute(predictions=preds, references=labels, average="macro")
    recall_score = recall.compute(predictions=preds, references=labels, average="macro")
    precision_score = precision.compute(predictions=preds, references=labels, average="macro")

    return {
        "accuracy": acc["accuracy"],
        "f1": f1_score["f1"],
        "recall": recall_score["recall"],
        "precision": precision_score["precision"]
    }

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'].clone().detach() for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])}


def initialise_model(shuffled_dataset, device="cpu", model="google/vivit-b-16x2-kinetics400"):
    """initialize model
    """ 
    labels = [0, 1]
    config = VivitConfig.from_pretrained(model)
    config.num_classes=len(labels)
    config.id2label = {str(i): c for i, c in enumerate(labels)}
    config.label2id = {c: str(i) for i, c in enumerate(labels)}
    config.num_frames=3
    config.video_size= [3, 224, 224]
    
    model = VivitForVideoClassification.from_pretrained(
    model,
    ignore_mismatched_sizes=True,
    config=config,).to(device)
    return model 