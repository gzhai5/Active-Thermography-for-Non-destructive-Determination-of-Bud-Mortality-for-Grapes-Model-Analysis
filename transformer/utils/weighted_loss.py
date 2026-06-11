import torch.nn as nn
from transformers import Trainer


class WeightedLossTrainer(Trainer):
    def __init__(self, *args, weights_tensor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights_tensor = weights_tensor

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss(weight=self.weights_tensor)
        loss = loss_fct(logits, labels.long())

        return (loss, outputs) if return_outputs else loss
