from torch.utils.data import Dataset, Subset
import torch
import json
import random


class VideoDiskDataset(Dataset):
    def __init__(self, metadata_path):
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = torch.load(self.metadata[idx]["path"], weights_only=True)
        return {
            "pixel_values": item["pixel_values"],  # shape: [T, 3, 224, 224]
            "labels": item["labels"]
        }
    
    def shuffle(self, random_state=42):
        rng = random.Random(random_state)
        rng.shuffle(self.metadata)

def reshuffle_dataset(dataset, seed=42):
    indices = list(range(len(dataset)))
    random.seed(seed)
    random.shuffle(indices)
    subset = Subset(dataset, indices)
    assert len(subset) == len(dataset), "Subset length does not match original dataset length."
    return subset