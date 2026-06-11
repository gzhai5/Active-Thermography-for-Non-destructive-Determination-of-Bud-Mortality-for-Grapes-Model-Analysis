from utils.video_disk_dataset import VideoDiskDataset, reshuffle_dataset
from torch.utils.data import ConcatDataset


def load_data(cultivar: str):
    base_path = "./refined_data_21"
    frame_folder = "frame-critical-3"

    def load_and_shuffle(cultivar_name: str, split: str, shuffle=False):
        path = f"{base_path}/{cultivar_name}/{frame_folder}/{split}/metadata.json"
        dataset = VideoDiskDataset(path)
        if shuffle:
            dataset = dataset.shuffle(random_state=42)
        return dataset

    if cultivar in ["RIES", "CF", "CON", "PN"]:
        train_dataset = load_and_shuffle(cultivar, "train", shuffle=False)
        train_dataset = reshuffle_dataset(train_dataset, seed=42)
        val_dataset = load_and_shuffle(cultivar, "valid", shuffle=False)
        val_dataset = reshuffle_dataset(val_dataset, seed=42)
        test_dataset = load_and_shuffle(cultivar, "test", shuffle=False)
        return train_dataset, val_dataset, test_dataset

    elif cultivar == "ALL":
        train_dataset = ConcatDataset([
            load_and_shuffle("RIES", "train", shuffle=False),
            load_and_shuffle("CF", "train", shuffle=False),
            load_and_shuffle("CON", "train", shuffle=False),
            load_and_shuffle("PN", "train", shuffle=False),
        ])
        train_dataset = reshuffle_dataset(train_dataset, seed=42)

        val_dataset = ConcatDataset([
            load_and_shuffle("RIES", "valid", shuffle=False),
            load_and_shuffle("CF", "valid", shuffle=False),
            load_and_shuffle("CON", "valid", shuffle=False),
            load_and_shuffle("PN", "valid", shuffle=False),
        ])
        val_dataset = reshuffle_dataset(val_dataset, seed=42)

        test_dataset = ConcatDataset([
            load_and_shuffle("RIES", "test", shuffle=False),
            load_and_shuffle("CF", "test", shuffle=False),
            load_and_shuffle("CON", "test", shuffle=False),
            load_and_shuffle("PN", "test", shuffle=False),
        ])

        return train_dataset, val_dataset, test_dataset

    else:
        raise ValueError(f"Unknown cultivar: {cultivar}. Expected one of ['RIES', 'CF', 'CON', 'PN', 'ALL'].")
