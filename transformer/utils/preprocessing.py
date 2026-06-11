import numpy as np
import os
import av
import gc
import torch
from transformers import VivitImageProcessor, VivitModel
from utils.data_handling import frames_convert_and_create_dataset_dictionary
from datasets import Dataset, concatenate_datasets
from math import ceil
from pympler import asizeof



# Detect GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
def process_example(example):
    inputs = image_processor(list(np.array(example['video'])), return_tensors='pt')
    inputs['labels'] = example['labels']
    return inputs



def create_dataset(video_dictionary):
    print('Starting dataset creation..., input size:', asizeof.asizeof(video_dictionary) / 1024**2, 'MB')
    
    # batches = []
    # if len(video_dictionary) > 200:
    #     batches.append(Dataset.from_list(video_dictionary[:200]))
    #     print('current batch size:', asizeof.asizeof(batches) / 1024**2, 'MB')
    #     batches.append(Dataset.from_list(video_dictionary[200:]))
    #     print('current batch size:', asizeof.asizeof(batches) / 1024**2, 'MB')
    #     print('0:', batches[0])
    #     print("1", batches[1])
    #     dataset = concatenate_datasets(batches)
    #     print('01', dataset)
    # else:
    #     dataset = Dataset.from_list(video_dictionary)

    dataset = Dataset.from_list(video_dictionary)
    print('dataset has been created with size:', asizeof.asizeof(dataset) / 1024**2, 'MB')

    # Free memory from raw video data
    del video_dictionary
    gc.collect()
    torch.cuda.empty_cache()  # Free GPU memory

    dataset = dataset.class_encode_column("labels")

    # Process dataset
    processed_dataset = dataset.map(process_example, batched=False, load_from_cache_file=False)
    print('dataset has been processed with size:', asizeof.asizeof(processed_dataset) / 1024**2, 'MB')

    processed_dataset = processed_dataset.remove_columns(['video'])
    shuffled_dataset = processed_dataset.shuffle(seed=42)
    print('dataset has been shuffled')

    # Convert pixel values efficiently
    shuffled_dataset = shuffled_dataset.map(
        lambda x: {'pixel_values': torch.tensor(x['pixel_values'], device=device).squeeze()},
        load_from_cache_file=False
    )
    print('dataset has been shuffled with size:', asizeof.asizeof(shuffled_dataset) / 1024**2, 'MB')

    # Free memory from processed dataset
    del dataset, processed_dataset
    gc.collect()
    torch.cuda.empty_cache()

    return shuffled_dataset