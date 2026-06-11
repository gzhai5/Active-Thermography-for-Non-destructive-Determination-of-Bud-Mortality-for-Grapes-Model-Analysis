import os, multiprocessing, gc
import wandb
import time
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import Trainer, TrainingArguments, AdamW, EarlyStoppingCallback
from utils.model_configuration import *
from utils.model_configuration import initialise_model, collate_fn
from utils.evaluate import evaluate_testset
from utils.weight import compute_weights
from utils.weighted_loss import WeightedLossTrainer
from utils.video_disk_dataset import VideoDiskDataset



# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
print(f'CPU count: {multiprocessing.cpu_count()}')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# load datasets
test_dataset_cf = VideoDiskDataset(r"C:\Users\Mud\Desktop\2ndPaper\transformer\refined_data_21\CF\frame-60\abnormal\test\metadata.json")
test_dataset_con = VideoDiskDataset(r"C:\Users\Mud\Desktop\2ndPaper\transformer\refined_data_21\CON\frame-60\abnormal\test\metadata.json")
test_dataset_ries = VideoDiskDataset(r"C:\Users\Mud\Desktop\2ndPaper\transformer\refined_data_21\PN\frame-60\abnormal\test\metadata.json")
test_dataset_pn = VideoDiskDataset(r"C:\Users\Mud\Desktop\2ndPaper\transformer\refined_data_21\RIES\frame-60\abnormal\test\metadata.json")
test_dataset = torch.utils.data.ConcatDataset([test_dataset_cf, test_dataset_con, test_dataset_ries, test_dataset_pn])
print(f"Test dataset are all loaded")



# Test on the test dataset
def run_test(NAME: str, test_dataset):
    results = evaluate_testset(test_dataset, f"./best_models_0417/{NAME}/", abnormal=True)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-score: {results['f1']:.4f}")
    print(f"ROC-AUC: {results['roc_auc']:.4f}")
    print(f"Confusion Matrix:\n{results['confusion_matrix']}")
    plt.figure(figsize=(6, 6))
    sns.heatmap(results['confusion_matrix'], annot=True, fmt="d", cmap="Blues", xticklabels=["Dead", "Alive"], yticklabels=["Dead", "Alive"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(f"./best_models_0417/{NAME}/abnormal_data_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Confusion matrix saved as confusion_matrix.png")

run_test("new-frame60-ALL-21-N-bs8-weight-5e-06-epoch8-cos", test_dataset)
run_test("new-frame60-CF-21-N-bs8-weight-5e-06-epoch8-cos", test_dataset_cf)
run_test("new-frame60-CON-21-N-bs8-weight-5e-06-epoch8-cos", test_dataset_con)
run_test("new-frame60-RIES-21-N-bs8-weight-5e-06-epoch8-cos", test_dataset_ries)
run_test("new-frame60-PN-21-N-bs8-weight-5e-06-epoch8-cos", test_dataset_pn)


# clean up
torch.cuda.empty_cache()
gc.collect()
wandb.finish()
print("Finished.")