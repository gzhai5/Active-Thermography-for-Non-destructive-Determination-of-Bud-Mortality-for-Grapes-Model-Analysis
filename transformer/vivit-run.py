import os, multiprocessing, gc
import wandb
import time
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from transformers import Trainer, TrainingArguments, AdamW, EarlyStoppingCallback
from utils.model_configuration import *
from utils.model_configuration import initialise_model, collate_fn
from utils.evaluate import evaluate_testset
from utils.weight import compute_weights
from utils.weighted_loss import WeightedLossTrainer
from utils.load_dataset import load_data
from utils.wandb_key import WANDB_KEY
from utils.set_seed import set_seed
from datetime import datetime



def main():
    # script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per GPU')
    parser.add_argument('--num_epochs', type=int, default=8, help='Number of epochs')
    parser.add_argument('--data', type=str, default='ALL', help='Dataset to use (RIES, CF, CON, PN, ALL)')
    parser.add_argument('--test', action='store_true', default=False, help='Test the model on the test dataset')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for initialization')
    args = parser.parse_args()


    # Params
    LR = args.lr if args.lr else 5e-4
    GRADIENT_ACCUMULATION_STEPS = 1
    BATCH_SIZE = args.batch_size if args.batch_size else 1
    NUM_EPOCHS = args.num_epochs if args.num_epochs else 8
    DATA = args.data if args.data else 'ALL'
    TEST = args.test if args.test else False
    SEED = args.seed if args.seed else 42
    set_seed(SEED)
    print(f"Running ViVit on {DATA} dataset with learning rate {LR}, batch size {BATCH_SIZE*GRADIENT_ACCUMULATION_STEPS}, {NUM_EPOCHS} epochs, and seed {SEED}.")


    start = time.time()
    # os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
    print(f'CPU count: {multiprocessing.cpu_count()}')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')


    # load datasets
    train_dataset, val_dataset, test_dataset = load_data(DATA)
    print(f"{DATA} dataset is loaded.")


    weights_tensor = compute_weights(train_dataset).to(device)
    print('Weights tensor:', weights_tensor)



    model = initialise_model(train_dataset, device)
    training_output_dir = "./tmp/results"
    training_args = TrainingArguments(
        output_dir=training_output_dir,         
        num_train_epochs=NUM_EPOCHS,         
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE, 
        learning_rate=LR,            
        weight_decay=0.01,
        logging_strategy="steps",              
        logging_dir="./logs",           
        logging_steps=10,                
        seed=SEED,                       
        eval_strategy="steps",    
        eval_steps=50,                   
        warmup_steps=int(0.1 * 20),      
        optim="adamw_torch",          
        lr_scheduler_type="cosine",      
        fp16=True,
        dataloader_num_workers=8,
        dataloader_persistent_workers=True,
        dataloader_pin_memory=True, 
        report_to="wandb",
        load_best_model_at_end=True,  # Automatically load the best model
        metric_for_best_model="eval_loss",  # Use validation loss to determine the best model
        greater_is_better=False,  # Lower loss is better
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,  # Gradient accumulation steps
    )   


    wandb.login(key=WANDB_KEY)
    PROJECT, MODEL_NAME, DATASET = "ViViT", "google/vivit-b-16x2-kinetics400", "bud-dataset"
    NAME = f"{datetime.today().strftime('%m%d')}-frame60-{DATA}-21-N-bs{BATCH_SIZE*GRADIENT_ACCUMULATION_STEPS}-weight-{LR}-epoch{NUM_EPOCHS}-cos"
    wandb.init(project=PROJECT, tags=[MODEL_NAME, DATASET], notes ="Fine tuning ViViT with grape buds dataset", name=NAME)


    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08)
    # Define the trainer
    trainer = WeightedLossTrainer(
        model=model,
        weights_tensor=weights_tensor,                   
        args=training_args,
        data_collator=collate_fn,             
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        optimizers=(optimizer, None),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],  
        compute_metrics = compute_metrics
    )

    # train
    with wandb.init(project=PROJECT, job_type="train", tags=[MODEL_NAME, DATASET], notes =f"Fine tuning {MODEL_NAME} with {DATASET}."):
        train_results = trainer.train()
    print(f"Train and process time passed: {((time.time() - start) / 60):.1f} minutes")


    # record down the val results for the best model to compare with other parameters
    best_model_metrics = trainer.evaluate()
    val_metrics = {
        "Batch Size": BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
        "Learning Rate": LR,
        "Best Epochs": trainer.state.epoch,
        "Best Steps": trainer.state.global_step,
        "Best Validation Loss": best_model_metrics["eval_loss"],
        "Best Validation Accuracy": best_model_metrics["eval_accuracy"],
        "Best Validation Precision": best_model_metrics["eval_precision"],
        "Best Validation Recall": best_model_metrics["eval_recall"],
        "Best Validation F1": best_model_metrics["eval_f1"],
    }
    # Save the validation metrics to a CSV file, append row if file exists
    val_metrics_df = pd.DataFrame([val_metrics])
    val_metrics_df.to_csv(f"./best_models/val_metrics.csv", mode='a', index=False, header=not os.path.exists(f"./best_models/val_metrics.csv"))
    print(f"Validation metrics saved to ./best_models/val_metrics.csv for run: {NAME}")


    # Save the model
    trainer.save_model(f"./best_models/{NAME}/")
    print(f"Model saved to ./best_models/{NAME}/")


    # Test on the test dataset
    if TEST:
        print("Testing on the test dataset...")
        results = evaluate_testset(test_dataset, f"./best_models/{NAME}/")
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
        plt.savefig(f"./best_models/{NAME}/confusion_matrix.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("Confusion matrix saved as confusion_matrix.png")


    # clean up
    torch.cuda.empty_cache()
    gc.collect()
    wandb.finish()
    print("Finished.")



if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()