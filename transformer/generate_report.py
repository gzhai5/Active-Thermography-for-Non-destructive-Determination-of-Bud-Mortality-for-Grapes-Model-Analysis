import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_run_folder_name(name):
    # Example name: 0501-frame60-ALL-21-N-bs16-weight-5e-06-epoch8-cos-seed349
    dataset = re.search(r'-(RIES|CF|CON|PN|ALL)-', name)
    batch_size = re.search(r'bs(\d+)', name)
    lr = re.search(r'weight-(\d*\.?\d+)', name)
    seed = re.search(r'seed(\d+)', name)
    
    return {
        "dataset": dataset.group(1) if dataset else None,
        "batch_size": int(batch_size.group(1)) if batch_size else None,
        "learning_rate": float(lr.group(1)) if lr else None,
        "seed": int(seed.group(1)) if seed else None,
    }

def parse_evaluation_file(file_path):
    with open(file_path, "r") as f:
        lines = f.read().splitlines()
    
    metrics = {}
    for line in lines:
        if "Confusion Matrix" in line:
            idx = lines.index(line)
            cm = []
            for i in range(1, 3):
                clean_line = re.sub(r"[\[\]]", "", lines[idx + i]).strip()
                cm.append(list(map(int, clean_line.split())))
            metrics["TN"], metrics["FP"] = cm[0]
            metrics["FN"], metrics["TP"] = cm[1]
            break
        else:
            key_val = line.split(": ")
            if len(key_val) == 2:
                metrics[key_val[0].strip()] = float(key_val[1].strip())
    return metrics

def collect_results(parent_dir):
    rows = []
    for run_folder in os.listdir(parent_dir):
        full_path = os.path.join(parent_dir, run_folder)
        if os.path.isdir(full_path):
            eval_file = os.path.join(full_path, "evaluation_results.txt")
            if os.path.exists(eval_file):
                params = parse_run_folder_name(run_folder)
                metrics = parse_evaluation_file(eval_file)
                row = {**params, **metrics, "run_name": run_folder}
                rows.append(row)
    return pd.DataFrame(rows)

# === Run analysis ===
parent_dir = r"C:\Users\Mud\Desktop\2ndPaper\transformer\best_models"  # Replace this with your actual path
df = collect_results(parent_dir)

# === Save to CSV ===
df.to_csv("summary_results.csv", index=False)