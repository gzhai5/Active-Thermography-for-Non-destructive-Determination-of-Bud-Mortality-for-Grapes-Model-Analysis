import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_run_folder_name(name):
    # Example name: 0502-ALL-logistic-bestparams-seed349
    dataset = re.search(r'-(RIES|CF|CON|PN|ALL)-', name)
    seed = re.search(r'seed(\d+)', name)
    model = re.search(r'-(logistic|svm|randomforest|xgb)-', name)
    
    return {
        "dataset": dataset.group(1) if dataset else None,
        "seed": int(seed.group(1)) if seed else None,
        "model": model.group(1) if model else None,
    }

def parse_evaluation_file(file_path):
    with open(file_path, "r") as f:
        lines = f.read().splitlines()
    
    metrics = {}
    cm_found = False
    for line in lines:
        line = line.strip()

        if line.lower().startswith("confusion matrix") and not cm_found:
            # Look for two lines of the matrix
            idx = lines.index(line)
            try:
                cm = []
                for i in range(1, 3):
                    clean_line = re.sub(r"[\[\]]", "", lines[idx + i].strip())
                    cm.append(list(map(int, clean_line.split())))
                metrics["TN"], metrics["FP"] = cm[0]
                metrics["FN"], metrics["TP"] = cm[1]
                cm_found = True
            except Exception:
                pass  # Gracefully continue if format is off
        elif ":" in line:
            key_val = line.split(": ", 1)
            key, val = key_val[0].strip(), key_val[1].strip()
            if key == "Best Parameters":
                metrics["Best Parameters"] = val
            else:
                try:
                    metrics[key] = float(val)
                except ValueError:
                    continue
    return metrics

def collect_results(parent_dir):
    rows = []
    for run_folder in os.listdir(parent_dir):
        full_path = os.path.join(parent_dir, run_folder)
        if os.path.isdir(full_path):
            eval_file = os.path.join(full_path, "results.txt")
            if os.path.exists(eval_file):
                params = parse_run_folder_name(run_folder)
                metrics = parse_evaluation_file(eval_file)
                row = {**params, **metrics, "run_name": run_folder}
                rows.append(row)
    return pd.DataFrame(rows)

# === Run analysis ===
parent_dir = r"C:\Users\Mud\Desktop\2ndPaper\traditional_ml\pipelines\results"  # Replace this with your actual path
df = collect_results(parent_dir)

# === Save to CSV ===
df.to_csv("summary_results.csv", index=False)