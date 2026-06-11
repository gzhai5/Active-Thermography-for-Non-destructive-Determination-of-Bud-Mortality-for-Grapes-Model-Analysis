import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, roc_auc_score, recall_score, precision_score, make_scorer
from imblearn.over_sampling import SMOTE
from datetime import datetime



def run(data: str, seed: int):
    # load the data
    if data == 'ALL':
        X_train = np.load(f'../../data-preprocess/data-split/data/features/train/X_train.npy', allow_pickle=True)
        y_train = np.load(f'../../data-preprocess/data-split/data/features/train/y_train.npy')
        X_test = np.load(f'../../data-preprocess/data-split/data/features/test/X_test.npy', allow_pickle=True)
        y_test = np.load(f'../../data-preprocess/data-split/data/features/test/y_test.npy')
    elif data == 'CF':
        X_train = np.load(f'../weka/data/cf/X_train_cf.npy', allow_pickle=True)
        y_train = np.load(f'../weka/data/cf/y_train_cf.npy')
        X_test = np.load(f'../weka/data/cf/X_test_cf.npy', allow_pickle=True)
        y_test = np.load(f'../weka/data/cf/y_test_cf.npy')
    elif data == 'CON':
        X_train = np.load(f'../weka/data/conc/X_train_con.npy', allow_pickle=True)
        y_train = np.load(f'../weka/data/conc/y_train_con.npy')
        X_test = np.load(f'../weka/data/conc/X_test_con.npy', allow_pickle=True)
        y_test = np.load(f'../weka/data/conc/y_test_con.npy')
    elif data == 'PN':
        X_train = np.load(f'../weka/data/pn/X_train_pn.npy', allow_pickle=True)
        y_train = np.load(f'../weka/data/pn/y_train_pn.npy')
        X_test = np.load(f'../weka/data/pn/X_test_pn.npy', allow_pickle=True)
        y_test = np.load(f'../weka/data/pn/y_test_pn.npy')
    elif data == 'RIES':
        X_train = np.load(f'../weka/data/ries/X_train_ries.npy', allow_pickle=True)
        y_train = np.load(f'../weka/data/ries/y_train_ries.npy')
        X_test = np.load(f'../weka/data/ries/X_test_ries.npy', allow_pickle=True)
        y_test = np.load(f'../weka/data/ries/y_test_ries.npy')
    else:
        raise ValueError("Invalid data type. Choose from 'ALL', 'CF', 'CON', 'PN', 'RIES'.")
    NAME = f"{datetime.today().strftime('%m%d')}-{data}-randomforest-bestparams-seed{seed}"
    print(f'{data} data loaded.')
    print(f'X_train shape: {X_train.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'X_test shape: {X_test.shape}')
    print(f'y_test shape: {y_test.shape}')


    # normalize the data is unnecessary for random forest
    X_train = pd.DataFrame.from_records(X_train.flatten())
    X_test = pd.DataFrame.from_records(X_test.flatten())


    best_params = {
        'ALL': { 
            'max_depth': 1000,
            'max_features': 'sqrt',
            'n_estimators': 1000,
            'min_samples_split': 10,
            'min_samples_leaf': 4,
            },
        'CF': { 
            'max_depth': 729,
            'max_features': 'log2',
            'n_estimators': 100,
            'min_samples_split': 2,
            'min_samples_leaf': 2, 
            },
        'CON': { 
            'max_depth': 3,
            'max_features': 'log2',
            'n_estimators': 100,
            'min_samples_split': 10,
            'min_samples_leaf': 4,
            },
        'PN': { 
            'max_depth': 4,
            'max_features': 'sqrt',
            'n_estimators': 264,
            'min_samples_split': 4,
            'min_samples_leaf': 4,
            },
        'RIES': { 
            'max_depth': 3,
            'max_features': 'log2',
            'n_estimators': 1000,
            'min_samples_split': 10,
            'min_samples_leaf': 1, 
            }
    }


    # val split to match other pipelines
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)

    # Initialize model with balanced class weights and best parameters
    model = RandomForestClassifier(class_weight='balanced', random_state=seed, **best_params[data])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate the metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_prob)

    # Print the results
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"AUC-ROC: {auc_roc}")
    print(f'confusion matrix: {confusion_matrix(y_test, y_pred)}')
    print(f"Best Parameters: {best_params[data]}")
    results = {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'auc_roc': auc_roc,
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'best_params': best_params[data]
    }
    os.makedirs(f'./results/{NAME}', exist_ok=True)
    with open(f'./results/{NAME}/results.txt', 'w') as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"F1 Score: {f1}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"ROC AUC: {auc_roc}\n")
        f.write(f'confusion matrix: {confusion_matrix(y_test, y_pred)}\n')
        f.write(f"Best Parameters: {best_params[data]}\n")
        f.write('------------------------------------\n')
        f.write('Confusion Matrix:\n')
        f.write(str(confusion_matrix(y_test, y_pred)) + '\n')
    print('------------------------------------')

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Dead", "Alive"])
    plt.figure(figsize=(6, 6))
    disp.plot(cmap="Blues")
    plt.savefig(f'./results/{NAME}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    feature_importances = model.feature_importances_
    plt.figure(figsize=(8, 6))
    plt.barh(X_train.columns, feature_importances)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importances from Logistic Regression')
    plt.savefig(f'./results/{NAME}/feature_importances.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line (random chance)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(f'./results/{NAME}/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='ALL', help='Data type: ALL, CF, CON, RIES, PN')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for initialization')
    args = parser.parse_args()
    seed = args.seed if args.seed else 42
    data = args.data if args.data else 'ALL'
    run(data, seed)


if __name__ == "__main__":
    main()