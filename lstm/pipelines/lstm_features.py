import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns
from collections import Counter
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import sys
sys.path.append(r'C:\Users\Mud\Desktop\2ndPaper\lstm')
from model.lstmfcn import generate_lstmfcn
from model.fcn import generate_fcn
from model.lstm import generate_lstm
from utils.class_weight import compute_class_weight
import tensorflow as tf
import random as python_random
from datetime import datetime
import argparse
import pandas as pd



def run(data: str, seed: int):
    # load the data
    if data == "ALL":
        X_test_time = np.load('../../data-preprocess/data-split/data/mean-time-series/test/X_test.npy')
        y_test = np.load('../../data-preprocess/data-split/data/features/test/y_test.npy')
        X_train_time = np.load('../../data-preprocess/data-split/data/mean-time-series/train/X_train.npy')
        y_train = np.load('../../data-preprocess/data-split/data/features/train/y_train.npy')
        X_test_features = np.load('../../data-preprocess/data-split/data/features/test/X_test.npy', allow_pickle=True)
        X_train_features = np.load('../../data-preprocess/data-split/data/features/train/X_train.npy', allow_pickle=True)
    elif data == "CF":
        X_test_time = np.load(r'C:\Users\Mud\Desktop\2ndPaper\lstm\cultivar-specific\data\cf\X_test_cf.npy')
        X_train_time = np.load(r'C:\Users\Mud\Desktop\2ndPaper\lstm\cultivar-specific\data\cf\X_train_cf.npy')
        y_test = np.load(r'C:\Users\Mud\Desktop\2ndPaper\lstm\cultivar-specific\data\cf\y_test_cf.npy')
        y_train = np.load(r'C:\Users\Mud\Desktop\2ndPaper\lstm\cultivar-specific\data\cf\y_train_cf.npy')
        X_test_features = np.load(r'C:\Users\Mud\Desktop\2ndPaper\traditional_ml\weka\data\cf\X_test_cf.npy', allow_pickle=True)
        X_train_features = np.load(r'C:\Users\Mud\Desktop\2ndPaper\traditional_ml\weka\data\cf\X_train_cf.npy', allow_pickle=True)
    elif data == "CON":
        X_test_time = np.load(r'C:\Users\Mud\Desktop\2ndPaper\lstm\cultivar-specific\data\conc\X_test_con.npy')
        X_train_time = np.load(r'C:\Users\Mud\Desktop\2ndPaper\lstm\cultivar-specific\data\conc\X_train_con.npy')
        y_test = np.load(r'C:\Users\Mud\Desktop\2ndPaper\lstm\cultivar-specific\data\conc\y_test_con.npy')
        y_train = np.load(r'C:\Users\Mud\Desktop\2ndPaper\lstm\cultivar-specific\data\conc\y_train_con.npy')
        X_test_features = np.load(r'C:\Users\Mud\Desktop\2ndPaper\traditional_ml\weka\data\conc\X_test_con.npy', allow_pickle=True)
        X_train_features = np.load(r'C:\Users\Mud\Desktop\2ndPaper\traditional_ml\weka\data\conc\X_train_con.npy', allow_pickle=True)
    elif data == "RIES":
        X_test_time = np.load(r'C:\Users\Mud\Desktop\2ndPaper\lstm\cultivar-specific\data\ries\X_test_ries.npy')
        X_train_time = np.load(r'C:\Users\Mud\Desktop\2ndPaper\lstm\cultivar-specific\data\ries\X_train_ries.npy')
        y_test = np.load(r'C:\Users\Mud\Desktop\2ndPaper\lstm\cultivar-specific\data\ries\y_test_ries.npy')
        y_train = np.load(r'C:\Users\Mud\Desktop\2ndPaper\lstm\cultivar-specific\data\ries\y_train_ries.npy')
        X_test_features = np.load(r'C:\Users\Mud\Desktop\2ndPaper\traditional_ml\weka\data\ries\X_test_ries.npy', allow_pickle=True)
        X_train_features = np.load(r'C:\Users\Mud\Desktop\2ndPaper\traditional_ml\weka\data\ries\X_train_ries.npy', allow_pickle=True)
    elif data == "PN":
        X_test_time = np.load(r'C:\Users\Mud\Desktop\2ndPaper\lstm\cultivar-specific\data\pn\X_test_pn.npy')
        X_train_time = np.load(r'C:\Users\Mud\Desktop\2ndPaper\lstm\cultivar-specific\data\pn\X_train_pn.npy')
        y_test = np.load(r'C:\Users\Mud\Desktop\2ndPaper\lstm\cultivar-specific\data\pn\y_test_pn.npy')
        y_train = np.load(r'C:\Users\Mud\Desktop\2ndPaper\lstm\cultivar-specific\data\pn\y_train_pn.npy')
        X_test_features = np.load(r'C:\Users\Mud\Desktop\2ndPaper\traditional_ml\weka\data\pn\X_test_pn.npy', allow_pickle=True)
        X_train_features = np.load(r'C:\Users\Mud\Desktop\2ndPaper\traditional_ml\weka\data\pn\X_train_pn.npy', allow_pickle=True)
    else:
        raise ValueError("Invalid data type. Please choose from 'ALL', 'CF', 'CON', 'RIES', or 'PN'.")
    NAME = f"{datetime.today().strftime('%m%d')}-{data}-lstm-features-dropout0-seed{seed}"
    print(f'{data} data is loaded.')
    print('X_train of time series shape:', X_train_time.shape)
    print('X_test of time series shape:', X_test_time.shape)
    print('X_train of features shape:', X_train_features.shape)
    print('X_test of features shape:', X_test_features.shape)
    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)


    # set the seed
    reset_random_seeds(seed)
    print(f'Seed is set to {seed}.')


    # normalize the time data
    X_train_time = normalize_curves_mean_method(X_train_time)
    X_test_time = normalize_curves_mean_method(X_test_time)

    # normalize the features data
    X_train_features = pd.DataFrame.from_records(X_train_features.flatten())
    X_test_features = pd.DataFrame.from_records(X_test_features.flatten())
    scaler = StandardScaler()
    features = X_train_features.columns
    X_train_features[features] = scaler.fit_transform(X_train_features[features])
    X_test_features[features] = scaler.transform(X_test_features[features])
    # compute mean & std of the training set, then normalize the whole dataset using z-normalization
    mean = np.mean(X_train_time)
    std = np.std(X_train_time)
    X_train_time = (X_train_time - mean) / std
    X_test_time = (X_test_time - mean) / std


    # Ensure data is shaped correctly
    X_train_time = X_train_time.reshape(-1, 1, 600)  # Convert to (samples, 1, timesteps)
    X_test_time = X_test_time.reshape(-1, 1, 600)
    y_train = y_train.reshape(-1, 1)  # Ensure labels are in shape (samples, 1)
    y_test = y_test.reshape(-1, 1)
    X_train_features = X_train_features.values
    X_test_features = X_test_features.values
    print('X_train shape of time series:', X_train_time.shape)
    print('y_train shape:', y_train.shape)
    print('X_test shape of time series:', X_test_time.shape)
    print('y_test shape:', y_test.shape)
    print('X_train shape of features:', X_train_features.shape)
    print('X_test shape of features:', X_test_features.shape)

    # split the data into training and validation set
    X_train_time, X_val_time, X_train_features, X_val_features, y_train, y_val = train_test_split(
        X_train_time, X_train_features, y_train, test_size=0.2, random_state=seed)
    print('X_train shape of time series:', X_train_time.shape)
    print('y_train shape:', y_train.shape)
    print('X_val shape of time series:', X_val_time.shape)
    print('y_val shape:', y_val.shape)
    print('X_train shape of features:', X_train_features.shape)
    print('X_val shape of features:', X_val_features.shape)

    # Train final model on full training data
    final_model = generate_lstm(feature_input_accept=True, dropout=0.0)
    class_weights = compute_class_weight(y_train)
    print('class_weights:', class_weights)
    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=15, mode='auto', factor=1. / np.cbrt(2), cooldown=0, min_lr=5e-7, verbose=2)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True, verbose=1)
    final_model.compile(optimizer=Adam(learning_rate=8e-6), loss='binary_crossentropy', metrics=['accuracy'])


    # Train model
    history = final_model.fit(
        [X_train_time, X_train_features], y_train,
        validation_data=([X_val_time, X_val_features], y_val),
        callbacks=[reduce_lr],
        shuffle=True,
        epochs=1100, batch_size=16, class_weight={0: class_weights[0], 1: class_weights[1]},
        verbose=1
    )

    # Save the model
    final_model.save(f'./results/{NAME}/model.h5')
    print(f'Train completed, and Model is saved to ./results/{NAME}/model.h5')

    # Predict on test set
    y_pred_prob = final_model.predict([X_test_time, X_test_features])
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    conf_matrix = confusion_matrix(y_test, y_pred)


    # plot the metrics
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test ROC AUC Score: {roc_auc:.4f}")
    print('Confusion matrix:\n', conf_matrix)
    result = { "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "roc_auc": roc_auc, "confusion_matrix": conf_matrix }
    with open(f"./results/{NAME}/evaluation_results.txt", "w") as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1 Score: {f1}\n")
        f.write(f"ROC AUC: {roc_auc}\n")
        f.write(f"Confusion Matrix:\n{conf_matrix}\n")
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Dead', 'Alive'], yticklabels=['Dead', 'Alive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'./results/{NAME}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Confusion matrix saved to ./results/{NAME}/confusion_matrix.png')
    


    # plot the training and validation accuracy and loss at each epoch
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure(figsize=(10, 7))
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'./results/{NAME}/loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.figure(figsize=(10, 7))
    plt.plot(epochs, acc, 'y', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'./results/{NAME}/accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()



def reset_random_seeds(seed=42):
   np.random.seed(seed)
   python_random.seed(seed)
   tf.random.set_seed(seed)

# normalize the data to have same starting pt for all time series
# input shape: (n_samples, n_timepoints), want each samples - its mean of 0-30th timepts
def normalize_curves_mean_method(unnormalized_datas):
    normalized_datas = []
    for unnormalized_data in unnormalized_datas:
        mean = np.mean(unnormalized_data[:30])
        normalized_data = unnormalized_data - mean
        normalized_datas.append(normalized_data)
    normalized_datas = np.array(normalized_datas)
    assert normalized_datas.shape == unnormalized_datas.shape
    return normalized_datas

# normalize the data using z-score, have mean 0 and std 1
def normalize_curves_zscore(unnormalized_datas):
    normalized_datas = []
    for unnormalized_data in unnormalized_datas:
        mean = np.mean(unnormalized_data)
        std = np.std(unnormalized_data)
        normalized_data = (unnormalized_data - mean) / std
        normalized_datas.append(normalized_data)
    normalized_datas = np.array(normalized_datas)
    assert normalized_datas.shape == unnormalized_datas.shape
    return normalized_datas


# apply ln() to the time series data
def apply_ln_to_curves(datas):
    ln_datas = []
    for data in datas:
        ln_data = np.log(data)
        ln_datas.append(ln_data)
    ln_datas = np.array(ln_datas)
    assert ln_datas.shape == datas.shape
    return ln_datas


# split the time series data into wanted time points
def split_time_series(datas, start, end):
    datas = datas[:, start:end]
    return datas


# plot the time series data, randomly picked n samples, different colors represent different classes
def plot_random_time_series(datas, n, labels):
    for i in range(n):
        idx = np.random.randint(datas.shape[0])
        plt.plot(datas[idx], color = 'blue' if labels[idx] == 1 else 'red', alpha = 0.2)
    plt.show()

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