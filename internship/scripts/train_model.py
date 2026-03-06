import os, argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from tensorflow.keras import layers, models

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='Dataset')
parser.add_argument('--epochs', type=int, default=20)
args = parser.parse_args()

df = pd.read_csv(os.path.join(args.dataset_dir, 'breathing_dataset.csv'))
labels = sorted(df.label.unique())
lmap   = {l:i for i,l in enumerate(labels)}
df['y'] = df.label.map(lmap)

feat_cols = [c for c in df.columns if c not in ['participant','window_start','label','y']]
participants = sorted(df.participant.unique())

def build_model(n_classes):
    model = models.Sequential([
        layers.Conv1D(16, 7, padding='same', activation='relu', input_shape=(120, 3)),
        layers.MaxPooling1D(2),
        layers.Conv1D(32, 5, padding='same', activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

all_pred, all_true = [], []

for test_p in participants:
    tr = df[df.participant != test_p]
    te = df[df.participant == test_p]

    # shape is (N, 120, 3) for Conv1D in keras
    Xtr = tr[feat_cols].values.reshape(-1, 120, 3)
    ytr = tr.y.values
    Xte = te[feat_cols].values.reshape(-1, 120, 3)
    yte = te.y.values

    model = build_model(len(labels))
    model.fit(Xtr, ytr, epochs=args.epochs, batch_size=64, verbose=0)

    preds = model.predict(Xte, verbose=0).argmax(axis=1)
    print(f'{test_p} accuracy: {accuracy_score(yte, preds):.4f}')
    all_pred.extend(preds)
    all_true.extend(yte)

all_pred = np.array(all_pred)
all_true = np.array(all_true)

print('\n=== Overall Results ===')
print(f'Accuracy  : {accuracy_score(all_true, all_pred):.4f}')
print(f'Precision : {precision_score(all_true, all_pred, average="weighted", zero_division=0):.4f}')
print(f'Recall    : {recall_score(all_true, all_pred, average="weighted", zero_division=0):.4f}')
cm = pd.DataFrame(confusion_matrix(all_true, all_pred), index=labels, columns=labels)
print('\nConfusion Matrix:')
print(cm)
