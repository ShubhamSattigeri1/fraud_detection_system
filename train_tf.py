"""
train_tf.py
───────────
Train a TensorFlow/Keras neural network for fraud detection.
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

print("Loading data...")
df = pd.read_csv("DATA/creditcard.csv")
features = [f"V{i}" for i in range(1, 29)]
X = df[features]
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Applying SMOTE...")
smote = SMOTE(sampling_strategy="auto", random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {X_train_sm.shape}")

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), features)]
)

print("Preprocessing...")
X_train_proc = preprocessor.fit_transform(X_train_sm)
X_test_proc = preprocessor.transform(X_test)

# Validation split
X_train_proc, X_val_proc, y_train_sm, y_val = train_test_split(
    X_train_proc, y_train_sm, test_size=0.15, random_state=42, stratify=y_train_sm
)

input_dim = X_train_proc.shape[1]

# Build model
model = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

# Class weights
class_weight = {0: 1.0, 1: len(y_train_sm[y_train_sm==0]) / len(y_train_sm[y_train_sm==1])}

print("Training...")
history = model.fit(
    X_train_proc, y_train_sm,
    validation_data=(X_val_proc, y_val),
    epochs=20,
    batch_size=1024,
    class_weight=class_weight,
    verbose=1
)

# Evaluate
print("\nEvaluating on test set...")
y_proba = model.predict(X_test_proc, verbose=0).flatten()
y_pred = (y_proba > 0.5).astype(int)

from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
print(classification_report(y_test, y_pred, digits=4))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
print(f"PR-AUC: {auc(recall_curve, precision_curve):.4f}")

# Save model and preprocessor
model.save("tf_model.keras")
joblib.dump(preprocessor, "tf_preprocessor.pkl")
print("\nSaved tf_model.keras and tf_preprocessor.pkl")