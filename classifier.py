# Project 2: Data Classification Using AI
# DecodeLabs Industrial Training - Batch 2026

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import numpy as np

def main():
    # --- PHASE 1: LOAD DATA ---
    print("=" * 50)
    print("PROJECT 2: DATA CLASSIFICATION USING AI")
    print("=" * 50)

    iris = load_iris()
    X = iris.data
    y = iris.target

    print(f"\n[INPUT] Dataset Loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"[INPUT] Classes: {iris.target_names}")

    # --- PHASE 2: SCALE FEATURES ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("\n[PROCESS] Feature scaling applied (Mean=0, Variance=1)")

    # --- PHASE 3: TRAIN-TEST SPLIT ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, shuffle=True
    )
    print(f"[PROCESS] Train size: {len(X_train)} | Test size: {len(X_test)}")

    # --- PHASE 4: TRAIN KNN MODEL ---
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    print("\n[PROCESS] KNN Model trained (k=5)")

    # --- PHASE 5: PREDICT & EVALUATE ---
    predictions = model.predict(X_test)

    accuracy = round(model.score(X_test, y_test) * 100, 2)
    f1 = round(f1_score(y_test, predictions, average='weighted'), 4)
    cm = confusion_matrix(y_test, predictions)

    print("\n" + "=" * 50)
    print("OUTPUT: MODEL EVALUATION")
    print("=" * 50)
    print(f"\nAccuracy  : {accuracy}%")
    print(f"F1 Score  : {f1}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nDetailed Report:")
    print(classification_report(y_test, predictions, target_names=iris.target_names))

if __name__ == "__main__":
    main()
