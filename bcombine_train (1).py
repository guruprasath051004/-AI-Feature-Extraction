import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import lightgbm as lgb
import joblib

# Define paths
dino_root = "features_dino"
clip_root = "features_clip"

# Define classes with labels
classes = [
    ("real_2500", 0),
    ("ai_2500", 1),  # your new dataset for real class
]

X_features = []
y_labels = []

for class_name, label in classes:
    dino_dir = os.path.join(dino_root, class_name)
    clip_dir = os.path.join(clip_root, class_name)

    if not os.path.exists(dino_dir) or not os.path.exists(clip_dir):
        continue

    for fname in os.listdir(dino_dir):
        dino_path = os.path.join(dino_dir, fname)
        clip_path = os.path.join(clip_dir, fname)

        if not os.path.exists(dino_path) or not os.path.exists(clip_path):
            print(f"⚠ Skipping {fname}: Feature missing.")
            continue

        try:
            dino_vec = np.load(dino_path)
            clip_vec = np.load(clip_path)
            combined = np.concatenate([dino_vec, clip_vec])
            X_features.append(combined)
            y_labels.append(label)
        except Exception as e:
            print(f"❌ Error loading {fname}: {e}")
            continue

# Train/test split
X = np.array(X_features)
y = np.array(y_labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "saved_model_lgb.pkl")
print("✅ Model saved as saved_model_lgb.pkl")