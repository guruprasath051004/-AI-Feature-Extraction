import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import joblib

# Step 1: Load feature files
clip_features = np.load("F:/ai_image_features2/clip_features.npy")
dino_features = np.load("F:/ai_image_features2/dino_features.npy")
labels = np.load("F:/ai_image_features2/labels.npy")

# Step 2: Combine features (you can also try only CLIP or DINO for comparison)
combined_features = np.concatenate([clip_features, dino_features], axis=1)

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    combined_features, labels, test_size=0.2, random_state=42, stratify=labels
)

# Step 4: Train XGBoost Classifier
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=10,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    use_label_encoder=False,
    eval_metric="mlogloss"
)

print("Training model...")
model.fit(X_train, y_train)

# Step 5: Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 6: Save the model
joblib.dump(model, "image_xgb_classifier.joblib")