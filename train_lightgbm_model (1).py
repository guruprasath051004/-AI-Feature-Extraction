import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

#Load extracted features and labels

X = np.load("F:/dinov2_features/features.npy")
y = np.load("F:/dinov2_features/labels.npy")

#Train/test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#LightGBM classifier

model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31)
model.fit(X_train, y_train)

#Evaluation

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("\n✅ Model trained successfully!")
print(f"Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Real", "AI-Generated"]))

#Save the trained model

joblib.dump(model, "F:/dinov2_features/lightgbm_model.pkl")
print("\n💾 Model saved as: F:/dinov2_features/lightgbm_model.pkl")