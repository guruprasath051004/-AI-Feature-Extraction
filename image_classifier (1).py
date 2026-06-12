import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score
import joblib

# Load features and labels
X_train = np.load("X_train_image.npy")
y_train = np.load("y_train_image.npy")
X_test = np.load("X_test_image.npy")
y_test = np.load("y_test_image.npy")

# Convert to LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Set LightGBM parameters
params = {
    'objective': 'multiclass',
    'num_class': len(np.unique(y_train)),
    'metric': 'multi_logloss',
    'verbosity': -1
}

# Train the model using early stopping with callback
print("Training LightGBM model...")
model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, test_data],
    valid_names=['train', 'valid'],
    num_boost_round=100,
    callbacks=[lgb.early_stopping(stopping_rounds=10)]
)

# Save the model
model.save_model("image_classifier_lgbm.txt")
print("Model saved as image_classifier_lgbm.txt")

# Predict and evaluate
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(y_test, y_pred_labels)
print(f"\n🧠 Test Accuracy: {accuracy:.4f}")