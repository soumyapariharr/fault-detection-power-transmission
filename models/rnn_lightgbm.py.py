import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import lightgbm as lgb
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, SimpleRNN

# 1. Load and preprocess data
df = pd.read_csv("info.csv")
X_raw = df.drop(columns=["Fault"]).values
y_raw = df["Fault"].values

# 2. Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_raw)

# 3. Create sequences
def create_sequences(data, labels, seq_len=20):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(labels[i+seq_len])
    return np.array(X), np.array(y)

SEQ_LEN = 20
X_seq, y_seq = create_sequences(X_scaled, y_raw, SEQ_LEN)
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq)

# 4. Feature extractor using RNN
def build_rnn_feature_extractor(input_shape):
    inputs = Input(shape=input_shape)
    x = SimpleRNN(64, return_sequences=False)(inputs)
    model = Model(inputs, x)
    return model

# Extract features
feature_model = build_rnn_feature_extractor(X_train.shape[1:])
X_train_feat = feature_model.predict(X_train)
X_test_feat = feature_model.predict(X_test)

# 5. Train LightGBM classifier
lgb_model = lgb.LGBMClassifier(class_weight='balanced')
lgb_model.fit(X_train_feat, y_train)

# 6. Evaluate
y_pred = lgb_model.predict(X_test_feat)
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC-AUC
y_pred_probs = lgb_model.predict_proba(X_test_feat)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_probs)
print(f"ROC-AUC Score: {roc_auc:.4f}")

fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_probs)
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid()
plt.show()

