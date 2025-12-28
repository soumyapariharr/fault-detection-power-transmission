import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve

from tensorflow.keras.layers import Input, LSTM, RepeatVector
from tensorflow.keras.models import Model

# Load and Preprocess
df = pd.read_csv("info.csv")
X_raw = df.drop(columns=["Fault"]).values
y_raw = df["Fault"].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_raw)

def create_sequences(data, labels, seq_len=20):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(labels[i+seq_len])
    return np.array(X), np.array(y)

SEQ_LEN = 20
X_seq, y_seq = create_sequences(X_scaled, y_raw, SEQ_LEN)
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq)

# LSTM Autoencoder
def build_autoencoder(input_shape):
    inputs = Input(shape=input_shape)
    encoded = LSTM(64, return_sequences=False)(inputs)
    decoded = RepeatVector(input_shape[0])(encoded)
    decoded = LSTM(input_shape[1], return_sequences=True)(decoded)
    model = Model(inputs, decoded)
    model.compile(optimizer='adam', loss='mse')
    return model

autoencoder = build_autoencoder(X_train.shape[1:])
autoencoder.fit(X_train, X_train, epochs=10, batch_size=64, validation_split=0.1)

# Encoder Model
encoder = Model(autoencoder.input, autoencoder.layers[1].output)
X_train_encoded = encoder.predict(X_train)
X_test_encoded = encoder.predict(X_test)

# OC-SVM
oc_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
oc_svm.fit(X_train_encoded)

y_pred = oc_svm.predict(X_test_encoded)
y_pred = np.where(y_pred == -1, 1, 0)

print("LSTM Autoencoder + OC-SVM Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# For ROC and Precision-Recall: use decision_function scores from OC-SVM
y_scores = oc_svm.decision_function(X_test_encoded)

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_scores)
print(f"ROC-AUC Score: {roc_auc:.4f}")

fpr, tpr, _ = roc_curve(y_test, y_scores)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_scores)
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid()
plt.show()
