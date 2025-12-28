# step4_train_and_evaluate.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from step3_model_definition import build_bigru_classifier

# Load sequences
X = np.load("X_sequences.npy")
y = np.load("y_sequences.npy")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Define model
model = build_bigru_classifier(input_shape=(X.shape[1], X.shape[2]))

# Use class weights to handle imbalance
class_weights = {0: 1.0, 1: 10.0}

# Train
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    class_weight=class_weights
)

# Save model
model.save("bigru_classifier_model.keras")

# Evaluate
y_pred = (model.predict(X_test) > 0.5).astype(int)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Fault'], yticklabels=['Normal', 'Fault'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

# After training
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig("loss_plot.png")
plt.close()