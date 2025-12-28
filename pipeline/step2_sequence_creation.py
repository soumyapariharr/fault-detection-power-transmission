# step2_sequence_creation.py
import numpy as np

# Load cleaned data
X = np.load("X_clean.npy")
y = np.load("y_labels.npy")

SEQUENCE_LENGTH = 30

X_seq = []
y_seq = []

for i in range(len(X) - SEQUENCE_LENGTH):
    X_seq.append(X[i:i+SEQUENCE_LENGTH])
    y_seq.append(y[i+SEQUENCE_LENGTH - 1])  # Label at the end of the sequence

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# Save
np.save("X_sequences.npy", X_seq)
np.save("y_sequences.npy", y_seq)
print(f"Step 2 complete: Created {X_seq.shape[0]} sequences of shape {X_seq.shape[1:]}")
