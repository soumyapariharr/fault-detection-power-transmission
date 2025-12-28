# step1_data_cleaning.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load balanced dataset
df = pd.read_csv("balanced_smart_grid_dataset.csv")

# Create final target column and drop the two fault indicators
df['Fault'] = ((df['Overload Condition'] == 1) | (df['Transformer Fault'] == 1)).astype(int)
df.drop(columns=['Overload Condition', 'Transformer Fault'], inplace=True)

# Normalize features
features = df.drop(columns=['Fault'])
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(features)

# Final dataset
X = normalized_features
y = df['Fault'].values

# Save preprocessed data
np.save("X_clean.npy", X)
np.save("y_labels.npy", y)
print("Step 1 complete: Cleaned and normalized features saved.")
