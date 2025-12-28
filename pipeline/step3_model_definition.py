# step3_model_definition.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, GRU, Dense, Dropout

def build_bigru_classifier(input_shape):
    model = Sequential([
        Bidirectional(GRU(64, return_sequences=True), input_shape=input_shape),
        Dropout(0.3),
        Bidirectional(GRU(32)),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', 'Precision', 'Recall']
    )
    return model
