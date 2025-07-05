import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import json
import os

# ------------------------------
# 1️⃣ 参数配置
# ------------------------------
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
MODEL_OUTPUT_PATH = "gw_cnn_classifier.h5"

# ------------------------------
# 2️⃣ 加载数据
# ------------------------------
print("Loading data...")

X_train = np.load('train_spectrogram_X.npy')
y_train = np.load('train_spectrogram_y.npy')
X_val = np.load('validation_spectrogram_X.npy')
y_val = np.load('validation_spectrogram_y.npy')

print(f"Train samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

# ------------------------------
# 3️⃣ 加载元数据并改进 sample_weight
# ------------------------------
def compute_weight(item):
    if item['type'] == 'noise' or item['params'] is None:
        return 1.0
    else:
        p = item['params']
        m1 = p.get('mass1', 30.0)
        m2 = p.get('mass2', 30.0)
        d = p.get('distance', 500.0)
        if d <= 0.0:
            d = 500.0

        # Chirp mass
        chirp_mass = ((m1 * m2)**(3/5)) / ((m1 + m2)**(1/5))
        raw = (chirp_mass ** (5/3)) / d
        # 对数压缩+归一化
        weight = np.log1p(raw * 1.0) / np.log1p(1.0)
        return max(min(weight, 1.0), 0.05)  # 防止太小

print("Loading injection metadata for weights...")

with open('injection_metadata.json', 'r') as f:
    metadata_train = json.load(f)
train_weights = np.array([compute_weight(item) for item in metadata_train], dtype=np.float32)

with open('injection_metadata_val.json', 'r') as f:
    metadata_val = json.load(f)
val_weights = np.array([compute_weight(item) for item in metadata_val], dtype=np.float32)

print(f"Sample weights constructed. Train shape: {train_weights.shape}, Validation shape: {val_weights.shape}")

# ------------------------------
# 4️⃣ 数据形状检查和归一化
# ------------------------------
if len(X_train.shape) == 3:
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]

y_train = y_train.astype(np.float32)
y_val = y_val.astype(np.float32)

# ------------------------------
# 5️⃣ 定义更深的CNN
# ------------------------------
def build_improved_cnn(input_shape):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))

    # Block 1
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Block 2
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Block 3
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.4))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

model = build_improved_cnn(X_train.shape[1:])
model.summary()

# ------------------------------
# 6️⃣ 编译模型
# ------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# ------------------------------
# 7️⃣ 定义回调
# ------------------------------
# early_stop = callbacks.EarlyStopping(
#     monitor='val_loss',
#     patience=5,
#     restore_best_weights=True
# )

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-5
)

# ------------------------------
# 8️⃣ 训练模型
# ------------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val, val_weights),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2,
    sample_weight=train_weights,
    # callbacks=[early_stop, reduce_lr]
    callbacks = [reduce_lr]
)

# ------------------------------
# 9️⃣ 评估模型
# ------------------------------
print("\nEvaluating on validation set:")
results = model.evaluate(X_val, y_val, sample_weight=val_weights, verbose=2)
print(f"Validation Loss: {results[0]:.4f}, Accuracy: {results[1]:.4f}, AUC: {results[2]:.4f}")

# ------------------------------
# 🔟 可视化
# ------------------------------
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# ------------------------------
# 🔟 保存模型
# ------------------------------
model.save(MODEL_OUTPUT_PATH)
print(f"✅ Model saved to {MODEL_OUTPUT_PATH}")

metadata = {
    "epochs": len(history.history['loss']),
    "batch_size": BATCH_SIZE,
    "input_shape": X_train.shape[1:],
    "model_file": MODEL_OUTPUT_PATH
}

with open('training_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("✅ Metadata saved to training_metadata.json")