
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD MNIST DATASET
# ==========================================

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Training images:", X_train.shape)
print("Testing images:", X_test.shape)

# ==========================================
# 2. DISPLAY A SAMPLE IMAGE
# ==========================================

plt.imshow(X_train[0], cmap="gray")
plt.title(f"Actual Digit: {y_train[0]}")
plt.axis("off")
plt.show()

# ==========================================
# 3. NORMALIZE IMAGE PIXELS
# ==========================================

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# ==========================================
# 4. CREATE ANN MODEL
# ==========================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

# ==========================================
# 5. COMPILE MODEL
# ==========================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================================
# 6. TRAIN MODEL
# ==========================================

model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2
)

# ==========================================
# 7. EVALUATE MODEL
# ==========================================

test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy * 100, "%")

# ==========================================
# 8. SAVE TRAINED MODEL
# ==========================================

model.save("mnist_ann_model.keras")

print("Model saved successfully!")
