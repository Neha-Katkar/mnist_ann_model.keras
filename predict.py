
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD TRAINED MODEL
# ==========================================

model = tf.keras.models.load_model("mnist_ann_model.keras")

# ==========================================
# 2. LOAD MNIST TEST DATA
# ==========================================

(_, _), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# ==========================================
# 3. SELECT ONE IMAGE
# ==========================================

image_index = 0

image = X_test[image_index]
actual_digit = y_test[image_index]

# ==========================================
# 4. PREPROCESS IMAGE
# ==========================================

image_normalized = image.astype("float32") / 255.0

image_input = np.expand_dims(image_normalized, axis=0)

# ==========================================
# 5. MAKE PREDICTION
# ==========================================

predictions = model.predict(image_input, verbose=0)

predicted_digit = np.argmax(predictions[0])
confidence = np.max(predictions[0]) * 100

# ==========================================
# 6. DISPLAY RESULT
# ==========================================

print("Actual Digit:", actual_digit)
print("Predicted Digit:", predicted_digit)
print(f"Confidence: {confidence:.2f}%")

plt.imshow(image, cmap="gray")
plt.title(f"Predicted: {predicted_digit} | Actual: {actual_digit}")
plt.axis("off")
plt.show()
