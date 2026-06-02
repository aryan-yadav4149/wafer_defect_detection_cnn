import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Model Load Karo
print("Loading model...")
model = tf.keras.models.load_model('wafer_cnn_model.h5')

# 2. Labels - Same order jo training me tha
labels = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch']

def predict_wafer_defect(wafer_map, show_plot=True):
    """
    Koi bhi wafer map do, defect bata dega
    """
    # Preprocess - Wahi jo training me kiya
    img = np.array(wafer_map, dtype=np.float32)
    img = tf.image.resize(img[..., np.newaxis], [26, 26]).numpy()
    img = img / 2.0
    img = img[np.newaxis,..., np.newaxis] # (1, 26, 26, 1)

    # Predict
    pred = model.predict(img, verbose=0)[0]
    pred_class = np.argmax(pred)
    confidence = pred[pred_class] * 100

    # Top 3 predictions
    top3_idx = np.argsort(pred)[-3:][::-1]

    print(f"\n=== PREDICTION RESULT ===")
    print(f"Predicted Defect: {labels[pred_class]}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"\nTop 3 Probabilities:")
    for idx in top3_idx:
        print(f" {labels[idx]:12s}: {pred[idx]*100:.2f}%")

    # Plot
    if show_plot:
        plt.figure(figsize=(10,4))

        plt.subplot(1,2,1)
        plt.imshow(wafer_map, cmap='viridis')
        plt.title(f'Input Wafer\nPred: {labels[pred_class]} ({confidence:.1f}%)')
        plt.axis('off')

        plt.subplot(1,2,2)
        plt.barh(labels, pred*100)
        plt.xlabel('Probability %')
        plt.title('All Class Probabilities')
        plt.xlim(0, 100)

        plt.tight_layout()
        plt.savefig('prediction_result.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("\nSaved: prediction_result.png")

    return labels[pred_class], confidence

# ===== TEST KARO =====
print("\nTesting on sample wafers from dataset...")
df = pd.read_pickle('LSWMD.pkl')

# Test 1: Random wafer
test_idx = 100
test_wafer = df.iloc[test_idx]['waferMap']
actual_defect = df.iloc[test_idx]['failureType']

print(f"\n--- Test Wafer #{test_idx} ---")
print(f"Actual Defect: {actual_defect}")
predicted, conf = predict_wafer_defect(test_wafer)

# Test 2: Ek aur wafer
test_idx = 500
test_wafer = df.iloc[test_idx]['waferMap']
actual_defect = df.iloc[test_idx]['failureType']

print(f"\n--- Test Wafer #{test_idx} ---")
print(f"Actual Defect: {actual_defect}")
predicted, conf = predict_wafer_defect(test_wafer)

print("\n✅ Prediction Complete")