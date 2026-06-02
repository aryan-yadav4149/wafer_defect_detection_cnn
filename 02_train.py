# ===== 1. LIBRARIES =====
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ===== 2. DATA LOAD + CLEAN =====
print("Step 1: Loading dataset...")
df = pd.read_pickle('LSWMD.pkl')

# Wahi clean_label function jo EDA me use kiya
def clean_label(x):
    if isinstance(x, np.ndarray):
        x = x.flatten()
        return x[0] if x.size > 0 and x[0]!= '' else 'none'
    elif isinstance(x, list):
        if len(x) == 0: return 'none'
        elif isinstance(x[0], list): return x[0][0] if len(x[0]) > 0 else 'none'
        elif isinstance(x[0], str): return x[0] if x[0]!= '' else 'none'
        else: return 'none'
    elif isinstance(x, str): return x if x!= '' else 'none'
    elif pd.isna(x): return 'none'
    else: return 'none'

df['failureType'] = df['failureType'].apply(clean_label)

# Khali data hatao
df = df.dropna(subset=['waferMap'])
df = df[df['waferMap'].apply(lambda x: len(x) > 0)]
# Sirf defect wale rakho, 'none' hatao. Kyunki 85% 'none' hai, model bigad jayega
df = df[df['failureType']!= 'none']

print(f"Defect wafers for training: {len(df)}")
print(df['failureType'].value_counts())

# ===== 3. WAFER MAP KO IMAGE ME CONVERT KARO =====
# Problem: Har wafer ka size alag hai. Koi 10x10, koi 50x50
# Solution: Sabko 26x26 resize kar do

def wafer_to_img(wafer_map):
    img = np.array(wafer_map, dtype=np.float32) # List -> Numpy array
    # 26x26 resize karo
    img = tf.image.resize(img[..., np.newaxis], [26, 26]).numpy()
    return img[:,:,0] # (26,26,1) -> (26,26)

print("Step 2: Converting wafers to 26x26 images...")
X = np.array([wafer_to_img(w) for w in df['waferMap']])
# Wafer me values: 0=background, 1=good die, 2=failed die
# 0-1 me normalize karo: 2.0 se divide kar do
X = X / 2.0
X = X[..., np.newaxis] # Shape: (25519, 26, 26) -> (25519, 26, 26, 1)
print(f"X shape: {X.shape}") # CNN ko 4D chahiye: samples, height, width, channels

# ===== 4. LABELS KO NUMBERS ME BADLO =====
# CNN ko 'Center', 'Edge-Ring' samajh nahi aata. Use 0,1,2... chahiye
labels = sorted(df['failureType'].unique()) # ['Center', 'Donut', 'Edge-Loc'...]
label_to_id = {label: i for i, label in enumerate(labels)}
id_to_label = {i: label for label, i in label_to_id.items()}

y = np.array([label_to_id[label] for label in df['failureType']])
# One-hot encoding: 2 -> [0,0,1,0,0,0]
y = tf.keras.utils.to_categorical(y, num_classes=len(labels))

print("Classes:", label_to_id)

# ===== 5. TRAIN-TEST SPLIT =====
# 80% data training ke liye, 20% testing ke liye
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# ===== 6. CNN MODEL BANAO =====
print("\nStep 3: Building CNN model...")
model = models.Sequential([
    # Layer 1: 32 filters, 3x3 size. Image me chhote patterns dhoondta hai
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(26,26,1)),
    layers.MaxPooling2D((2,2)), # Size aadha kar do: 26x26 -> 13x13

    # Layer 2: 64 filters. Ab bade patterns dhoondta hai
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)), # 13x13 -> 6x6

    # Layer 3: 128 filters. Complex patterns
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)), # 6x6 -> 3x3

    layers.Flatten(), # 3x3x128 = 1152 numbers ki list bana do

    layers.Dense(128, activation='relu'), # 128 neurons, dimag ka hissa
    layers.Dropout(0.5), # 50% neurons band kar do. Overfitting rokta hai

    layers.Dense(len(labels), activation='softmax') # 8 output: har defect ka probability
])

model.compile(
    optimizer='adam', # Model ko sikhane wala algo
    loss='categorical_crossentropy', # Kitni galti hui ye nikalta hai
    metrics=['accuracy'] # % sahi bataya wo
)

model.summary() # Model ka structure print karo

# ===== 7. MODEL KO TRAIN KARO =====
print("\nStep 4: Training... Ye 10-15 min lega")
history = model.fit(
    X_train, y_train,
    epochs=20, # 20 baar poora data dekh ke seekhega
    batch_size=128, # Ek baar me 128 images dekh ke seekhega
    validation_data=(X_test, y_test) # Har epoch ke baad test karo
)

# ===== 8. RESULT DEKHO =====
print("\nStep 5: Evaluating...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc*100:.2f}%") # 95%+ aani chahiye

# Confusion Matrix: Kaunsa defect kis se confuse hua
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=labels))

# Confusion matrix plot
plt.figure(figsize=(10,8))
cm = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# Training graph
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.legend()
plt.title('Loss')
plt.savefig('training_history.png')
plt.show()

# ===== 9. MODEL SAVE KARO =====
model.save('wafer_cnn_model.h5')
print("\nModel saved: wafer_cnn_model.h5")
print("CNN Training Complete ✅")