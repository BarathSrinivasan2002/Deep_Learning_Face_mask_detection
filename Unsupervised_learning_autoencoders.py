import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.utils import img_to_array
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import xml.etree.ElementTree as ET

import kagglehub

# Download latest version
path = kagglehub.dataset_download("andrewmvd/face-mask-detection")

print("Path to dataset files:", path)

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    annotations = []
    for obj in root.findall('object'):
        filename = root.find('filename').text
        label = obj.find('name').text
        bbox = obj.find('bndbox')
        coords = {
            'xmin': int(bbox.find('xmin').text),
            'ymin': int(bbox.find('ymin').text),
            'xmax': int(bbox.find('xmax').text),
            'ymax': int(bbox.find('ymax').text)
        }
        annotations.append({'filename': filename, 'label': label, 'bbox': coords})
    return annotations

ANNOTATION_PATH = 'C:/Users/matia/.cache/kagglehub/datasets/andrewmvd/face-mask-detection/versions/1/annotations'
xml_files = [os.path.join(ANNOTATION_PATH, filename) for filename in os.listdir(ANNOTATION_PATH) if filename.endswith('.xml')]

data = []
for xml_file in xml_files:
    annotations = parse_xml(xml_file)

    for annotation in annotations:
        data.append([annotation['filename'], annotation['label'], annotation['bbox']])

df = pd.DataFrame(data, columns=['filename', 'label', 'bbox'])
df['label'].value_counts()

def visualize_annotation(image_path, annotations):
    image = cv2.imread(image_path)

    for ann in annotations:
        bbox = ann['bbox']
        cv2.rectangle(image, (bbox['xmin'], bbox['ymin']), (bbox['xmax'], bbox['ymax']), (255, 0, 0), 2)
        cv2.putText(image, ann['label'], (bbox['xmin'], bbox['ymin'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
visualize_annotation('C:/Users/matia/.cache/kagglehub/datasets/andrewmvd/face-mask-detection/versions/1/images/maksssksksss0.png', parse_xml('C:/Users/matia/.cache/kagglehub/datasets/andrewmvd/face-mask-detection/versions/1/annotations/maksssksksss0.xml'))

IMAGE_PATH = 'C:/Users/matia/.cache/kagglehub/datasets/andrewmvd/face-mask-detection/versions/1/images/'
label_pair = {
    'with_mask': 0,
    'without_mask': 1,
    'mask_weared_incorrect': 2
}

face_images = []
face_labels = []

for i in range(len(df)):
    row = df.iloc[i]
    bbox = row['bbox']
    image = cv2.imread(IMAGE_PATH + row['filename'])
    image = image[bbox['ymin']:bbox['ymax'], bbox['xmin']:bbox['xmax']]
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)

    face_images.append(image)
    face_labels.append(label_pair[row['label']])

face_images = np.array(face_images, dtype='float32')
face_labels = np.array(face_labels)

train_x, test_val_x, train_y, test_val_y = train_test_split(face_images, face_labels, test_size=0.3, shuffle=True, stratify=face_labels)
test_x, val_x, test_y, val_y = train_test_split(test_val_x, test_val_y, test_size=0.5, shuffle=True, stratify=test_val_y)

print(train_x.shape, val_x.shape, test_x.shape)
print(train_y.shape, val_y.shape, test_y.shape)

# Build the convolutional autoencoder
input_img = tf.keras.Input(shape=(224, 224, 3))

# Encoder
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
decoded = tf.keras.layers.Conv2D(3, (3, 3), activation='tanh', padding='same')(x)

# Autoencoder model
autoencoder = tf.keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

# Train the autoencoder 
history_autoencoder = autoencoder.fit(train_x, train_x,
                                      epochs=50,
                                      batch_size=32,
                                      shuffle=True,
                                      validation_data=(val_x, val_x))
    
# Print images before and after feature extraction
import random

n = 5
sample_idxs = random.sample(range(test_x.shape[0]), n)
sample_images = test_x[sample_idxs]
reconstructed = autoencoder.predict(sample_images)

plt.figure(figsize=(15, 4))
for i in range(n):
    # Original image
    ax = plt.subplot(2, n, i + 1)
    plt.imshow((sample_images[i] + 1) / 2)  
    plt.title("Original")
    plt.axis("off")
    
    # Reconstructed image
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow((reconstructed[i] + 1) / 2)
    plt.title("Reconstructed")
    plt.axis("off")
plt.show()


# Extract the encoder model
encoder = tf.keras.Model(input_img, encoded)
encoder.summary()

# Build the classifier model using the pre-trained encoder
classifier_input = tf.keras.Input(shape=(224, 224, 3))
features = encoder(classifier_input)
x = tf.keras.layers.GlobalAveragePooling2D()(features)
x = tf.keras.layers.Dense(64, activation='relu')(x)
classifier_output = tf.keras.layers.Dense(3, activation='softmax')(x)

classifier_model = tf.keras.Model(classifier_input, classifier_output)
classifier_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
classifier_model.summary()

# Train the classifier using the training labels
history_classifier = classifier_model.fit(train_x, train_y,
                                          epochs=20,
                                          batch_size=32,
                                          validation_data=(val_x, val_y))

# Evaluate on the test set
test_loss, test_acc = classifier_model.evaluate(test_x, test_y)
print("Test Loss:", test_loss)
print("Test accuracy:", test_acc)

# Generate predictions
y_pred_probs = classifier_model.predict(test_x)
y_pred = np.argmax(y_pred_probs, axis=1)

# Compare predictions with ground truth
label_map = {0: 'with_mask', 1: 'without_mask', 2: 'mask_weared_incorrect'}

print("\nClassification Report:")
print(classification_report(test_y, y_pred, target_names=label_map.values()))

print("Confusion Matrix:")
cm = confusion_matrix(test_y, y_pred)
print(cm)

# Visualization
import random
num_samples = 5
indices = random.sample(range(len(test_x)), num_samples)

plt.figure(figsize=(12, 6))
for i, idx in enumerate(indices):
    plt.subplot(1, num_samples, i + 1)
    plt.imshow((test_x[idx] + 1) / 2)
    true_label = label_map[test_y[idx]]
    pred_label = label_map[y_pred[idx]]
    plt.title(f"T: {true_label}\nP: {pred_label}")
    plt.axis("off")
plt.tight_layout()
plt.show()