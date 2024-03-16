import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Function to preprocess images
def preprocess_image(image_path):
    image = Image.open(image_path).convert('L').resize((300, 300))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# Function to evaluate models
def evaluate_model(model, classes, samples_dir):
    # Folder containing the images
    image_folder = samples_dir
    print(f"Processing images from folder: {os.path.basename(image_folder)}")
    # Get a list of all files in the folder
    image_paths = [os.path.join(root, file) for root, dirs, files in os.walk(image_folder) for file in files if file.endswith(('png', 'jpg', 'jpeg'))]

    # Load and preprocess the images, converting to grayscale
    images = [Image.open(path).convert('L').resize((300, 300)) for path in image_paths]
    image_arrays = [np.array(img) / 255.0 for img in images]
    image_arrays = np.array(image_arrays)

    # Add a channel dimension if the model expects input shape (None, 300, 300, 1)
    if model.input_shape[-1] == 1:
        image_arrays = np.expand_dims(image_arrays, axis=-1)

    # Make batch predictions
    predictions_batch = model.predict(image_arrays)
    print(f"Predictions made for {len(predictions_batch)} images")

    # Process predictions for each image
    predicted_labels = []
    for i, predictions in enumerate(predictions_batch):
        all_predicted_labels[i] = all_predicted_labels[i] + classes[np.argmax(predictions)]
        predicted_labels.append(classes[np.argmax(predictions)])

    # True labels
    true_labels = []
    for i, path in enumerate(image_paths):
        true_label = os.path.normpath(path).split(os.path.sep)[-2][mega_i]
        all_true_labels[i] = all_true_labels[i] + true_label
        true_labels.append(true_label)

    return true_labels, predicted_labels

# Set paths to the test folder, models, and class labels
model_path = r'C:\Users\PlicEduard\AI4_SOC\models'

test_folder = os.path.join(model_path, 'test')
model_paths = [os.path.join(model_path, f"{i}.h5") for i in range(1, 4)]
class_labels = [['A', 'B', 'C', 'D', 'E', 'F', 'H'], ['a', 'h', 'k', 'r', 's', 'x'], ['c', 'i', 'o', 'x']]

# Load models
print("Loading models...")
models = [load_model(model_path) for model_path in model_paths]
print("Models loaded.")

# Get a list of all files in the test folder
image_paths = [os.path.join(root, file) for root, dirs, files in os.walk(test_folder) for file in files if file.endswith(('png', 'jpg', 'jpeg'))]
print(f"Total images found: {len(image_paths)}")

all_true_labels = ['' for _ in range(len(image_paths))]
all_predicted_labels = ['' for _ in range(len(image_paths))]

# Make predictions for each model
for i, model in enumerate(models):
    mega_i = i
    print(f"Evaluating model {i + 1}...")
    true_labels, predicted_labels = evaluate_model(model, class_labels[i], samples_dir=test_folder)

# Print all_true_labels and all_predicted_labels
print(all_true_labels)
print(all_predicted_labels)


import itertools
# Generate all possible 3-letter combinations
all_combinations = [''.join(comb) for comb in itertools.product(*class_labels)]

# Initialize confusion matrix
cm = np.zeros((len(all_combinations), len(all_combinations)), dtype=int)

# Iterate over true and predicted labels to update confusion matrix
for true_label, pred_label in zip(all_true_labels, all_predicted_labels):
    true_index = all_combinations.index(true_label)
    pred_index = all_combinations.index(pred_label)
    cm[true_index][pred_index] += 1

# Print confusion matrix
print("Confusion Matrix:")
print(cm)
# Plot confusion matrix
plt.figure(figsize=(10, 8))
plt.matshow(cm, cmap=plt.cm.Greens)
plt.colorbar()
plt.xlabel('True Label', fontsize=12)  # Customize x-axis label
plt.ylabel('Predicted Label', fontsize=12)  # Customize y-axis label
plt.title('Confusion Matrix - Final Data', fontsize=18)  # Customize title
plt.show()
