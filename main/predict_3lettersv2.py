import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from collections import defaultdict

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

test_folder = os.path.join(model_path, 'test3')
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


# Initialize dictionaries to count occurrences of combinations
true_label_counts = defaultdict(int)
predicted_label_counts = defaultdict(int)

# Count occurrences of combinations in true labels
for label in all_true_labels:
    true_label_counts[label] += 1

# Count occurrences of combinations in predicted labels
for label in all_predicted_labels:
    predicted_label_counts[label] += 1

# Get unique combinations from true labels and predicted labels
unique_combinations = set(all_true_labels + all_predicted_labels)

# Convert the set back to a list if needed
unique_combinations_list = list(unique_combinations)
unique_combinations_list = sorted(unique_combinations_list)

# Print the list of unique combinations
print("Unique combinations:", unique_combinations_list)

# Generate confusion matrix
conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)

# Print confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
# Plot confusion matrix
plt.figure(figsize=(2, 2))
plt.matshow(conf_matrix, cmap=plt.cm.Greens)
plt.colorbar()

# Set ticks along x and y axes using unique_combinations_list
tick_indices = range(len(unique_combinations_list))
plt.xticks(tick_indices, unique_combinations_list, rotation=90, fontsize=11)
plt.yticks(tick_indices, unique_combinations_list, fontsize=11)

plt.xlabel('Predicted Class', fontsize=28) # Customize x-axis label
plt.ylabel('True Class', fontsize=28) # Customize y-axis label
plt.title('Confusion Matrix', fontsize=36) # Customize title
plt.show()

 
list_1=all_true_labels
list_2=all_predicted_labels
same_count = 0
two_same_count = 0
one_same_count = 0
wrong_count = 0

for item_1, item_2 in zip(list_1, list_2):
    if item_1 == item_2:
        same_count += 1
    elif len(set(item_1) & set(item_2)) == 2:
        two_same_count += 1
    elif len(set(item_1) & set(item_2)) == 1:
        one_same_count += 1
    else:
        wrong_count += 1

print("Same:", same_count/len(all_predicted_labels)*100)
print("Two same:", two_same_count/len(all_predicted_labels)*100)
print("One same:", one_same_count/len(all_predicted_labels)*100)
print("Wrong:", wrong_count/len(all_predicted_labels)*100)