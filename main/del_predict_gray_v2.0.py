import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix

# Load the model
samples_dir = r'C:\Users\PlicEduard\AI2\A_B_C_D_E_F_H_100_10\test'
model_dir = r'C:\Users\PlicEduard\AI2\A_B_C_D_E_F_H_100_10\pr'
#classes = ['c','i','o','x']
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'H']

print(classes)

model_files = [model for model in os.listdir(model_dir) if model.endswith('.h5')]
for model_file in model_files:
    model = load_model(os.path.join(model_dir, model_file))

    # Print the model summary to verify its architecture
    model.summary()


    # Folder containing the images
    image_folder = samples_dir

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

    # Process predictions for each image
    predicted_labels = [np.argmax(predictions) for predictions in predictions_batch]

    # True labels
    true_labels = [classes.index(os.path.normpath(path).split(os.path.sep)[-2]) for path in image_paths]

    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Print confusion matrix
    print(cm)

    # Process predictions for each image
    good_predict = 0
    bad_predict = 0
    for i, (path, predictions) in enumerate(zip(image_paths, predictions_batch)):
        class_index = np.argmax(predictions)
        predicted_class = classes[class_index]
        confidence = predictions[class_index]

        # process path
        filename = (os.path.normpath(path).split(os.path.sep)[-2])
        if filename == predicted_class:
            good_predict += 1
        else:
            bad_predict += 1

    total_accuracy = good_predict / (good_predict + bad_predict)
    print(f'Model {model_file} je správně z {total_accuracy*100:.2f} %')

    # Extract the model file name
    model_filename = os.path.basename(model_file)

    # Rename the model file
    new_model_filename = f"{total_accuracy:.4f}_{model_filename}"
    os.rename(os.path.join(model_dir, model_file), os.path.join(model_dir, new_model_filename))
