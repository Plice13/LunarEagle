import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the model
main_dir = r'C:\Users\PlicEduard\AI\more\runs_martin\Axx_Hsx_600'
model_name = 'model_bw_axx_hsx__e-15_spe-160_vspe-40_bs-32.h5'
model = load_model(os.path.join(main_dir, model_name))

# Print the model summary to verify its architecture
model.summary()


# Folder containing the images
image_folder = os.path.join(main_dir, 'test')

# Get a list of all files in the folder
image_paths = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith(('png', 'jpg', 'jpeg'))]

# Load and preprocess the images, converting to grayscale
images = [Image.open(path).convert('L').resize((300, 300)) for path in image_paths]
image_arrays = [np.array(img) / 255.0 for img in images]
image_arrays = np.array(image_arrays)

# Add a channel dimension if the model expects input shape (None, 300, 300, 1)
if model.input_shape[-1] == 1:
    image_arrays = np.expand_dims(image_arrays, axis=-1)

# Make batch predictions
predictions_batch = model.predict(image_arrays)
print(predictions_batch)
# Process predictions for each image
good_predict = 0
bad_predict = 0
for i, (path, predictions) in enumerate(zip(image_paths, predictions_batch)):
    class_index = np.argmax(predictions)
    classes = ['Axx', 'Hsx']
    predicted_class = classes[class_index]
    confidence = predictions[class_index]

    #proces path
    filename = (os.path.basename(path)).split('-')[0]
    if filename == predicted_class:
        good_predict +=1
    else:
        bad_predict +=1
    print(f"Image {i+1}: {os.path.basename(filename.split('_')[0])} - Predicted Class: {predicted_class}, Confidence: {confidence}")
total_accuracy = good_predict/(good_predict+bad_predict)
print(f'Model je správně z {total_accuracy*100} %')