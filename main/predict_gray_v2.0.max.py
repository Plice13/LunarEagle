import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix

# Load the model
samples_dir = r'C:\Users\PlicEduard\AI2\Axx_Bxi_Bxo_Cai_Cao_Chi_Cho_Cki_Cko_Cri_Cro_Csi_Cso_Dac_Dai_Dao_Dhi_Dkc_Dki_Dko_Dri_Dro_Dsc_Dsi_Dso_Eac_Eai_Ekc_Eki_Eko_Esc_Esi_Fac_Fkc_Fki_Hax_Hhx_Hkx_Hrx_Hsx_0_0\test'
model_dir = r'C:\Users\PlicEduard\AI2\Axx_Bxi_Bxo_Cai_Cao_Chi_Cho_Cki_Cko_Cri_Cro_Csi_Cso_Dac_Dai_Dao_Dhi_Dkc_Dki_Dko_Dri_Dro_Dsc_Dsi_Dso_Eac_Eai_Ekc_Eki_Eko_Esc_Esi_Fac_Fkc_Fki_Hax_Hhx_Hkx_Hrx_Hsx_0_0'
#classes = ['c','i','o','x']
#classes = ['A', 'B', 'C', 'D', 'E', 'F', 'H']
classes = ['Axx','Bxi','Bxo','Cai','Cao','Chi','Cho','Cki','Cko','Cri','Cro','Csi','Cso','Dac','Dai','Dao','Dhi','Dkc','Dki','Dko','Dri','Dro','Dsc','Dsi','Dso','Eac','Eai','Ekc','Eki','Eko','Esc','Esi','Fac','Fkc','Fki','Hax','Hhx','Hkx','Hrx','Hsx']

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

    cm = np.array(cm)
    np.set_printoptions(threshold=np.inf)

    # Print confusion matrix
    print(cm)

    # Process predictions for each image
    good_1_predict = 0
    bad_1_predict = 0
    good_2_predict = 0
    bad_2_predict = 0
    good_3_predict = 0
    bad_3_predict = 0
    good_predict = 0
    bad_predict = 0
    for i, (path, predictions) in enumerate(zip(image_paths, predictions_batch)):
        class_index = np.argmax(predictions)
        predicted_class = classes[class_index]
        confidence = predictions[class_index]

        # process path
        filename = (os.path.normpath(path).split(os.path.sep)[-2])

        # first
        if filename[0] == predicted_class[0]:
            good_1_predict += 1
        else:
            bad_1_predict += 1


        # middle
        if filename[1] == predicted_class[1]:
            good_2_predict += 1
        else:
            bad_2_predict += 1

        # last
        if filename[2] == predicted_class[2]:
            good_3_predict += 1
        else:
            bad_3_predict += 1

        if filename == predicted_class:
            good_predict += 1
        else:
            bad_predict += 1

    total_1_accuracy = good_1_predict / (good_1_predict + bad_1_predict)
    total_2_accuracy = good_2_predict / (good_2_predict + bad_2_predict)
    total_3_accuracy = good_3_predict / (good_3_predict + bad_3_predict)
    total_accuracy = good_predict / (good_predict + bad_predict)


    print(f'Model {model_file} je správně z {total_accuracy*100:.2f} %')
    print(f'Model {model_file} má správné první písmeno z {total_1_accuracy*100:.2f} %')
    print(f'Model {model_file} má správné prostřední písmeno z {total_2_accuracy*100:.2f} %')
    print(f'Model {model_file} má správné poslední písmeno z {total_3_accuracy*100:.2f} %')
