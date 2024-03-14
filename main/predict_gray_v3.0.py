import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix

# Load the model
samples_dir = r'C:\Users\PlicEduard\AI4_SOC\Axx_Bxi_Bxo_Cai_Cao_Chi_Cho_Cki_Cko_Cri_Cro_Csi_Cso_Dac_Dai_Dao_Dhc_Dhi_Dho_Dkc_Dki_Dko_Dri_Dro_Dsc_Dsi_Dso_Eac_Eai_Eao_Ehc_Ehi_Ekc_Eki_Eko_Esc_Esi_Eso_Fac_Fai_Fhc_Fkc_Fki_Fko_Fsi_Hax_Hhx_Hkx_Hrx_Hsx\test'
model_dir = r'C:\Users\PlicEduard\AI4_SOC\Axx_Bxi_Bxo_Cai_Cao_Chi_Cho_Cki_Cko_Cri_Cro_Csi_Cso_Dac_Dai_Dao_Dhc_Dhi_Dho_Dkc_Dki_Dko_Dri_Dro_Dsc_Dsi_Dso_Eac_Eai_Eao_Ehc_Ehi_Ekc_Eki_Eko_Esc_Esi_Eso_Fac_Fai_Fhc_Fkc_Fki_Fko_Fsi_Hax_Hhx_Hkx_Hrx_Hsx'
model_name = '0.0800_model_bw__e-138_spe-32_vspe-54_bs-32_9L-c64(4,4)-mp(3,3)-c32(3,3)-mp(2,2)-c16(2,2)-mp(2,2)-f-d96-d50_loss-2.019897.h5'
#classes = ['c','i','o','x']
#classes = ['A', 'B', 'C', 'D', 'E', 'F', 'H']
#classes = ['Axx','Bxi','Bxo','Cai','Cao','Chi','Cho','Cki','Cko','Cri','Cro','Csi','Cso','Dac','Dai','Dao','Dhi','Dkc','Dki','Dko','Dri','Dro','Dsc','Dsi','Dso','Eac','Eai','Ekc','Eki','Eko','Esc','Esi','Fac','Fkc','Fki','Hax','Hhx','Hkx','Hrx','Hsx']
#classes = ['Axx', 'Bxi', 'Bxo', 'Cai', 'Cao', 'Chi', 'Cho', 'Cki', 'Cko', 'Cri', 'Cro', 'Csi', 'Cso', 'Dac', 'Dai', 'Dao', 'Dhc', 'Dhi', 'Dho', 'Dkc', 'Dki', 'Dko', 'Dri', 'Dro', 'Dsc', 'Dsi', 'Dso', 'Eac', 'Eai', 'Eao', 'Ehc', 'Ehi', 'Ekc', 'Eki', 'Eko', 'Esc', 'Esi', 'Eso', 'Fac', 'Fai', 'Fhc', 'Fkc', 'Fki', 'Fko', 'Fsi', 'Hax', 'Hhx', 'Hkx', 'Hrx', 'Hsx']

classes = os.path.basename(model_dir).split('_')



model_files = [model for model in os.listdir(model_dir) if model.endswith('.h5')]
for model_file in model_files:
    if model_file != model_name:
        break
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

    # Print confusion matrix
    # Plot confusion matrix
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    plt.matshow(cm, cmap=plt.cm.Greens)
    plt.colorbar()
    tick_marks = np.arange(1, len(classes) + 1)  # Adjust tick positions
    plt.xticks(tick_marks - 1, tick_marks)  # Adjust tick labels
    plt.yticks(tick_marks - 1, tick_marks)  # Adjust tick labels
    plt.xlabel('Správná třída', fontsize=12) # Customize y-axis label
    plt.ylabel('Predikovaná třída', fontsize=12) # Customize x-axis label
    plt.title('Konfuzní matice', fontsize=18) # Customize title
    plt.show()
    
    print(cm)
