import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix

# Load the model
#model_dir = r'C:\Users\PlicEduard\AI4_SOC\A_B_C_D_E_F_H'
#model_dir = r'C:\Users\PlicEduard\AI4_SOC\Axx_Bxi_Cai_Cso'
#model_dir =r'C:\Users\PlicEduard\AI4_SOC\Axx_Csi_Eac_Hsx'
#model_dir = r'C:\Users\PlicEduard\AI4_SOC\a_h_k_r_s_x'
model_dir = r'C:\Users\PlicEduard\AI4_SOC\c_i_o_x'

samples_dir = os.path.join(model_dir, 'test')

model_name = 'best' #0.92 for ACEH
#model_name = '0.92'

#classes = ['c','i','o','x']
#classes = ['A', 'B', 'C', 'D', 'E', 'F', 'H']  
#classes = ['Axx','Bxi','Bxo','Cai','Cao','Chi','Cho','Cki','Cko','Cri','Cro','Csi','Cso','Dac','Dai','Dao','Dhi','Dkc','Dki','Dko','Dri','Dro','Dsc','Dsi','Dso','Eac','Eai','Ekc','Eki','Eko','Esc','Esi','Fac','Fkc','Fki','Hax','Hhx','Hkx','Hrx','Hsx']
#classes = ['Axx', 'Bxi', 'Bxo', 'Cai', 'Cao', 'Chi', 'Cho', 'Cki', 'Cko', 'Cri', 'Cro', 'Csi', 'Cso', 'Dac', 'Dai', 'Dao', 'Dhc', 'Dhi', 'Dho', 'Dkc', 'Dki', 'Dko', 'Dri', 'Dro', 'Dsc', 'Dsi', 'Dso', 'Eac', 'Eai', 'Eao', 'Ehc', 'Ehi', 'Ekc', 'Eki', 'Eko', 'Esc', 'Esi', 'Eso', 'Fac', 'Fai', 'Fhc', 'Fkc', 'Fki', 'Fko', 'Fsi', 'Hax', 'Hhx', 'Hkx', 'Hrx', 'Hsx']

classes = os.path.basename(model_dir).split('_')



model_files = [model for model in os.listdir(model_dir) if model.endswith('.h5')]
for model_file in model_files:
    if model_file.startswith(model_name):
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

            if filename == predicted_class:
                good_predict += 1
            else:
                bad_predict += 1

        total_accuracy = good_predict / (good_predict + bad_predict)


        print(f'Model {model_file} je správně z {total_accuracy*100:.2f} %')

        print(classes)

        if model_name == 'best':
            # Prepend spaces to each class name
            for n in range(len(classes)):  # Use range(len(classes)) to iterate
                classes[n] = '   ' + classes[n]  # Prepend two spaces to each class

            print(classes)

            
        import matplotlib.pyplot as plt
        import numpy as np

        # Plot confusion matrix using the default 'Greens' colormap
        plt.figure(figsize=(10, 8), facecolor='none')  # Set facecolor to none for transparency
        plt.matshow(cm, cmap=plt.cm.Greens, vmin=0, vmax=np.max(cm))  # Use the default 'Greens' colormap
        plt.colorbar()

        tick_marks = np.arange(len(classes))  # Adjust tick positions

        # Set x and y axis labels with bold text
        plt.xticks(tick_marks, classes, fontsize=12, fontweight='bold')  # Bold class labels on x-axis
        plt.yticks(tick_marks, classes, fontsize=12, fontweight='bold')  # Bold class labels on y-axis

        title = 'Confusion Matrix'

        # Set title with bold text
        plt.title(title, fontsize=22, fontweight='bold')  # Bold title
        plt.subplots_adjust(top=0.85, bottom=0.2)  # Fix the top and bottom margins

        # Set axis labels with bold text
        plt.xlabel('Predicted Class', fontsize=18, fontweight='bold')  # Bold x-axis label
        plt.ylabel('True Class', fontsize=18, fontweight='bold')  # Bold y-axis label

        # Save the figure with a transparent background
        plt.savefig(f"EUCYS/{title}+{classes}.svg", transparent=True)  # Save the figure with transparent background
        plt.show()

                
        print(cm)
