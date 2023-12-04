import os
import cv2
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras import layers, models, optimizers
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import statistics
import keyboard

def build_and_config_model(number_of_classes):
    ###-----Build Your Model------###
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 1)))  # Change input shape to (300, 300, 1)
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(number_of_classes, activation='softmax'))  # Assuming 3 classes

    model.summary()


    ####----Configuring the model for training-----####
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.RMSprop(learning_rate=1e-4), metrics=['acc'])
    return model

def custom_image_generator(generator, directory, batch_size, target_size, class_mode):
    for data_batch, labels_batch in generator.flow_from_directory(directory, target_size=target_size, batch_size=batch_size, class_mode=class_mode, shuffle=True):
        data_batch_gray = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in data_batch]
        data_batch_gray = np.array(data_batch_gray)
        data_batch_gray = np.expand_dims(data_batch_gray, axis=-1)
        
        # cv2.imshow('im', data_batch_gray[0])
        # print(labels_batch[0])
        # cv2.waitKey(0)

        # cv2.imshow('im', data_batch_gray[1])
        # print(labels_batch[1])
        # cv2.waitKey(0)

        # cv2.imshow('im', data_batch_gray[2])
        # print(labels_batch[2])
        # cv2.waitKey(0)

        # cv2.imshow('im', data_batch_gray[3])
        # print(labels_batch[3])
        # cv2.waitKey(0)

        # cv2.destroyAllWindows()
        # cv2.release()
        # exit()
        yield (data_batch_gray, labels_batch)

def plot_results(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(range(1, len(history.history['acc']) + 1), history.history['acc'], 'r', label='Training acc')
    plt.plot(range(1, len(history.history['val_loss']) + 1), history.history['val_loss'], 'y', label='Validation loss')
    plt.plot(range(1, len(history.history['loss']) + 1), history.history['loss'], 'purple', label='Training loss')
    plt.plot(range(1, len(history.history['val_acc']) + 1), history.history['val_acc'], 'g', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.show()

main_dir = r'C:\Users\PlicEduard\AI\Axx_Cso_Dai_500'
train_dir = os.path.join(main_dir, 'train')
val_dir = os.path.join(main_dir, 'val')
test_dir = os.path.join(main_dir, 'test')

classes = ['Axx', 'Cso', 'Dai']
number_of_classes = len(classes)
samples = 250 * number_of_classes
bs = 32
vs = int(samples*0.1)
spe = samples//bs

model = build_and_config_model(number_of_classes)


train_datagen = ImageDataGenerator(rescale=1./255)  # rescale pixel values to [0, 1]
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = custom_image_generator(train_datagen, train_dir, batch_size=bs, target_size=(300, 300), class_mode='categorical')
val_generator = custom_image_generator(val_datagen, val_dir, batch_size=bs, target_size=(300, 300), class_mode='categorical')
test_generator = custom_image_generator(test_datagen, test_dir, batch_size=bs, target_size=(300, 300), class_mode='categorical')

train_history = model.fit(train_generator, epochs=10, steps_per_epoch=spe, validation_data=val_generator, validation_steps=vs)#, validation_steps=50, class_weight={0: 1, 1: 1, 2: 1})
val_loss_list = train_history.history['val_loss']
accuracy_list = train_history.history['acc']

starting_loss = val_loss_list[0]
minimum = starting_loss
counter = 11
####----Fit the Model----####
#while val_loss_list[-1:] < val_loss_list[-3:-2]:
while counter < 30:
    # Check if 'q' key is pressed
    if keyboard.is_pressed('q'):
        print("You pressed 'q'. Exiting the loop.")
        break

    train_history = model.fit(train_generator, epochs=1, steps_per_epoch=spe, validation_data=val_generator, validation_steps=vs)#, validation_steps=50, class_weight={0: 1, 1: 1, 2: 1})
    val_loss = train_history.history['val_loss'][0]
    accuracy_list.append(train_history.history['acc'][0])
    val_loss_list.append(val_loss)

    counter+=1
print(f'\n Final list was:\n {val_loss_list}')
######-----Save the Model-------######q
model_name = f'model_bw_{classes}__e-{counter}_spe-{spe}_vspe-{vs}_bs-{bs}.h5'


model.save(os.path.join(main_dir, model_name))

# plot_results(train_history)
# Plotting the values
plt.plot(val_loss_list, marker='o', linestyle='-')
plt.plot(accuracy_list, marker='o', linestyle='-')

# Adding labels and title
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Plot of val_loss_list')

# Display the plot
plt.show()

# Folder containing the images
image_folder = os.path.join(main_dir, 'test')

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
print(predictions_batch)
# Process predictions for each image
good_predict = 0
bad_predict = 0
for i, (path, predictions) in enumerate(zip(image_paths, predictions_batch)):
    class_index = np.argmax(predictions)
    predicted_class = classes[class_index]
    confidence = predictions[class_index]

    #proces path
    filename = (os.path.normpath(path).split(os.path.sep)[-2])
    if filename == predicted_class:
        good_predict +=1
    else:
        bad_predict +=1
    print(f"{filename} - Predict: {predicted_class}, Confidence: {confidence:.4f}")
total_accuracy = good_predict/(good_predict+bad_predict)
print(f'Model je správně z {total_accuracy*100} %')

os.path.join(main_dir, model_name)
os.rename(os.path.join(main_dir, model_name), os.path.join(main_dir, f"{total_accuracy:.4f}_{model_name}"))
