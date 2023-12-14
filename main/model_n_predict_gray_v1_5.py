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
from time import sleep


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

def prepare_model():
    train_datagen = ImageDataGenerator(rescale=1./255)  # rescale pixel values to [0, 1]
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = custom_image_generator(train_datagen, train_dir, batch_size=bs, target_size=(300, 300), class_mode='categorical')
    val_generator = custom_image_generator(val_datagen, val_dir, batch_size=bs, target_size=(300, 300), class_mode='categorical')
    test_generator = custom_image_generator(test_datagen, test_dir, batch_size=bs, target_size=(300, 300), class_mode='categorical')
    return train_generator, val_generator, test_generator

def test_model(model):
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

def plot_results(lists):
    for llist_n_name in lists:
        # Plotting the values with different line styles or colors
        plt.plot(llist_n_name[0], marker='x', linestyle='-', label=str(llist_n_name[1]))

    # Adding labels and title
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Plot of values')

    # Display the legend
    plt.legend()

    # Display the plot
    plt.show()

def build_and_config_model(number_of_classes):
    ###-----Build Your Model------###
    model = models.Sequential()

    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(300, 300, 1)))  # Change input shape to (300, 300, 1)
    model.add(layers.MaxPooling2D((2, 2)))
   
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(8, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(number_of_classes, activation='softmax'))  # Assuming 3 classes

    model.summary()


    ####----Configuring the model for training-----####
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.RMSprop(learning_rate=1e-4), metrics=['acc'])
    return model

def get_parameters(path_base, bs, scalable_factor=1):
    path_base = path_base.split('-')[0]

    classes = path_base.split('_')[:-2]
    number_of_classes = len(classes)
    samples = (int(path_base.split('_')[-2]) * number_of_classes)//scalable_factor
    v_samples = (int(path_base.split('_')[-1]) * number_of_classes)//scalable_factor
    spe = samples//bs
    vs = v_samples//4

    return classes, number_of_classes, vs, spe

if __name__=='__main__':
    # set up
    main_dir = r'C:\Users\PlicEduard\AI2\Axx_Bxo_1600_160'
    train_dir = os.path.join(main_dir, 'train')
    val_dir = os.path.join(main_dir, 'val')
    test_dir = os.path.join(main_dir, 'test')
    
    # make parameters
    scalable_factor = 1
    bs = 32
    classes, number_of_classes, vs, spe = get_parameters(os.path.basename(main_dir), bs, scalable_factor=scalable_factor)
    max_counter = 300 

    # make model
    model = build_and_config_model(number_of_classes)

    # prepare model
    train_generator, val_generator, test_generator = prepare_model()

    # prepare lists
    acc_list = []
    loss_list = []
    val_acc_list = []
    val_loss_list = []

    # train model
    counter = 1
    while counter <= max_counter:
        # for knowing when press Q
        print(f'<{counter}>')
        sleep(2)

        # qiut learning before end
        if keyboard.is_pressed('q'):
            print("You pressed 'q'. Exiting the loop.")
            break

        # train model and save data
        train_history = model.fit(train_generator, epochs=1, steps_per_epoch=spe, validation_data=val_generator, validation_steps=vs)#, validation_steps=50, class_weight={0: 1, 1: 1, 2: 1})
        acc_list.append(train_history.history['acc'][0])
        loss_list.append(train_history.history['loss'][0])
        val_acc_list.append(train_history.history['val_acc'][0])
        val_loss_list.append(train_history.history['val_loss'][0])

        # if val_loss is low, save model
        best_val_loss = min(val_loss_list)
        if best_val_loss == val_loss_list[-1]:
            model_name = f'model_bw_{classes}__e-{counter}_spe-{spe}_vspe-{vs}_bs-{bs}_4L(64(4x4),32,16,8)_loss-{round(val_loss_list[-1],6)}.h5'
            model.save(os.path.join(main_dir, model_name))

        counter+=1

    # save model
    print(f'\n Final list was:\n {val_loss_list}')
    model_name = f'end_of_model_bw_{classes}_e-{counter}_spe-{spe}_vspe-{vs}_bs-{bs}_4L(64(4x4),32,16,8)_loss-{round(val_loss_list[-1],6)}.h5'
    model.save(os.path.join(main_dir, model_name))

    # plot results
    to_plot_list=[[acc_list,'acc'], [loss_list,'loss'], [val_acc_list,'val_ass'], [val_loss_list,'val_loss']]
    plot_results(to_plot_list)

    ## test model
    # load model
    best_epoch = val_loss_list.index(min(val_loss_list))
    best_val_loss = min(val_loss_list)
    model_name = f'model_bw_{classes}__e-{best_epoch + 1}_spe-{spe}_vspe-{vs}_bs-{bs}_4L(64(4x4),32,16,8)_loss-{round(best_val_loss, 6)}.h5'
    model = keras.models.load_model(os.path.join(main_dir, model_name))

    #test model
    test_model(model)