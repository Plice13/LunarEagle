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
from sklearn.metrics import confusion_matrix
import time


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

        # Make batch predictions
    predictions_batch = model.predict(image_arrays)

    # Process predictions for each image
    predicted_labels = [np.argmax(predictions) for predictions in predictions_batch]

    # True labels
    true_labels = [classes.index(os.path.normpath(path).split(os.path.sep)[-2]) for path in image_paths]

    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Print confusion matrix

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
    print(cm)
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

def build_and_config_model(hidden_layers_configuration):
    ###-----Build Your Model------###
    hlc = hidden_layers_configuration
    model = models.Sequential()
    for i in range(len(hidden_layers_configuration)):
        if i == 0:
            model.add(layers.Conv2D(hlc[i][1][0], hlc[i][1][1], activation=hlc[i][1][2], input_shape=(300, 300, 1)))  # Change input shape to (300, 300, 1)
        else:
            type_of_layer = hlc[i][0]
            if type_of_layer == 'c':
                model.add(layers.Conv2D(hlc[i][1][0], hlc[i][1][1], activation=hlc[i][1][2])) 
            elif type_of_layer == 'mp':
                model.add(layers.MaxPooling2D(hlc[i][1][1]))
            elif type_of_layer == 'd':
                model.add(layers.Dense(hlc[i][1][0], activation=hlc[i][1][2]))
            elif type_of_layer == 'f':
                model.add(layers.Flatten())

    model.summary()


    ####----Configuring the model for training-----####
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.RMSprop(learning_rate=1e-4), metrics=['acc'])
    return model

def get_layers_string(layers_configuration):
    string_rep = f'{len(layers_configuration)}L'
    for layer in layers_configuration:
        layer_type = layer[0]
        if layer_type == 'c':
            units = layer[1][0]
            kernel_size = layer[1][1]
            activation = layer[1][2]
            string_rep += f'-c{units}({kernel_size[0]},{kernel_size[1]})'
        elif layer_type == 'mp':
            pool_size = layer[1][1]
            string_rep += f'-mp({pool_size[0]},{pool_size[1]})'
        elif layer_type == 'f':
            string_rep += '-f'
        elif layer_type == 'd':
            units = layer[1][0]
            activation = layer[1][2]
            string_rep += f'-d{units}'
    return string_rep

def get_parameters(path_base):
    classes = path_base.split('_')
    number_of_classes = len(classes)

    return classes, number_of_classes

if __name__=='__main__':
    # set up
    main_dir = r'C:\Users\PlicEduard\AI4_SOC\Axx_Bxi_Cai_Cso'
    train_dir = os.path.join(main_dir, 'train')
    val_dir = os.path.join(main_dir, 'val')
    test_dir = os.path.join(main_dir, 'test')
    
    # make parameters
    bs = 32
    classes, number_of_classes = get_parameters(os.path.basename(main_dir))
    max_counter = 300 
    vs = 36
    spe = 14
 
    # make model
    number_of_hidden_layers = 3
    
    noc = number_of_classes
    # structure like [  [   convolution/maxpooling/dense/flattern    ,   [number of neurons, size of matrix, activation function]##end of layer config   ]##end of layer, [...new layer...] ]##end of model
    layers_configuration =[['c',    [16,    (4,4),  'relu'  ]],
                           ['mp',   [None,  (3,3),  None    ]],
                           ['c',    [8,    (3,3),  'relu'  ]],
                           ['mp',   [None,  (2,2),  None    ]],
                           ['f',    [None,  None,   None    ]],
                           ['d',    [32,    None,   'relu'  ]],
                           ['d',    [noc,   None, 'softmax' ]]] # noc = number_of_classes
    layers_string = get_layers_string(layers_configuration)
    print(layers_string)
    model = build_and_config_model(layers_configuration)

    # prepare model
    train_generator, val_generator, test_generator = prepare_model()

    # prepare lists
    acc_list = []
    loss_list = []
    val_acc_list = []
    val_loss_list = []

    # get weight of classes
    weights = list()
    for clss in classes:
        weights.append(len(os.listdir(os.path.join(train_dir,clss)))+len(os.listdir(os.path.join(val_dir,clss)))+len(os.listdir(os.path.join(test_dir,clss))))
    sum_of_samples = sum(weights)
    print(weights)
    for i in range(len(weights)):
        weights[i] = round(sum_of_samples/weights[i])
    print(weights)

    
    start_time = time.time()
    print(f'In time {start_time}: spe je {spe} a vs je {vs}')


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
        train_history = model.fit(train_generator, epochs=1, steps_per_epoch=spe, validation_data=val_generator, validation_steps=vs)#, class_weight={0: weights[0], 1: weights[1], 2: weights[2], 3: weights[3], 4: weights[4], 5: weights[5]}) #, validation_steps=50, class_weight={0: 1, 1: 1, 2: 1})
        acc_list.append(train_history.history['acc'][0])
        loss_list.append(train_history.history['loss'][0])
        val_acc_list.append(train_history.history['val_acc'][0])
        val_loss_list.append(train_history.history['val_loss'][0])
        
        now_time = time.time()
        elapsed_time = (now_time - start_time)/60
        print("In time: {:.2f} minutes".format(elapsed_time))
        with open(os.path.join(main_dir, "log.txt"), "a") as log_file:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log_file.write("In time {}, running {:.2f} minutes the epoch {} has produced loss {}\n".format(timestamp, elapsed_time,counter,val_loss_list[-1]))


        # if val_loss is low, save model
        best_val_loss = min(val_loss_list)
        if best_val_loss == val_loss_list[-1]:
            model_name = f'model_bw__e-{counter}_spe-{spe}_vspe-{vs}_bs-{bs}_{layers_string}_loss-{round(val_loss_list[-1],6)}.h5'
            model.save(os.path.join(main_dir, model_name))

        if counter%15 == 0:
            model_name = f'model_bw__e-{counter}_spe-{spe}_vspe-{vs}_bs-{bs}_{layers_string}_loss-{round(val_loss_list[-1],6)}.h5'
            model.save(os.path.join(main_dir, model_name))

        counter+=1

    # save model
    print(f'\n Final list was:\n {val_loss_list}')
    model_name = f'end_of_model_bw__e-{counter}_spe-{spe}_vspe-{vs}_bs-{bs}_{layers_string}_loss-{round(val_loss_list[-1],6)}.h5'
    model.save(os.path.join(main_dir, model_name))

    # plot results
    to_plot_list=[[acc_list,'acc'], [loss_list,'loss'], [val_acc_list,'val_ass'], [val_loss_list,'val_loss']]
    plot_results(to_plot_list)

    ## test model
    # load model
    best_epoch = val_loss_list.index(min(val_loss_list))
    best_val_loss = min(val_loss_list)
    model_name = f'model_bw__e-{best_epoch + 1}_spe-{spe}_vspe-{vs}_bs-{bs}_{layers_string}_loss-{round(best_val_loss, 6)}.h5'
    model = keras.models.load_model(os.path.join(main_dir, model_name))

    #test model
    test_model(model)
    print(f'spe je {spe} a vs je {vs}, nejlepsi epocha {best_epoch} a obrázku {weights}')