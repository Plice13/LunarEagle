###-----Build Your Model------###
from keras import layers, models

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 1)))  # Change input shape to (300, 300, 1)
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))  # Assuming 3 classes

model.summary()


####----Configuring the model for training-----####
from tensorflow import keras
from keras import optimizers
import os
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.RMSprop(learning_rate=1e-4), metrics=['acc'])

#####-----Data Preprocessing-----######

#main_dir = r'C:\Users\PlicEduard\AI\more\runs\Axx Hsx 600'
main_dir = r'../../data_gray/Axx Hsx 600'
train_dir = os.path.join(main_dir, 'train')
val_dir = os.path.join(main_dir, 'val')
test_dir = os.path.join(main_dir, 'test')

model_name = 'model_bw_axx_hsx.h5'

import cv2
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

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

train_datagen = ImageDataGenerator(rescale=1./255)  # rescale pixel values to [0, 1]
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = custom_image_generator(train_datagen, train_dir, batch_size=32, target_size=(300, 300), class_mode='categorical')
val_generator = custom_image_generator(val_datagen, val_dir, batch_size=32, target_size=(300, 300), class_mode='categorical')
test_generator = custom_image_generator(test_datagen, test_dir, batch_size=32, target_size=(300, 300), class_mode='categorical')

####----Fit the Model----####
history = model.fit(train_generator, epochs=15, validation_data=val_generator)#, validation_steps=50, class_weight={0: 1, 1: 1, 2: 1})

######-----Save the Model-------######
model.save(os.path.join(main_dir, model_name))

######-----Displaying curves of loss and accuracy during training-------######
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(range(1, len(history.history['acc']) + 1), history.history['acc'], 'r', label='Training acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.show()
plt.plot(range(1, len(history.history['val_acc']) + 1), history.history['val_acc'], 'b', label='Validation acc')
plt.title('Training and validation loss')
plt.legend()
plt.show()
