###-----Build Your Model------###
from keras import layers

from keras import models

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(300,300, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(2, activation='sigmoid'))
model.summary()


####----Configuring the model for training-----####
from tensorflow import keras
from keras import optimizers
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.RMSprop(lr=1e-4), metrics=['acc'])

#####-----Data Preprocessing-----######
import os
main_dir = r'C:\Users\PlicEduard\AI\more\runs\Axx Hsx 600'
train_dir = os.path.join(main_dir, 'train')
validation_dir = os.path.join(main_dir, 'val')

model_name = 'model_rgb_axx_hsx.h5'

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, batch_size=16, target_size=(300, 300), class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(validation_dir, batch_size=16, target_size=(300, 300), class_mode='categorical')

####----Fit the Model----####
history = model.fit(train_generator, steps_per_epoch=69, epochs=1, validation_data=validation_generator, validation_steps=50, class_weight={0: 1, 1: 1, 2: 1})

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
