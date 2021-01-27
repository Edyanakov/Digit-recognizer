from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow as tf

config = {
    'IMG_SHAPE': (28,28,1),
    'NUM_CLASSES': 10,
    'EPOCHS': 10,
    'BATCH_SIZE': 32,
    'LR': 0.001,
}

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
#One hot encoding labels
y_cat_train = to_categorical(y_train,10)
y_cat_test = to_categorical(y_test,10)
#Normalizing images
x_train = x_train / 255.0
x_test = x_test / 255.0
#Reshaping images to 1 RGB channel(black and white images)
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)


def build_model(num_classes):
    model = Sequential()
    model.add(Conv2D(filters=32,kernel_size=(5,5),input_shape=config['IMG_SHAPE'], activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=16,kernel_size=(3,3),input_shape=config['IMG_SHAPE'], activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(lr=config['LR']),
                  metrics=['accuracy'])

    return model

model = build_model(config['NUM_CLASSES'])
model.summary()
early_stop = EarlyStopping(monitor='val_loss', patience=3)

model.fit(x_train, y_cat_train,
          epochs=config['EPOCHS'], batch_size=config['BATCH_SIZE'],
          validation_data=(x_test, y_cat_test),
          callbacks=[early_stop])

model.save('mnist_recognizer.h5')
print('Model saved as hd5 file')

score = model.evaluate(x_test, y_cat_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])