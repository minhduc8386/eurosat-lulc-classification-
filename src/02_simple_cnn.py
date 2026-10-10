import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

from config import IMG_SIZE, NUM_CLASSES, RANDOM_STATE, EPOCHS
from data_utils import load_train_val, make_augmentation
from train_utils import train_and_save, plot_history

tf.keras.utils.set_random_seed(RANDOM_STATE)

data = load_train_val()

model = Sequential(name="simple_cnn")
model.add(Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(make_augmentation())
model.add(Conv2D(32, (3, 3), activation="relu"))     # 62x62x32
model.add(MaxPooling2D((2, 2)))                      # 31x31x32
model.add(Conv2D(64, (3, 3), activation="relu"))     # 29x29x64
model.add(MaxPooling2D((2, 2)))                      # 14x14x64
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(NUM_CLASSES, activation="softmax"))
model.summary()

history = train_and_save(model, "simple_cnn", data, learning_rate=1e-3, epochs=EPOCHS)
plot_history(history, "Simple CNN", "simple_cnn_curve.png")
