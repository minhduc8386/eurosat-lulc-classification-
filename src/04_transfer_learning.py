import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.layers import (Input, Dense, Dropout, GlobalAveragePooling2D,
                          BatchNormalization, Resizing, Rescaling)
from keras.applications import ResNet50V2

from config import (IMG_SIZE, NUM_CLASSES, RANDOM_STATE, TL_SIZE, EPOCHS_PHASE1, EPOCHS_PHASE2,
                    RESULTS_DIR)
from data_utils import load_train_val, make_augmentation
from train_utils import train_and_save, plot_history

tf.keras.utils.set_random_seed(RANDOM_STATE)


def count_trainable(model):
    return sum(int(np.prod(w.shape)) for w in model.trainable_weights)


data = load_train_val()

base_model = ResNet50V2(input_shape=(TL_SIZE, TL_SIZE, 3), include_top=False, weights="imagenet")
base_model.trainable = False

model = Sequential(name="resnet50v2_tl")
model.add(Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(make_augmentation())
model.add(Resizing(TL_SIZE, TL_SIZE))            # 64 -> 160
model.add(Rescaling(scale=2.0, offset=-1.0))     # [0, 1] -> [-1, 1]
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.2))
model.add(Dense(NUM_CLASSES, activation="softmax"))
model.summary()
print("Trainable (phase 1):", count_trainable(model))

# phase 1: frozen backbone, train only the new classifier
history1 = train_and_save(model, "resnet50v2_phase1", data, learning_rate=1e-3, epochs=EPOCHS_PHASE1)

# phase 2: unfreeze conv5 (BatchNorm stays frozen), small learning rate
base_model.trainable = True
for layer in base_model.layers:
    layer.trainable = layer.name.startswith("conv5") and not isinstance(layer, BatchNormalization)
print("Trainable (phase 2):", count_trainable(model))

history2 = train_and_save(model, "resnet50v2_finetuned", data, learning_rate=1e-5,
                          epochs=len(history1) + EPOCHS_PHASE2, initial_epoch=len(history1))

full_history = pd.concat([history1, history2], ignore_index=True)
full_history.to_csv(os.path.join(RESULTS_DIR, "resnet50v2_full_history.csv"), index=False)
plot_history(full_history, "ResNet50V2 – phase 1 + phase 2", "resnet50v2_curve.png",
             mark_epoch=len(history1) + 0.5)
