import os
import numpy as np
from keras.models import load_model

from config import CLASS_NAMES, MODELS_DIR
from data_utils import load_split

# read the test data
X, y = load_split("test")

# load each trained model from models/ and predict
for filename in ["simple_cnn.keras", "complex_cnn_best.keras", "resnet50v2_finetuned.keras"]:
    model = load_model(os.path.join(MODELS_DIR, filename))
    y_pred = np.argmax(model.predict(X, verbose=0), axis=1)

    print(filename)
    print("accuracy:", round(np.mean(y_pred == y), 4))
    print("y_pred:", [CLASS_NAMES[i] for i in y_pred[:5]])
    print("y_true:", [CLASS_NAMES[i] for i in y[:5]])
