import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib
matplotlib.use("Agg")

# paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "EuroSAT_RGB")
SPLITS_CSV = os.path.join(PROJECT_DIR, "data", "splits.csv")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")

for folder in [MODELS_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(folder, exist_ok=True)

DATASET_URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"

# data
CLASS_NAMES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
               "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 64

# training
RANDOM_STATE = 42
BATCH_SIZE = 64
EPOCHS = 30
PATIENCE = 5

TL_SIZE = 160          # input size for ResNet50V2
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 10
