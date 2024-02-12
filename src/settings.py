import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)

# Tumor labels
LABELS = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']

# French translation of the labels
TRANSLATION = {'glioma_tumor': 'glioblastome',
               'no_tumor': 'aucune tumeur',
               'meningioma_tumor': 'méningiome',
               'pituitary_tumor': 'tumeur pituitaire'}

# Image size (pixels).
# Should not be too large, to avoid a long training time.
# Should not be too small, to avoid bad performances.
IMAGE_SIZE = 150

# Limiting the number of images for debugging purpose
# Should be an integer or None
IMG_LIMIT = None

# Epochs. Should be around 20
EPOCHS = 20
