import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)

# Tumor labels
LABELS = ['glioma', 'meningioma', 'pituitary', 'notumor']

# French translation of the labels
TRANSLATION = {'glioma': 'glioblastome',
               'meningioma': 'méningiome',
               'pituitary': 'tumeur pituitaire',
               'notumor': 'aucune tumeur'}

# Image size (pixels).
# Should not be too large, to avoid a long training time.
# Should not be too small, to avoid bad performances.
IMAGE_SIZE = 150

# Limiting the number of images for debugging purpose
# Should be an integer or None
IMG_LIMIT = None

# Epochs. Should be around 15
EPOCHS = 12

# Batch size.
BATCH_SIZE = 32
