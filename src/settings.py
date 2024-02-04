import matplotlib.pyplot as plt
import seaborn as sns

# Color palette
COLORS_DARK = ["#1F1F1F", "#313131", '#636363', '#AEAEAE', '#DADADA']
COLORS_RED = ["#331313", "#582626", '#9E1717', '#D35151', '#E9B4B4']
COLORS_GREEN = ['#01411C', '#4B6F44', '#4F7942', '#74C365', '#D0F0C0']

# Tumor labels
LABELS = ['no_tumor', 'glioma_tumor', 'meningioma_tumor', 'pituitary_tumor']

# French translation of the labels
TRANSLATION = {'no_tumor': 'aucune tumeur',
               'glioma_tumor': 'glioblastome',
               'meningioma_tumor': 'méningiome',
               'pituitary_tumor': 'tumeur pituitaire'}

# Image size (pixels)
IMAGE_SIZE = 150

# Limiting the number of images for testing/debugging purpose
# Should be an integer or None
IMG_LIMIT = None

# Epochs
EPOCHS = 20

if __name__ == '__main__':
    sns.palplot(COLORS_DARK)
    sns.palplot(COLORS_GREEN)
    sns.palplot(COLORS_RED)

    plt.show()
