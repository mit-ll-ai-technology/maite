import numpy as np
from PIL import Image


def create_random_image():
    """
    Creates a random image and returns it as a PIL image.
    """
    image = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    return image
