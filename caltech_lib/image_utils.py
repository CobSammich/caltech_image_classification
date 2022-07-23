"""

"""

from typing import Tuple, List
import torch
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import ipdb

def read_image(filename: str) -> np.ndarray:
    image = Image.open(filename)
    # gray -> RGB
    if image.mode == 'L':
        image = image.convert('RGB')
    return np.array(image)

def upsample_image(image: np.ndarray, new_size: Tuple[int, int]) -> np.ndarray:
    # TODO: This assumes image_size is the same for both values (image_size[0])
    upscale_ratio = new_size[0] / min(image.shape[:2])
    new_height = int(new_size[0] * upscale_ratio)
    new_width = int(new_size[1] * upscale_ratio)
    image = cv2.resize(image, dsize=(new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return image


def normalize(x: np.ndarray) -> np.ndarray:
    return (x - np.min(x)) / (np.max(x) - np.min(x))

# plotting functions
def plot_batch(images: np.ndarray, labels: List[str] = None):
    n_images = images.shape[0]
    N_COLS = 8
    n_rows = n_images // N_COLS
    f,ax = plt.subplots(n_rows, N_COLS)

    if labels is None:
        labels = ["" for _ in range(n_images)]

    for i, (image, label) in enumerate(zip(images, labels)):
        r,c = divmod(i, N_COLS)
        if type(image) == torch.Tensor:
            image = np.moveaxis(normalize(image.detach().cpu().numpy()), 0, -1)
        ax[r,c].imshow(image)
        ax[r,c].set_title(label)
        ax[r,c].set_xticks([])
        ax[r,c].set_yticks([])
    plt.show()
    plt.close(f)



