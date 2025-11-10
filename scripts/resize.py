from PIL import Image
import random
import matplotlib.pyplot as plt

import torch
from torchvision.transforms import v2, InterpolationMode

# If you're trying to run that on Colab, you can download the assets and the
# helpers from https://github.com/pytorch/vision/tree/main/gallery/
from helpers import plot

from os import listdir
from os.path import isfile, join

plt.rcParams["savefig.bbox"] = "tight"

# if you change the seed, make sure that the randomly-applied transforms
# properly show that the image can be both transformed and *not* transformed!
torch.manual_seed(0)


DIR_PATH = "./data/pyxel/downloads/"
onlyfiles = [join(DIR_PATH, f) for f in listdir(DIR_PATH) if isfile(join(DIR_PATH, f))]


def display_image(image_path: str):
    print("Opening image:", image_path)
    orig_img = Image.open(image_path)

    print("Original Size:", orig_img.size)
    resized_imgs = [
        v2.Resize(size=size, interpolation=InterpolationMode.NEAREST_EXACT)(orig_img)
        for size in (
            (32, 32),
            (64, 64),
            (128, 128),
            (orig_img.size[1], orig_img.size[0]),
        )
    ]
    plot([orig_img] + resized_imgs)
    plt.show()


for path in random.sample(onlyfiles, 10):
    display_image(path)

# displayImage("./data/pyxel/downloads/daily-rock-132826.png")
