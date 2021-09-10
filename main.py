from math import ceil
from re import VERBOSE
import torch
from torch.optim import optimizer
from torchvision.models import vgg19
from torchvision import transforms
import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from PIL import Image
import os
from datetime import datetime
import json
import torch.multiprocessing as mp

# from by_layer_model import ByLayerModel
from style_vgg19 import StyleVGG19
from named_image import NamedImage
from style_transfer_loss import StyleTansferLoss
from total_variation_loss import TotalVariationLoss
from trainer import Trainer


def run_content_image(content_image, style_images):

    # assert(set(style_names).issubset(set(layers)))
    # assert(set(content_names).issubset(set(layers)))

    criterion = StyleTansferLoss(style_layers=STYLE_NAMES, content_layers=CONTENT_NAMES, 
                                                alpha=ALPHA, beta=BETA, device=DEVICE, 
                                                content_weights=CONTENT_WEIGHTS, 
                                                square_error=SQUARE_ERROR,
                                                gram_matirx_norm=GRAM_MATRIX_NORM, styles_imgs_weights= STYLE_IMGS_WEIGTHS )

    model = StyleVGG19(replace_pooling=REPLACE_POOLING)

    trainer = Trainer(ephoch_num=EPOCH_NUM, input_size=INPUT_SIZE, criterion=criterion,
                      model=model, device=DEVICE, random_starts=RANDOM_STARTS,
                      verbose=VERBOSE, optimizer=OPTIMIZER, save_every=PLOT_EVERY, multiple_styles=MULTIPULE_STYLES)

    if not MULTIPULE_STYLES: 
        for style_image in style_images:

            folder_name = os.path.join(output_folder, f"{content_image.name}", f"{style_image.name}")
            os.makedirs(folder_name)

            trainer.save_path = os.path.join(folder_name, "training")

            inputs, loss_values = trainer.train(style_image.image, content_image.image)

            plt.rcParams["figure.figsize"] = (16, 9)

            plt.semilogy(np.arange(len(loss_values)) + 1,
                        loss_values, label=f"{style_image.name}")
            plt.legend()
            plt.savefig(os.path.join(output_folder,
                        f"{content_image.name}_loss.png"))



            trasform = transforms.ToPILImage()

            for i, t in enumerate(inputs):

                img = trasform(t)
                img.save(os.path.join(folder_name, f"{i}.png"))
    else:
            st_images=[i.image for i in style_images]
            st_images_name=",".join([i.name for i in style_images])
            folder_name = os.path.join(output_folder, f"{content_image.name}", f"{st_images_name}")
            os.makedirs(folder_name)
            trainer.save_path = os.path.join(folder_name, "training")
            inputs, loss_values = trainer.train(st_images, content_image.image)

            plt.rcParams["figure.figsize"] = (16, 9)
            plt.semilogy(np.arange(len(loss_values)) + 1,
                        loss_values, label=f"{st_images_name}")
            plt.legend()
            plt.savefig(os.path.join(output_folder,
                        f"{content_image.name}_loss.png"))

            trasform = transforms.ToPILImage()

            for i, t in enumerate(inputs):
                img = trasform(t)
                img.save(os.path.join(folder_name, f"{i}.png"))


    plt.clf()


def read_images(images_path):
    """Read images from a folder

    :param images_path: a folder to read
    :return: a list of NamedImage objects
    """
    style_images = []
    for name in os.listdir(images_path):
        filepath = os.path.join(images_path, name)
        style_images.append(NamedImage(filepath))

    return style_images

def filter_images(images, names_subset):
    return [img for img in images if img.name in names_subset]


def multiprocsess_run(content_images, style_images):
    procsesses = []

    for img in content_images:
        p = mp.Process(target=run_content_image, args=(img, style_images))
        p.start()
        procsesses.append(p)

    for p in procsesses:
        p.join()


def run(content_images, style_images):

    for img in content_images:
        run_content_image(img, style_images)

if __name__ == "__main__":
    EPOCH_NUM = 100
    INPUT_SIZE = (3, 256, 256)
    SEED = 6643527
    RANDOM_STARTS = 1
    ALPHA = 1
    BETA = 1e9
    DEVICE = "cuda"
    STYLE_WEIGTHS = {'conv1_1' : 0.2,
                     'conv2_1' : 0.2,
                     'conv3_1' : 0.2,
                     'conv4_1' : 0.2,
                     'conv5_1' : 0.2}

    STYLE_NAMES = list(STYLE_WEIGTHS.keys())
    CONTENT_WEIGHTS = {"relu4_2": 1}
    CONTENT_NAMES = list(CONTENT_WEIGHTS.keys())

    VARIATION_LAMBDA = 0
    REPLACE_POOLING = True
    SQUARE_ERROR = True

    GRAM_MATRIX_NORM = False


    OPTIMIZER = 'LBFGS'
    OPTIMIZER = OPTIMIZER.lower()
    assert OPTIMIZER in ['lbfgs', 'adam']

    BASE_OUTPUT_DIR = "outputs_all" 

    PLOT_EVERY = -1
    
    MULTIPULE_STYLES= True

    configuration = {"epoch num": EPOCH_NUM, "input size": INPUT_SIZE, "SEED": SEED,
                     "RANDOM STARTS": RANDOM_STARTS, "ALPHA": ALPHA, "BETA": BETA, 
                     "device": DEVICE, "style names": STYLE_NAMES, "content names": CONTENT_NAMES,
                     "style weigths":STYLE_WEIGTHS, "content weigths": CONTENT_WEIGHTS, 
                     "variation lambda": VARIATION_LAMBDA, "replace pooling": REPLACE_POOLING,
                     "square error": SQUARE_ERROR, "gram matrix norm": GRAM_MATRIX_NORM, 
                     "optimizer": OPTIMIZER, "multipule styles": MULTIPULE_STYLES}

    date = datetime.today()

    output_folder = os.path.join(BASE_OUTPUT_DIR, date.strftime(
        "%Y-%m-%d"), date.strftime("%H:%M:%S"))

    os.makedirs(output_folder)

    with open(os.path.join(output_folder, "configuration.json"), "w", encoding="utf-8") as f:
        json.dump(configuration, f, ensure_ascii=False, indent=4)

    print(configuration)

    torch.manual_seed(SEED)

    CONTENT_FOLDER = "content"
    STYLE_FOLDER = "style_photos"


    STYLE_FOLDER = "high_res_styles"

    content_images = read_images(CONTENT_FOLDER)
    style_images = read_images(STYLE_FOLDER)

    STYLE_IMGS_WEIGTHS ={"Edvard_Munch_The_Scream" : 0.8 , "Vincent_van_Gogh_The_Starry_Nght" :0.2 } 

    # content_images = filter_images(content_images, ["stonehenge",  "tom", "tel_aviv"])
    # style_images = filter_images(style_images, ["Edvard_Munch_The_Scream"])


    # content_images = filter_images(content_images, ["stonehenge", "tom"])
    style_images = filter_images(style_images, ["Edvard_Munch_The_Scream", "Vincent_van_Gogh_The_Starry_Nght"])

    # content_images = filter_images(content_images, ['tom', 'boxing', 'obama', 'jumping_dog'])
    # style_images = filter_images(style_images, ["Vincent_van_Gogh_69"])
    
    content_images = filter_images(content_images, ['Lenna', 'tel_aviv'])


    # multiprocsess_run(content_images, style_images)

    run(content_images, style_images)