import torch
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

from by_layer_model import ByLayerModel
from named_image import NamedImage
from style_transfer_loss import StyleTansferLoss



def train(ephoch_num, input_size, criterion, style_image, content_image, device="cuda", random_starts=1, verbose=True):

    inputs = torch.rand([random_starts] + list(input_size),
                        requires_grad=True, device=device)

    layers = ["conv1_1", "relu1_1", "conv1_2", "relu1_2", "maxpool1",
              "conv2_1", "relu2_1", "conv2_2", "relu2_2", "maxpool2",
              "conv3_1", "relu3_1", "conv3_2", "relu3_2", "conv3_3",
              "relu3_3", "conv3_4", "relu3_4", "maxpool3",
              "conv4_1", "relu4_1", "conv4_2", "relu4_2", "conv4_3",
              "relu4_3", "conv4_4", "relu4_4", "maxpool4",
              "conv5_1", "relu5_1", "conv5_2", "relu5_2", "conv5_3",
              "relu5_3", "conv5_4", "relu5_4", "maxpool5"]
    model = vgg19(pretrained=True).to(device)
    splitted_model = ByLayerModel(model.features, names=layers)

    for p in model.parameters():
        p.requires_grad_(False)

    style_image.requires_grad_(False)
    content_image.requires_grad_(False)

    loss_values = []

    # optimizer = torch.optim.SGD([inputs], lr=1e-5)
    # criterion = compute_content_loss

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])  # following the documentation of VGG19

    transform = transforms.Compose(
        [transforms.Resize(input_size[1:]), normalize])

    style_image = transform(style_image).to(device)
    content_image = transform(content_image).to(device)

    optimizer = torch.optim.Adam([inputs], lr=1e-3)

    style_outputs = splitted_model(style_image)
    content_outputs = splitted_model(content_image)

    for epcoh_num in trange(ephoch_num, disable=not verbose):

        outputs = splitted_model(normalize(inputs))

        loss = criterion(outputs, style_outputs, content_outputs)

        loss.backward()

        loss_values.append(loss.item())

        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            inputs.clamp_(0, 1)



        #     for i in inputs:
        #         i.clamp_(0, 255)
        # if before.equal(inputs):
        #     raise RuntimeError

    return inputs, loss_values


def run_content_image(content_image, style_images):

    # assert(set(style_names).issubset(set(layers)))
    # assert(set(content_names).issubset(set(layers)))

    criterion = StyleTansferLoss(style_layers=STYLE_NAMES, content_layers=CONTENT_NAMES, alpha=ALPHA, beta=BETA, device=DEVICE, content_weights=CONTENT_WEIGHTS)

    for style_image in style_images:

        inputs, loss_values = train(EPOCH_NUM, INPUT_SIZE, criterion, style_image.image, content_image.image,
                                    random_starts=RANDOM_STARTS, verbose=True, device=DEVICE)

        plt.rcParams["figure.figsize"] = (16, 9)

        plt.semilogy(np.arange(len(loss_values)) + 1,
                     loss_values, label=f"{style_image.name}")
        plt.legend()
        plt.savefig(os.path.join(output_folder,
                    f"{content_image.name}_loss.png"))

        folder_name = os.path.join(output_folder, f"{content_image.name}", f"{style_image.name}")
        os.makedirs(folder_name)

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
    EPOCH_NUM = 15000
    INPUT_SIZE = (3, 512, 512)
    SEED = 7442
    RANDOM_STARTS = 1
    ALPHA = 1
    BETA = 5e7
    DEVICE = "cuda"
    STYLE_NAMES = ["conv1_1", "conv2_1", "conv3_1", "conv4_1"]
    CONTENT_NAMES = ["conv4_2", "conv5_2"]
    CONTENT_WEIGHTS = {"conv4_2": 0.333, "conv5_2":0.666}

    configuration = {"epoch num": EPOCH_NUM, "input size": INPUT_SIZE, "SEED": SEED,
                     "RANDOM STARTS": RANDOM_STARTS, "ALPHA": ALPHA, "BETA": BETA, 
                     "device": DEVICE, "style names": STYLE_NAMES, "content names": CONTENT_NAMES}

    date = datetime.today()

    output_folder = os.path.join("outputs_all", date.strftime(
        "%Y-%m-%d"), date.strftime("%H:%M:%s"))

    os.makedirs(output_folder)

    with open(os.path.join(output_folder, "configuration.json"), "w", encoding="utf-8") as f:
        json.dump(configuration, f, ensure_ascii=False, indent=4)

    print(configuration)

    torch.manual_seed(SEED)

    CONTENT_FOLDER = "content"
    STYLE_FOLDER = "style_photos"

    content_images = read_images(CONTENT_FOLDER)
    style_images = read_images(STYLE_FOLDER)

    content_images = filter_images(content_images, ["tel_aviv"])
    style_images = filter_images(style_images, ["Vincent_van_Gogh_69"])


    # multiprocsess_run(content_images, style_images)

    run(content_images, style_images)