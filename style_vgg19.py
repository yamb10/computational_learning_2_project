
from by_layer_model import ByLayerModel
import torch
import torch.nn as nn
from torchvision.models import vgg19

class StyleVGG19(ByLayerModel):
    def __init__(self, replace_pooling=False) -> None:

        self.layers = ["conv1_1", "relu1_1", "conv1_2", "relu1_2", "maxpool1",
              "conv2_1", "relu2_1", "conv2_2", "relu2_2", "maxpool2",
              "conv3_1", "relu3_1", "conv3_2", "relu3_2", "conv3_3",
              "relu3_3", "conv3_4", "relu3_4", "maxpool3",
              "conv4_1", "relu4_1", "conv4_2", "relu4_2", "conv4_3",
              "relu4_3", "conv4_4", "relu4_4", "maxpool4",
              "conv5_1", "relu5_1", "conv5_2", "relu5_2", "conv5_3",
              "relu5_3", "conv5_4", "relu5_4", "maxpool5"]
        
        model = vgg19(pretrained=True).features

        for p in model.parameters():
            p.requires_grad_(False)

        if replace_pooling:
            self.layers = [layer.replace("max", "avg") for layer in self.layers]

            for idx, layer in enumerate(model):
                if isinstance(layer, nn.MaxPool2d):
                    model[idx] = nn.AvgPool2d(kernel_size=layer.kernel_size, 
                                              stride=layer.stride,
                                              padding=layer.padding,
                                              ceil_mode=layer.ceil_mode)



        super().__init__(model, names=self.layers)

        