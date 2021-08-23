import torch
from torchvision.models import vgg19
from torchvision import transforms
import pandas as pd
import numpy as np
from tqdm import tqdm, trange

import matplotlib.pyplot as plt
from model import ByLayerModel

from PIL import Image



def load_image(path):
    image = Image.open(path)
    x = transforms.functional.to_tensor(image)
    x.unsqueeze_(0)

    return x

def gram_matrix(A):
    """
    A - Tensor
    """
    a, b, c = A.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = A.view(a * b, c)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c )

def compute_style_loss(P, F):
    """
    P - Tensor. a genereted vector of tensor of layers *batch size* num of filter* size of filter 
     P[l] is a matrix of size Nl x Ml, where Nl is the number of filters in the l-th layer of the VGG net and Ml is the number of elements in each filter.
        P contains the results of applying the net to the **original** image.

    F - Tensor. similar to F, but applied on the **generated** image.
    """
    
    num_layers, batch_size, c, d,e= C.size()
    C=C.view(num_layers,batch_size,c,d*e)
    C=G.view(num_layers,batch_size,c,d*e)
    w=torch.ones(num_layers)/5
    loss = torch.zeros(batch_size)
    for l in range(num_layers):
        p, f = P[l], F[l]
        a, g = gram_matrix(p), gram_matrix(f) #check with batch size > 1
        loss += w[l]* 1/((2*d*e)**2) * torch.linalg.norm(a-g,dim=(1,2)) ** 2 
    return loss
        


def compute_layer_content_loss(C, G):  
    """
    C-  a genereted vector of tensor of batch size* num of filter* size of filter 
    G-  a given vector of tensor of batch size* num of filter* size of filter 
    """
    num_layers, batch_size, c, d,e=C.size()
    C=C.view(num_layers,batch_size,c,d*e)
    G=G.view(num_layers,batch_size,c,d*e)
    w=torch.ones(num_layers)/5
    loss = torch.zeros(batch_size)
    for l in range(num_layers):
        p, f = C[l], G[l]
        loss += w[l]* 1/2 * torch.linalg.norm(p-f,dim=(1,2)) ** 2# unsure if this is the formula 
    return loss
         

def compute_content_loss(outputs, style_outputs, style_names, content_outputs, content_names, alpha=1, beta=1):
    """

    """
    x_style =torch.stack([ outputs[key] for key in outputs.keys() if key in style_names])
    x_con =torch.stack([ outputs[key]  for key in outputs.keys()  if key in content_names ] )
    y_style = torch.stack([ style_outputs[key] for key in style_outputs.keys()  if key in style_names ])
    y_con = torch.stack([ content_outputs[key] for key in content_outputs.keys() if key in content_names ])

    
    return alpha*compute_layer_content_loss(x_con,y_con)+ beta*compute_layer_content_loss(x_style, y_style)
    
    


def train(ephoch_num, input_size, style_image, content_image, alpha=1, beta=1):

    inputs = torch.rand([1] + list(input_size), requires_grad=True)

    layers = ["conv1_1", "relu1_1", "conv1_2","relu1_2", "maxpool1",
                                                         "conv2_1", "relu2_1", "conv2_2", "relu2_2", "maxpool2",
                                                         "conv3_1", "relu3_1", "conv3_2", "relu3_2", "conv3_3",
                                                            "relu3_3", "conv3_4", "relu3_4","maxpool3",
                                                         "conv4_1", "relu4_1", "conv4_2", "relu4_2", "conv4_3", 
                                                            "relu4_3", "conv4_4", "relu4_4", "maxpool4",
                                                        "conv5_1", "relu5_1", "conv5_2", "relu5_2", "conv5_3", 
                                                            "relu5_3", "conv5_4", "relu5_4", "maxpool5"]
    model = vgg19(pretrained=True)
    splitted_model = ByLayerModel(model.features, names=layers)

    for p in model.parameters():
        p.requires_grad_(False)

    style_image.requires_grad_(False)
    content_image.requires_grad_(False)

    loss_values = [] 

    optimizer = torch.optim.Adam([inputs])
    criterion = compute_content_loss
    
    style_names = ["conv1_1"]
    content_names = ["conv1_1"]

    assert(set(style_names).issubset(set(layers)))
    assert(set(content_names).issubset(set(layers)))
    

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    
    transform = transforms.Compose([transforms.Resize(input_size[1:]), normalize])

    style_image = transform(style_image)
    content_image = transform(content_image)

    
    style_outputs = splitted_model(style_image)
    content_outputs = splitted_model(content_image)

    for epcoh_num in trange(ephoch_num):
    
        outputs = splitted_model(normalize(inputs))
        
        loss = criterion(outputs, style_outputs, style_names, content_outputs, content_names, alpha=alpha, beta=beta)
        
        loss.backward()

        loss_values.append(loss.item())

        optimizer.step()
        optimizer.zero_grad()


        

    return inputs, loss_values 







if __name__ == "__main__":
    EPOCH_NUM = 100
    INPUT_SIZE = (3, 224, 224)

    STYLE_IMAGE = "images/images/Francisco_Goya/Francisco_Goya_79.jpg"
    CONTENT_IMAGE = "content/tom.jpg"

    style_image = load_image(STYLE_IMAGE)
    content_image = load_image(CONTENT_IMAGE)



    inputs, loss_values = train(EPOCH_NUM, INPUT_SIZE, style_image, content_image)

    plt.plot(np.arange(len(loss_values)) + 1, loss_values)
    plt.savefig('loss.png')
    
    # img = inputs.detach().cpu().numpy()
    img = transforms.ToPILImage()(inputs.squeeze(0))
    
    img.save("output.png")

    

    