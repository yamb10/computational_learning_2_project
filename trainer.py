import os
from matplotlib import pyplot as plt
from enum import Enum
from torchvision import transforms
import torch
from tqdm import trange

from total_variation_loss import TotalVariationLoss

class Starting_Place(Enum):
    RANDOM = 0
    BLACK = 1
    WHITE = 2
    HORIZONTAL_SPLIT = 3
    VERTICAL_SPLIT = 4
    CHECKERS = 5





class Trainer:
    def __init__(self, ephoch_num, input_size, criterion, model, device="cuda",
                  random_starts=1, verbose=False, optimizer='lbfgs',
                  variation_labmda=0, save_every=-1, save_path=None, multiple_styles= False):
        self.ephoch_num = ephoch_num
        self.input_size = input_size
        self.criterion = criterion
        self.device = device
        self.random_starts = random_starts
        self.verbose = verbose
        self.optimizer_type = optimizer.lower()
        self.variation_labmda = variation_labmda
        
        self.model = model.to(device)
        self.regularizer = TotalVariationLoss()

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])  # following the documentation of VGG19
        self.resize = transforms.Resize(input_size[1:]) 

        self.transform = transforms.Compose([self.resize, self.normalize])
        
        self.save_every = save_every
        self._save_path = save_path
        self.multiple_styles= multiple_styles 

    def train(self, style_image, content_image, start=Starting_Place.RANDOM, checkers_num=2):
        loss_values = []
        if self.multiple_styles:
            style_image = torch.cat([self.transform(i) for i in style_image ]).to(self.device)
        else:
            style_image = self.transform(style_image).to(self.device)
        content_image = self.transform(content_image).to(self.device)

        style_image.requires_grad_(False)
        content_image.requires_grad_(False)

        style_outputs = self.model(style_image)
        content_outputs = self.model(content_image)

        dims = [self.random_starts] + list(self.input_size)

        if start == Starting_Place.RANDOM:  
            inputs = torch.rand([self.random_starts] + list(self.input_size),
                                 requires_grad=True, device=self.device)
        elif start == Starting_Place.BLACK:
            inputs = torch.zeros([self.random_starts] + list(self.input_size),
                                 requires_grad=True, device=self.device)
        elif start == Starting_Place.WHITE:
            inputs = torch.ones([self.random_starts] + list(self.input_size),
                                 requires_grad=True, device=self.device)
        elif start == Starting_Place.HORIZONTAL_SPLIT or start == Starting_Place.VERTICAL_SPLIT:
            if  start == Starting_Place.HORIZONTAL_SPLIT:
                axis = 2
            else:
                axis = 3
            dims[axis] //= 2 
            white = torch.ones(dims,requires_grad=False, device=self.device)
            black = torch.zeros(dims,requires_grad=False, device=self.device)
            inputs =  torch.cat([white, black], dim=axis)
            inputs.requires_grad_(True)

        elif start == Starting_Place.CHECKERS:
            assert all(dims[i] % checkers_num == 0 for i in [2,3]) 
            block_size = (dims[2] // checkers_num,  dims[3] // checkers_num)
            checkers_num //= 2
            inputs = torch.kron(torch.Tensor([[1, 0] * checkers_num, [0, 1] * checkers_num] * checkers_num), torch.ones(block_size))
 

            inputs = inputs.repeat(dims[:2] + [1,1])
            
            inputs = inputs.detach().to(device=self.device).requires_grad_() 
            
        else:
            inputs = start
            # assert(all(x == y for x, y in zip(inputs.shape[1:], self.input_size)))


        if self.optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam([inputs])
        elif self.optimizer_type == 'lbfgs':
            self.optimizer = torch.optim.LBFGS([inputs])

        for epoch_num in trange(self.ephoch_num, disable=not self.verbose, ncols=150):
            
            def closure():
                self.optimizer.zero_grad()
                outputs = self.model(self.normalize(inputs))
                
                loss = self.criterion(outputs, style_outputs, content_outputs)

                if self.variation_labmda != 0:
                    loss += self.variation_labmda * self.regularizer(inputs)

                loss.backward()
                loss_values.append(loss.item())
                return loss


            with torch.no_grad():
                if self.save_every > 0 and (epoch_num % self.save_every == 0 or (self.ephoch_num == (epoch_num + 1))):
                    for i, input in enumerate(inputs):
                        img = transforms.ToPILImage()(input.clamp(0, 1))
                        if self.save_path is None:
                            plt.imshow(img)
                        else:
                            img.save(os.path.join(self.save_path, f"{epoch_num}_{i}.png"))
                            
            self.optimizer.step(closure)

        with torch.no_grad():
            inputs.clamp_(0, 1)


            #     for i in inputs:
            #         i.clamp_(0, 255)
            # if before.equal(inputs):
            #     raise RuntimeError

        return inputs, loss_values

    @property
    def save_path(self):
        return self._save_path   


    @save_path.setter
    def save_path(self, value):
        self._save_path = value
        os.makedirs(self._save_path)