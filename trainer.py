from torchvision import transforms
import torch
from tqdm import trange

from total_variation_loss import TotalVariationLoss

class Trainer:
    def ___init__(self, ephoch_num, input_size, criterion, model, device="cuda",
                  random_starts=1, verbose=False, optimizer='lbfgs',
                  variation_labmda=0):
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
        self.resize = transforms.Resize(input_size) 

        self.transform = transforms.Compose([self.resize, self.normalize])
        

    def train(self, style_image, content_image):
        loss_values = []
        style_image = self.transform(style_image).to(self.device)
        content_image = self.transform(content_image).to(self.device)

        style_image.requires_grad_(False)
        content_image.requires_grad_(False)

        style_outputs = self.model(style_image)
        content_outputs = self.model(content_image)

        inputs = torch.rand([self.random_starts] + list(self.input_size),
                            requires_grad=True, device=self.device)

        for epcoh_num in trange(self.ephoch_num, disable=not self.verbose, ncols=150):
            
            def closure():
                self.optimizer.zero_grad()
                outputs = self.model(self.normalize(inputs))

                loss = self.criterion(outputs, style_outputs, content_outputs)

                if self.variation_labmda != 0:
                    loss += self.variation_labmda * self.regularizer(inputs)

                loss.backward()
                loss_values.append(loss.item())
                return loss

            self.optimizer.step(closure)

        with torch.no_grad():
            inputs.clamp_(0, 1)


            #     for i in inputs:
            #         i.clamp_(0, 255)
            # if before.equal(inputs):
            #     raise RuntimeError

        return inputs, loss_values
