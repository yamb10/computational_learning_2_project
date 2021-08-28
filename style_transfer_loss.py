import torch
import torch.nn as nn

class StyleTansferLoss(nn.Module):
    def __init__(self, style_layers, content_layers, style_weights=None, content_weights=None, device="cuda", alpha=1, beta=1, square_error=True) -> None:
        super().__init__()
        if style_weights is None:
            self.style_weights = torch.ones(len(style_layers), device=device, requires_grad=False) / len(style_layers)
        else:
            style_weights = [style_weights[key] for key in style_layers]
            assert len(style_weights) == len(style_layers)
            self.style_weights = style_weights
        
        if content_weights is None:
            self.content_weights = torch.ones(len(content_layers), device=device, requires_grad=False) / len(content_layers)
        else:
            content_weights = [content_weights[key] for key in content_layers]
            assert len(content_weights) == len(content_layers)
            self.content_weights = content_weights
        
        self.alpha = alpha
        self.beta = beta
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.square_error = square_error



    @staticmethod
    def gram_matrix(A):
        """
        A - Tensor
        """
        bs, a, b, c = A.size()
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = A.view(bs, a, b * c)  # resise F_XL into \hat F_XL

        # compute the gram product
        G = torch.bmm(features, features.transpose(1, 2))

        # we 'normalize' the values of the gram matrix  \\\\ Yam's comment: why tf would you do that????? the def is called Gram matrix not 'normalized Gram matrix' -_-
        # by dividing by the number of element in each feature maps.
        return G  # .div(a * b * c )


    def compute_style_loss(self, P, F):
        """
        names of variables are uninformative, but follow the notation in the Gatys paper.
        P - list. a genereted vector of tensor of layers *batch size* num of filter* size of filter 
        P[l] is a matrix of size Nl x Ml, where Nl is the number of filters in the l-th layer of the VGG net and Ml is the number of elements in each filter.
            P contains the results of applying the net to the **original** image.

        F - list. similar to F, but applied on the **generated** image.
        weights - a manual rescaling weight given to each layer. If given, has to be a Tensor of size len(P)
        """

        num_layers = len(P)
        
        loss = 0
        for l in range(num_layers):
            p, f = P[l], F[l]
            _, c, d, e = p.size()
            a, g = self.gram_matrix(p), self.gram_matrix(f)  # check with batch size > 1
            if self.square_error:
                loss += self.style_weights[l] * 1/((2*d*e*c)**2) * torch.linalg.norm(a-g) ** 2
            else:
                loss += self.style_weights[l] * 1/((2*d*e*c)) * torch.sum(torch.abs(a-g))
        return loss


    def compute_layer_content_loss(self, C, G):
        """
        C-  list. a genereted vector of tensor of batch size* num of filter* size of filter 
        G-  list. a given vector of tensor of batch size* num of filter* size of filter 
        """

        num_layers = len(C)

        loss = 0  # torch.zeros(batch_size, device=C.device)
        for l in range(num_layers):
            p, f = C[l], G[l]
            if self.square_error:
                loss += self.content_weights[l] * torch.linalg.norm(p-f) ** 2  # fixed the formula
            else:
                loss += self.content_weights[l] * torch.sum(torch.abs(p-f))  # fixed the formula

        return loss / 2

        # return sum(map(lambda x: torch.linalg.norm(x[0]-x[1]) ** 2, zip(C, G))) / 2


    def forward(self, outputs, style_outputs, content_outputs):

        x_style = [outputs[key] for key in outputs.keys() if key in self.style_layers]  # cant stack as tensors might have different shapes for different layers
        x_con = [outputs[key] for key in outputs.keys() if key in self.content_layers]
        y_style = [style_outputs[key]for key in style_outputs.keys() if key in self.style_layers]
        y_con = [content_outputs[key]for key in content_outputs.keys() if key in self.content_layers]

        return (self.alpha * self.compute_layer_content_loss(x_con, y_con)
                + self.beta*self.compute_style_loss(x_style, y_style))
