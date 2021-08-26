from PIL import Image
from torchvision import transforms
import os

class NamedImage:
    def __init__(self, path):
        self.image = self.load_image(path)
        self.image_name = os.path.basename(path)

    @staticmethod
    def load_image(path):
        image = Image.open(path)
        x = transforms.functional.to_tensor(image)
        x.unsqueeze_(0)

        return x
    
    @property
    def name(self):
        return os.path.splitext(self.image_name)[0]