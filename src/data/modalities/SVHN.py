import torch
from PIL import Image
from torchvision import transforms

from utils.save_samples import write_samples_img_to_file
from .modality import Modality


class SVHN(Modality):
    def __init__(self, plot_image_size):
        super(SVHN, self).__init__(name="svhn", likelihood="laplace")
        self.data_size = torch.Size((3, 32, 32))
        self.plot_image_size = plot_image_size
        self.transform_plot = self.get_plot_transform()
        self.gen_quality_eval = True
        self.file_suffix = ".png"

    def save_data(self, d, fn, args):
        img_per_row = args["img_per_row"]
        write_samples_img_to_file(d, fn, img_per_row)

    def plot_data(self, d):
        out = self.transform_plot(d.squeeze(0).cpu()).cuda().unsqueeze(0)
        return out

    def get_plot_transform(self):
        transf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(
                    size=list(self.plot_image_size)[1:], interpolation=Image.BICUBIC
                ),
                transforms.ToTensor(),
            ]
        )
        return transf
