import torch

from .modality import Modality
from utils.save_samples import write_samples_img_to_file


class CMNIST(Modality):
    def __init__(self, name: str):
        super(CMNIST, self).__init__(name=name, likelihood="laplace")
        self.data_size = torch.Size((3, 28, 28))
        self.gen_quality_eval = True
        self.file_suffix = ".png"
        # self.transform = transforms.Compose([transforms.ToTensor()])

    def save_data(self, d, fn, args):
        img_per_row = args["img_per_row"]
        write_samples_img_to_file(d, fn, img_per_row)

    def plot_data(self, d):
        # out = self.transform(d.squeeze(0).cpu()).cuda().unsqueeze(0)
        # return out
        return d
