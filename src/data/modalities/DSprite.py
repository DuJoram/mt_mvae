import torch

from utils.save_samples import write_samples_img_to_file
from .modality import Modality


class DSprite(Modality):
    def __init__(self, name: str, likelihood: str = "bernoulli"):
        super(DSprite, self).__init__(name=name, likelihood=likelihood)
        self.data_size = torch.Size((1, 64, 64))
        self.gen_quality_eval = True
        self.file_suffix = ".png"

    def save_data(self, d, fn, args):
        img_per_row = args["img_per_row"]
        write_samples_img_to_file(d, fn, img_per_row)

    def plot_data(self, d: torch.Tensor):
        return d
