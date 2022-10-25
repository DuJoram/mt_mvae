import torch

from utils.save_samples import write_samples_img_to_file
from .modality import Modality


class MimicPA(Modality):
    def __init__(self):
        super(MimicPA, self).__init__(name="PA", likelihood="laplace")
        self.data_size = torch.Size((1, 128, 128))
        self.gen_quality_eval = True
        self.file_suffix = ".png"

    def save_data(self, d, fn, args):
        img_per_row = args["img_per_row"]
        write_samples_img_to_file(d, fn, img_per_row)

    def plot_data(self, d):
        p = d.repeat(1, 1, 1, 1)
        return p
