import torch

from utils import plot
from utils.save_samples import write_samples_text_to_file
from utils.text import tensor_to_text
from .modality import Modality


class MimicText(Modality):
    def __init__(
        self,
        len_sequence,
        alphabet,
        plot_image_size,
        font,
    ):
        super(MimicText, self).__init__(name="text", likelihood="categorical")
        self.alphabet = alphabet
        self.len_sequence = len_sequence
        self.data_size = torch.Size((len(alphabet), len_sequence))
        self.plot_image_size = plot_image_size
        self.font = font
        self.gen_quality_eval = False
        self.file_suffix = ".txt"

    def save_data(self, d, fn, args):
        write_samples_text_to_file(tensor_to_text(self.alphabet, d.unsqueeze(0)), fn)

    def plot_data(self, data):
        out = plot.text_to_pil(
            data.unsqueeze(0), self.plot_image_size, self.alphabet, self.font
        )
        return out
