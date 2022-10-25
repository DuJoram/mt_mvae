import abc
from typing import Tuple, Optional

import torch


class Decoder(torch.nn.Module, abc.ABC):
    def __init__(self):
        super(Decoder, self).__init__()

    @abc.abstractmethod
    def forward(
        self,
        style_latents: Optional[torch.Tensor],
        class_latents: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], ...]:
        pass
