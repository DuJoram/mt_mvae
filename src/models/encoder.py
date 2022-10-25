import torch
from typing import Tuple, Optional

import abc

class Encoder(torch.nn.Module, abc.ABC):
    def __init__(self):
        super(Encoder, self).__init__()

    @abc.abstractmethod
    def forward(self, input_batch) -> Tuple[Optional[torch.Tensor],...]:
        pass
