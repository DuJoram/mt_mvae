import torch
from torch import nn

from typing import List

class DSpritesImageClassifier(nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            num_shape_labels: int = 3,
            num_scale_labels: int = 6,
            num_orientation_labels: int = 40,
            num_x_position_labels: int = 32,
            num_y_position_labels: int = 32
    ):
        super(DSpritesImageClassifier, self).__init__()

        sequence: List[nn.Module] = list()
        self._shared_sequence = nn.Sequential(
            nn.Conv2d(  # (B, 1|3, 64, 64) -> (B, 8, 32, 32)
                in_channels=in_channels,
                out_channels=8,
                kernel_size=5,
                stride=2,
                padding=2
            ),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Conv2d(  # -> (B, 16, 16, 16)
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Conv2d(  # -> (B, 32, 8, 8)
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Conv2d(  # -> (B, 64, 4, 4)
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Flatten()  # -> (B, 1024)
        )

        self._shape_classifier = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=num_shape_labels)

        )

        self._scale_classifier = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=num_scale_labels)
        )

        self._orientation_classifier = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=num_orientation_labels)
        )

        self._x_position_classifier = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=num_x_position_labels)
        )

        self._y_position_classifier = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=num_y_position_labels)
        )

    def forward(self, input_batch):
        shared = self._shared_sequence(input_batch)
        shape = self._shape_classifier(shared)
        scale = self._scale_classifier(shared)
        orientation = self._orientation_classifier(shared)
        x_position = self._x_position_classifier(shared)
        y_position = self._y_position_classifier(shared)

        if not self.training:
            shape = torch.softmax(shape, dim=-1)
            scale = torch.softmax(scale, dim=-1)
            orientation = torch.softmax(orientation, dim=-1)
            x_position = torch.softmax(x_position, dim=-1)
            y_position = torch.softmax(y_position, dim=-1)

        return shape, scale, orientation, x_position, y_position
