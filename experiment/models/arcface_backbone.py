from collections import namedtuple

import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class BottleneckIR(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, 1, stride, bias=False),
                nn.BatchNorm2d(depth),
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, 3, 1, 1, bias=False),
            nn.BatchNorm2d(depth),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, 3, stride, 1, bias=False),
            nn.BatchNorm2d(depth),
        )

    def forward(self, x):
        return self.res_layer(x) + self.shortcut_layer(x)


class Bottleneck(namedtuple("Block", ["in_channel", "depth", "stride"])):
    pass


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [
        Bottleneck(depth, depth, 1) for _ in range(num_units - 1)
    ]


def get_blocks(num_layers):
    if num_layers == 50:
        return [
            get_block(64, 64, 3),
            get_block(64, 128, 4),
            get_block(128, 256, 14),
            get_block(256, 512, 3),
        ]
    raise ValueError(f"Unsupported num_layers: {num_layers}")


class Backbone(nn.Module):
    def __init__(self, input_size=(112, 112), num_layers=50, mode="ir"):
        super().__init__()
        assert input_size[0] in [112, 224]
        assert num_layers in [50, 100, 152]
        assert mode == "ir"

        blocks = get_blocks(num_layers)
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(0.4),
            Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.BatchNorm1d(512, affine=False),
        )

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    BottleneckIR(
                        bottleneck.in_channel,
                        bottleneck.depth,
                        bottleneck.stride,
                    )
                )
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        return self.output_layer(x)

