import torch
import torch.nn as nn
import torch.nn.init as init

from test_rig.config import ONNX_MODEL_PATH, INPUT_SHAPE, MODEL_LENGTH


# https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html

class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor, inplace=False):
        super(SuperResolutionNet, self).__init__()

        self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)


def separate_custom_func(x):
    return x * 2


class OnnxModule(nn.Module):
    def __init__(self,
                 in_channels=INPUT_SHAPE[1],
                 out_channels=INPUT_SHAPE[1],
                 kernel_size=1,
                 ):
        super().__init__()
        for i in range(MODEL_LENGTH):
            self.__setattr__(f"conv{i}", nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
            ))

    def forward(self, x):
        for i in range(MODEL_LENGTH):
            x = self.__getattr__(f"conv{i}")(x)
        return {"output": x}


def save():
    model = OnnxModule()
    # model = SuperResolutionNet(upscale_factor=3)
    # model = nn.CosineSimilarity()

    onnx_program = torch.onnx.export(
        model,
        (torch.rand(INPUT_SHAPE),
         #  torch.rand(INPUT_SHAPE)
         ),
        # Must be torch array
        # (
        #     torch.rand(INPUT_SHAPE, dtype=INPUT_TYPE_PT),
        #  torch.rand(INPUT_SHAPE, dtype=INPUT_TYPE_PT),
        # ),
        # (torch.randint(0,255,INPUT_SHAPE,dtype=torch.int16),),
        dynamo=True,
        # report=True,

        optimize=True,
        # verify=True,
    )

    onnx_program.save(ONNX_MODEL_PATH)
