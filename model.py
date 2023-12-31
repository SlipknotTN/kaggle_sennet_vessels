"""
Unet model from "Fundus Images using Modified U-net Convolutional Neural Network" (Afolabi, 2020)
"""
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from config import ConfigParams


class UnetAfolabi(nn.Module):
    def __init__(self):
        super(UnetAfolabi, self).__init__()
        self.ds_block_1 = ConvBlock(in_channels=1, out_channels=64, num_blocks=1)
        self.ds_block_2 = ConvBlock(in_channels=64, out_channels=64, num_blocks=3)
        self.ds_block_3 = ConvBlock(in_channels=64, out_channels=64, num_blocks=3)
        self.ds_block_4 = ConvBlock(in_channels=64, out_channels=64, num_blocks=3)
        self.bottom = ConvBlock(in_channels=64, out_channels=64, num_blocks=3)
        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsampling_2x = nn.UpsamplingNearest2d(scale_factor=2.0)
        self.us_block_4 = ConvBlock(in_channels=128, out_channels=32, num_blocks=3)
        self.us_block_3 = ConvBlock(in_channels=96, out_channels=32, num_blocks=3)
        self.us_block_2 = ConvBlock(in_channels=96, out_channels=32, num_blocks=3)
        self.us_block_1 = ConvBlock(in_channels=96, out_channels=32, num_blocks=3)
        self.last_conv = nn.Conv2d(
            in_channels=32, out_channels=1, kernel_size=(1, 1), padding=0
        )

    def forward(self, x):
        # Input: 1x512x512
        x_1 = self.ds_block_1(x)  # 64x512x512
        x_1_down = self.max_pooling(x_1)  # 64x256x256
        x_2 = self.ds_block_2(x_1_down)  # 64x256x256
        x_2_down = self.max_pooling(x_2)  # 64x128x128
        x_3 = self.ds_block_3(x_2_down)  # 64x128x128
        x_3_down = self.max_pooling(x_3)  # 64x64x64
        x_4 = self.ds_block_4(x_3_down)  # 64x64x64
        x_4_down = self.max_pooling(x_4)  # 64x32x32
        x_bottom = self.bottom(x_4_down)  # 64x32x32
        x_4_up = self.upsampling_2x(x_bottom)  # 64x64x64
        x_4_cat = torch.cat([x_4, x_4_up], dim=1)  # 128x64x64
        x_4_up_conv = self.us_block_4(x_4_cat)  # 32x64x64
        x_3_up = self.upsampling_2x(x_4_up_conv)  # 32x128x128
        x_3_cat = torch.cat([x_3, x_3_up], dim=1)  # 96x128x128
        x_3_up_conv = self.us_block_3(x_3_cat)  # 32x128x128
        x_2_up = self.upsampling_2x(x_3_up_conv)  # 32x256x256
        x_2_cat = torch.cat([x_2, x_2_up], dim=1)  # 96x256x256
        x_2_up_conv = self.us_block_2(x_2_cat)  # 32x256x256
        x_1_up = self.upsampling_2x(x_2_up_conv)  # 32x512x512
        x_1_cat = torch.cat([x_1, x_1_up], dim=1)  # 96x512x512
        x_1_up_conv = self.us_block_1(x_1_cat)  # 32x512x512
        return self.last_conv(x_1_up_conv)  # 1x512x512


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        """
        Args:
            in_channels: number of input channels only for the first  convolution layer
            out_channels: number of the output channels of all the convolution layers
            which is also the in_channels for the next step
            num_blocks: number of conv+leaky_relu+bn blocks
        """
        super(ConvBlock, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            input_channels = in_channels if i == 0 else out_channels
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=out_channels,
                        kernel_size=(3, 3),
                        padding=1,
                    ),
                    nn.LeakyReLU(negative_slope=0.018),
                    nn.BatchNorm2d(out_channels),
                )
            )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def init_model(config: ConfigParams) -> nn.Module:
    if config.model_name == "unet_afolabi":
        model = UnetAfolabi()
    elif config.model_smp_model is not None:
        model = init_smp_model(config)
    else:
        raise Exception("Unable to initialize the model, please check the config")
    return model


def init_smp_model(config: ConfigParams) -> nn.Module:
    if config.model_smp_model != "unet":
        raise Exception("Only unet model is supported")
    model = smp.Unet(
        encoder_name=config.smp_encoder,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=config.smp_encoder_weights,  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=config.model_input_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,  # model output channels (number of classes in your dataset)
    )
    return model
