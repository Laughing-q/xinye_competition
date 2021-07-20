from torch import nn
import torch
from .coatnet_utils import MBConvBlock, FFN, RelativeAttention, PatchEmded, DownSample
from timm.models.layers import DropPath
from torch.nn import init
# TODO:
#  1.确定shortcut部分bn层的使用；
#  2.确定激活函数使用；


class StemStage(nn.Module):
    """
    原论文中输入分辨率为224×224，此时s0两倍下采样，婼输入分辨率为112×112，不下采样
    """

    def __init__(self, out_chs, repeat_num, img_size=224):
        super(StemStage, self).__init__()
        self.output_channel = out_chs[0]
        self.activation = nn.GELU()
        self.norlization = nn.BatchNorm2d(out_chs[0])
        self.img_size = img_size
        self.stage0_model = nn.Sequential()

        self.conv_downsample = nn.Conv2d(in_channels=3,
                                         out_channels=self.output_channel,
                                         kernel_size=3,
                                         stride=2 if self.img_size == 224 else 1,
                                         padding=1,
                                         )

        self.conv2 = nn.Conv2d(in_channels=self.output_channel,
                               out_channels=self.output_channel,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               )

        self.s1 = nn.Sequential(self.conv_downsample,
                                self.norlization,
                                self.activation,
                                )

        self.s2 = nn.Sequential(self.conv2,
                                self.norlization,
                                self.activation,
                                )

        for i in range(repeat_num[0]):
            if i == 0:
                self.stage0_model.add_module('stage0_layer{}'.format(2 * i), self.s1)
            else:
                self.stage0_model.add_module('stage0_layer{}'.format(2 * i), self.s2)
            self.stage0_model.add_module('stage0_layer{}'.format(2 * i + 1), self.s2)

    def forward(self, input):
        x = self.stage0_model(input)

        return x


class StageOne(nn.Module):
    """Stage one"""

    def __init__(self, out_chs, repeat_num, drop_path_ratio=0.3):
        super(StageOne, self).__init__()
        self.output_channel = out_chs[1]
        self.activation = nn.GELU()
        self.norlization = nn.BatchNorm2d(out_chs[1])
        self.id_branch_downsample = DownSample(is_attention=False)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio else nn.Identity()
        self.stage1_model = nn.Sequential()

        self.conv_downsample = nn.Sequential(nn.Conv2d(in_channels=out_chs[0],
                                                       out_channels=out_chs[1],
                                                       kernel_size=3,
                                                       stride=2,
                                                       padding=1,
                                                       ),
                                             self.norlization,
                                             self.activation, )

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=out_chs[1],
                                             out_channels=out_chs[1],
                                             kernel_size=1,
                                             stride=1,
                                             padding=0,
                                             ),

                                   self.norlization,
                                   self.activation, )

        self.MBConv = nn.Sequential(MBConvBlock(ksize=3,
                                                input_filters=out_chs[1],
                                                output_filters=out_chs[1],
                                                stride=1,
                                                se_ratio=4,
                                                expand_ratio=1,
                                                image_size=112,
                                                ),
                                    self.norlization,
                                    self.activation,)

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=out_chs[0],
                                             out_channels=out_chs[1],
                                             kernel_size=1,
                                             stride=1,
                                             padding=0,
                                             ),
                                   )

        self.stage1_model.add_module('stage1_layer_downsample', self.conv_downsample)
        for i in range(repeat_num[1]):
            self.stage1_model.add_module('stage1_layer{}'.format(3 * i), self.conv1)
            self.stage1_model.add_module('stage1_layer{}'.format(3 * i + 1), self.MBConv)
            self.stage1_model.add_module('stage1_layer{}'.format(3 * i + 2), self.conv1)

    def forward(self, input):
        x = self.stage1_model(input)
        x = self.drop_path(x) + self.conv2(self.id_branch_downsample(input))
        x = self.activation(self.norlization(x))

        return x


class StageTwo(nn.Module):
    """Stage Two"""
    def __init__(self, out_chs, repeat_num, drop_path_ratio=0.3):
        super(StageTwo, self).__init__()
        self.output_channel = out_chs[2]
        self.activation = nn.GELU()
        self.norlization = nn.BatchNorm2d(out_chs[2])
        self.id_branch_downsample = DownSample(is_attention=False)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio else nn.Identity()
        self.stage2_model = nn.Sequential()

        self.conv_downsample = nn.Sequential(nn.Conv2d(in_channels=out_chs[1],
                                                       out_channels=out_chs[2],
                                                       kernel_size=3,
                                                       stride=2,
                                                       padding=1,
                                                       ),
                                             self.norlization,
                                             self.activation, )

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=out_chs[2],
                                             out_channels=out_chs[2],
                                             kernel_size=1,
                                             stride=1,
                                             padding=0,
                                             ),

                                   self.norlization,
                                   self.activation, )

        self.MBConv = nn.Sequential(MBConvBlock(ksize=3,
                                                input_filters=out_chs[2],
                                                output_filters=out_chs[2],
                                                stride=1,
                                                se_ratio=4,
                                                expand_ratio=1,
                                                image_size=112,
                                                ),
                                    self.norlization,
                                    self.activation, )

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=out_chs[1],
                                             out_channels=out_chs[2],
                                             kernel_size=1,
                                             stride=1,
                                             padding=0,
                                             ),
                                   )

        self.stage2_model.add_module('stage2_layer_downsample', self.conv_downsample)
        for i in range(repeat_num[2]):
            self.stage2_model.add_module('stage2_layer{}'.format(3 * i), self.conv1)
            self.stage2_model.add_module('stage2_layer{}'.format(3 * i + 1), self.MBConv)
            self.stage2_model.add_module('stage2_layer{}'.format(3 * i + 2), self.conv1)

    def forward(self, input):
        x = self.stage2_model(input)
        x = self.drop_path(x) + self.conv2(self.id_branch_downsample(input))
        x = self.activation(self.norlization(x))

        return x


class StageThree(nn.Module):
    def __init__(self, out_chs, repeat_num):
        super(StageThree, self).__init__()
        self.relative = RelativeAttention(dim=out_chs[3], drop_path_ratio=0.3, attn_drop=0.5, proj_drop=0.5)
        self.dowmsample = DownSample(is_attention=False)
        self.ffn = FFN(drop_path_ratio=0.3, dropout=0.5, dim=out_chs[3])
        self.patch_embed = PatchEmded(in_chans=out_chs[2], embed_dim=out_chs[3])
        self.stage3_model = nn.Sequential()

        self.stage3_model.add_module('stage3_layer_dowmsample', self.dowmsample)
        self.stage3_model.add_module('stage3_layer_patch_embed', self.patch_embed)
        for i in range(repeat_num[3]):
            self.stage3_model.add_module('stage3_layer{}'.format(2 * i), self.relative)
            self.stage3_model.add_module('stage3_layer{}'.format(2 * i + 1), self.ffn)

    def forward(self, input):
        x = self.stage3_model(input)

        return x


class StageFour(nn.Module):
    def __init__(self, out_chs, repeat_num):
        super(StageFour, self).__init__()
        self.relative = RelativeAttention(img_size=7, dim=out_chs[4], drop_path_ratio=0.3, attn_drop=0.5, proj_drop=0.5)
        self.dowmsample = DownSample(is_attention=True)
        self.ffn = FFN(drop_path_ratio=0.3, dropout=0.5, dim=out_chs[4])
        self.patch_embed = PatchEmded(in_chans=out_chs[3], embed_dim=out_chs[4])
        self.stage4_model = nn.Sequential()

        self.stage4_model.add_module('stage4_layer_patch_embed', self.patch_embed)
        self.stage4_model.add_module('stage4_layer_dowmsample', self.dowmsample)
        for i in range(repeat_num[4]):
            self.stage4_model.add_module('stage4_layer{}'.format(2 * i), self.relative)
            self.stage4_model.add_module('stage4_layer{}'.format(2 * i + 1), self.ffn)

    def forward(self, input):
        B, N, C = input.shape
        x = input.reshape(B, int(N**0.5), int(N**0.5), C).permute(0, 3, 1, 2)
        x = self.stage4_model(x)

        return x


class CoAtNet(nn.Module):
    def __init__(self, image_size, repeat_num=None, out_chs=None, class_num=256):
        super().__init__()
        self.s0 = StemStage(out_chs, repeat_num, img_size=image_size)
        self.s1 = StageOne(out_chs, repeat_num)
        self.s2 = StageTwo(out_chs, repeat_num)
        self.s3 = StageThree(out_chs, repeat_num)
        self.s4 = StageFour(out_chs, repeat_num)
        self.avg_pooling = nn.AdaptiveAvgPool1d(1)
        # self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))  ###
        self.drop_out = nn.Dropout(0.5)
        self.fc_output = nn.Linear(128, class_num)
        self.projection = nn.Linear(49, 128)
        self.activate = nn.GELU()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):

        # stage0
        y = self.s0(x)
        # stage1
        y = self.s1(y)
        # stage2
        y = self.s2(y)
        # # stage3
        y = self.s3(y)
        # # stage4
        y = self.s4(y)
        # output_head
        y = self.avg_pooling(y)
        y = torch.reshape(y, (y.shape[0], -1))
        y = self.projection(y)
        y = self.activate(y)
        y = self.fc_output(y)

        return y


if __name__ == '__main__':
    out_ch = [64, 96, 192, 384, 768]
    repeat_num = [2, 2, 2, 2, 2]
    x = torch.randn(1, 3, 112, 112).cuda()
    coatnet = CoAtNet(112, repeat_num, out_ch, class_num=256).cuda()
    y = coatnet(x)
    print(y.shape)
