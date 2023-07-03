import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Short cut connection. If out_channels != in_channels, we need to reshape the input x
        # using 1x1 convolution to match the dimensions.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Add the short cut
        out = F.relu(out)
        return out


def create_residual_set(in_channels, out_channels, block_num, reduce_dim=False):
    """
    Helper function for creating a `set` of residual blocks in ResNet
    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param block_num: number of blocks in this `set`
    :param reduce_dim: whether the first block will reduce the dimension with stride=2
    :return:
    """
    assert block_num > 0 and in_channels > 0
    resnet_set = []
    init_stride = 2 if reduce_dim else 1
    resnet_set.append(ResidualBlock(in_channels, out_channels, stride=init_stride))
    for i in range(block_num - 1):
        resnet_set.append(ResidualBlock(out_channels, out_channels, stride=1))
    return nn.Sequential(*resnet_set)


class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes=10):
        super(ResNet, self).__init__()
        self.conv_init = nn.Conv2d(in_channels, 64, 7, stride=2)
        self.bn_init = nn.BatchNorm2d(64)
        self.mp_init = nn.MaxPool2d(3, stride=2)

        # setting up the residual sets
        res_set1 = create_residual_set(64, 64, 3, False)
        res_set2 = create_residual_set(64, 128, 4, True)
        res_set3 = create_residual_set(128, 256, 6, True)
        res_set4 = create_residual_set(256, 512, 3, True)
        self.res_sets = nn.Sequential(res_set1, res_set2, res_set3, res_set4)

        # global average pooling and FC layers
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor):
        # initial processing
        output = self.mp_init(F.relu(self.bn_init(self.conv_init(x))))
        # core resnet
        output = self.res_sets(output)
        # post processing
        output = self.gap(output).view(output.size(0), -1)
        output = self.fc(output)
        return output


if __name__ == '__main__':
    demo = torch.rand(10, 5, 5, 5)
    conv = nn.Conv2d(5, 10, 3, 1)
    demo2 = conv(demo)
    print(demo2.shape)