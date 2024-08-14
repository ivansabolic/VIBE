# Copyright 2023 solo-learn development team.
from typing import Type, List, Optional, Callable, Any

from torch import Tensor
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from torchvision.models import resnet18, ResNet, ResNet18_Weights
from torchvision.models import resnet50

__all__ = ["resnet18", "resnet18_wo_relu", "resnet50"]

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models._utils import _ovewrite_named_param

from models.resnet import Bottleneck


def _resnet_wo_relu(
    block,
    layers,
    weights,
    progress,
    **kwargs,
) -> ResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet18_wo_relu(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


def resnet18_wo_relu(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.

    Args:
        weights (:class:`~torchvision.models.ResNet18_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet18_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet18_Weights
        :members:
    """
    weights = ResNet18_Weights.verify(weights)

    return _resnet_wo_relu(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlock_wo_relu(BasicBlock):

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        # out = self.relu(out)

        return out


class ResNet18_wo_relu(ResNet):
    def __init__(self,
                 block: Type[BasicBlock],
                 layers: List[int],
                 num_classes: int = 1000,
                 zero_init_residual: bool = False,
                 groups: int = 1,
                 width_per_group: int = 64,
                 replace_stride_with_dilation: Optional[List[bool]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 ):

        # super(ResNet18_wo_relu, self).__init__(block, layers, num_classes, zero_init_residual, groups, width_per_group,
        #                                        replace_stride_with_dilation, norm_layer)
        nn.Module.__init__(self)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(BasicBlock_wo_relu, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]


    # def _make_layer(self, block, planes, num_blocks, stride):
    #     strides = [stride] + [1] * (num_blocks - 1)
    #     layers = []
    #     for stride in strides:
    #         layers.append(block(self.in_planes, planes, stride))
    #         self.in_planes = planes * block.expansion
    #     return nn.Sequential(*layers)
    #
    # def forward(self, x, return_feature=False, return_feature_list=False):
    #     feature1 = F.relu(self.bn1(self.conv1(x)))
    #     feature2, _ = self.layer1(feature1)
    #     feature3, _ = self.layer2(feature2)
    #     feature4, _ = self.layer3(feature3)
    #     feature5, feature5_ = self.layer4(feature4)
    #     feature5 = self.avgpool(feature5)
    #     feature5_ = self.avgpool(feature5_)
    #     feature = feature5.view(feature5.size(0), -1)
    #     feature_ = feature5_.view(feature5_.size(0), -1)
    #     logits_cls = self.fc(feature)
    #     feature_list = [feature1, feature2, feature3, feature4, feature5]
    #     if return_feature:
    #         return logits_cls, feature_
    #     elif return_feature_list:
    #         return logits_cls, feature_list
    #     else:
    #         return logits_cls
    #
    # def forward_threshold(self, x, threshold):
    #     feature1 = F.relu(self.bn1(self.conv1(x)))
    #     feature2 = self.layer1(feature1)
    #     feature3 = self.layer2(feature2)
    #     feature4 = self.layer3(feature3)
    #     feature5 = self.layer4(feature4)
    #     feature5 = self.avgpool(feature5)
    #     feature = feature5.clip(max=threshold)
    #     feature = feature.view(feature.size(0), -1)
    #     logits_cls = self.fc(feature)
    #
    #     return logits_cls
    #
    # def forward___(self, x, return_feature=False, return_feature_list=False):
    #     feature1 = F.relu(self.bn1(self.conv1(x)))
    #     feature2 = self.layer1(feature1)
    #     feature3 = self.layer2(feature2)
    #     feature4 = self.layer3(feature3)
    #     feature5_ = self.layer4(feature4)
    #     feature5 = F.relu(feature5_)
    #     feature5 = self.avgpool(feature5)
    #     feature5_ = self.avgpool(feature5_)
    #     feature = feature5.view(feature5.size(0), -1)
    #     feature_ = feature5_.view(feature5_.size(0), -1)
    #     logits_cls = self.fc(feature)
    #     feature_list = [feature1, feature2, feature3, feature4, feature5]
    #     if return_feature:
    #         return logits_cls, feature_
    #     elif return_feature_list:
    #         return logits_cls, feature_list
    #     else:
    #         return logits_cls
    #
    # def intermediate_forward(self, x, layer_index):
    #     out = F.relu(self.bn1(self.conv1(x)))
    #
    #     out = self.layer1(out)
    #     if layer_index == 1:
    #         return out
    #
    #     out = self.layer2(out)
    #     if layer_index == 2:
    #         return out
    #
    #     out = self.layer3(out)
    #     if layer_index == 3:
    #         return out
    #
    #     out = self.layer4(out)
    #     if layer_index == 4:
    #         return out
    #
    #     raise ValueError
    #
    # def get_fc(self):
    #     fc = self.fc
    #     return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()
    #
    # def get_fc_layer(self):
    #     return self.fc


