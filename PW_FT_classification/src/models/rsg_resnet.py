import os
import copy
import warnings
from collections import OrderedDict
import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1
from .losses import LDAMLoss, FocalLoss
from .RSG import *

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_state_dict_from_url

# Exportable class names for external use
__all__ = [
    'RSGResNet'
]

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
}


class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = (F.normalize(x, dim=1)).mm(F.normalize(self.weight, dim=0))
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, head_lists=[], zero_init_residual=False,
                 groups=1, width_per_group=64, phase_train=True, epoch_thresh=0, 
                 replace_stride_with_dilation=None, norm_layer=None, n_center=15, 
                 transfer_strength=1.0):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.phase_train = phase_train
        
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if self.phase_train:
           self.head_lists = head_lists
           self.RSG = RSG(n_center = n_center, 
                          feature_maps_shape = [256*block.expansion, 14, 14], 
                          num_classes = num_classes, 
                          contrastive_module_dim = 256, 
                          head_class_lists = self.head_lists, 
                          transfer_strength = transfer_strength, 
                          epoch_thresh = epoch_thresh)

        # self.fc_ = NormedLinear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    
    def _forward_impl(self, x, epoch=0, batch_target=None, phase_train=False):
        """
        Forward pass implementation for the ResNet backbone.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the ResNet backbone.
        """
        # Applying the ResNet layers and operations
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if phase_train:
            x, cesc_total, loss_mv_total, combine_target = self.RSG.forward(
                x, self.head_lists, batch_target, epoch
            )

        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if phase_train:
            return x, cesc_total, loss_mv_total, combine_target
        else:
            return x
        
    def forward(self, x, epoch=0, batch_target=None, phase_train=True):
        return self._forward_impl(x, epoch, batch_target, phase_train)


class ResNetBackbone(ResNet):
    """
    Custom ResNet backbone class for feature extraction.

    Inherits from the torchvision ResNet class and allows customization of the architecture.
    """

    def __init__(
        self,
        block,
        layers,
        num_classes,
        head_lists=[],
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        phase_train=False,
        epoch_thresh=0,
        replace_stride_with_dilation=None,
        norm_layer=None,
        n_center=15,
        transfer_strength=0.1
    ):
        """
        Initialize the ResNet backbone.

        Args:
            block (nn.Module): Type of block to use (BasicBlock or Bottleneck).
            layers (list of int): Number of layers in each block.
            zero_init_residual (bool): Zero-initialize the last BN in each residual branch.
            groups (int): Number of groups for group normalization.
            width_per_group (int): Width per group.
            replace_stride_with_dilation (list of bool or None): Use dilation instead of stride.
            norm_layer (callable or None): Norm layer to use.
        """
        super(ResNetBackbone, self).__init__(
            block=block,
            layers=layers,
            num_classes=num_classes,
            head_lists=head_lists,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            phase_train=phase_train,
            epoch_thresh=epoch_thresh,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer,
            n_center=n_center,
            transfer_strength=transfer_strength
        )


class RSGResNet(nn.Module):
    """
    Custom ResNet classifier class.

    Extends nn.Module and provides a complete ResNet-based classifier, including feature extraction and classification layers.
    """

    name = 'RSGResNet'

    def __init__(self, train_class_counts, num_cls=10, num_layers=18, loss_type="CE", 
                 head_lists=[], phase_train=False, epoch_thresh=0, n_center=15, 
                 transfer_strength=0.1):
        """
        Initialize the RSGResNet.

        Args:
            num_cls (int): Number of classes for the classifier.
            num_layers (int): Number of layers in the ResNet model (e.g., 18, 50).
        """
        super(RSGResNet, self).__init__()
        self.num_cls = num_cls
        self.num_layers = num_layers
        self.loss_type = loss_type
        self.train_class_counts = train_class_counts
        self.feature = None
        self.classifier = None
        self.criterion_cls = None
        self.head_lists = head_lists
        self.phase_train = phase_train
        self.epoch_thresh = epoch_thresh
        self.n_center = n_center
        self.transfer_strength = transfer_strength

        # Initialize the network with the specified settings
        self.setup_net()

    def setup_net(self):
        """
        Set up the ResNet network and initialize its weights.
        """
        kwargs = {}

        # Selecting the appropriate ResNet architecture and pre-trained weights
        if self.num_layers == 18:
            block = BasicBlock
            layers = [2, 2, 2, 2]
            #self.pretrained_weights = ResNet18_Weights.IMAGENET1K_V1
            self.pretrained_weights = state_dict = load_state_dict_from_url(model_urls['resnet18'],
                                              progress=True)
        elif self.num_layers == 50:
            block = Bottleneck
            layers = [3, 4, 6, 3]
            #self.pretrained_weights = ResNet50_Weights.IMAGENET1K_V1
            self.pretrained_weights = state_dict = load_state_dict_from_url(model_urls['resnet50'],
                                              progress=True)
        else:
            raise Exception('ResNet Type not supported.')

        # Constructing the feature extractor and classifier
        self.feature = ResNetBackbone(block, layers, num_classes=self.num_cls, 
                                      head_lists=self.head_lists, phase_train=self.phase_train, 
                                      epoch_thresh=self.epoch_thresh, n_center=self.n_center, 
                                      transfer_strength=self.transfer_strength, **kwargs)
        self.classifier = NormedLinear(512 * block.expansion, self.num_cls)

    def setup_criteria(self):
        """
        Set up the criterion for the classifier.
        """
        if self.loss_type == "CE":
            self.criterion_cls = nn.CrossEntropyLoss()
        elif self.loss_type == "LDAM":
            self.criterion_cls = LDAMLoss(cls_num_list=self.train_class_counts, max_m=0.5, s=30)
        elif self.loss_type == 'Focal':
            self.criterion_cls = FocalLoss(gamma=1)
        else:
            warnings.warn('Loss type is not listed')
            return

    def feat_init(self):
        """
        Initialize the feature extractor with pre-trained weights.
        """
        # Load pre-trained weights and adjust for the current model
        #init_weights = self.pretrained_weights.get_state_dict(progress=True)
        init_weights = self.pretrained_weights
        init_weights = OrderedDict({k.replace('module.', '').replace('feature.', ''): init_weights[k]
                                    for k in init_weights})

        # Load the weights into the feature extractor
        self.feature.load_state_dict(init_weights, strict=False)

        # Identify missing and unused keys in the loaded weights
        load_keys = set(init_weights.keys())
        self_keys = set(self.feature.state_dict().keys())

        missing_keys = self_keys - load_keys
        unused_keys = load_keys - self_keys
        print('missing keys: {}'.format(sorted(list(missing_keys))))
        print('unused_keys: {}'.format(sorted(list(unused_keys))))
