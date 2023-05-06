# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join as pjoin

import torch.nn as nn
import numpy as np
from scipy import ndimage
from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys


class SwinUnet(nn.Module):
    def __init__(self, 
                img_size=224, 
                embed_dim=96, 
                depths_enoder=[2,2,2,2], 
                depths_decoder=[1,2,2,2],
                num_heads = [3,6,12,24],
                num_classes=20):
        super(SwinUnet, self).__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depths_enoder = depths_enoder
        self.depths_decoder = depths_decoder
        self.num_heads = num_heads

        self.swin_unet = SwinTransformerSys(img_size=self.img_size,
                                            num_classes=self.num_classes,
                                            embed_dim=self.embed_dim,
                                            depths=self.depths_enoder,
                                            depths_decoder=self.depths_decoder,
                                            num_heads=self.num_heads)
                                
    def forward(self, x, desc_feat):
        logits = self.swin_unet(x, desc_feat)
        return logits
 