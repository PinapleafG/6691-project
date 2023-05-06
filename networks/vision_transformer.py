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
                clip_dim=768,
                depths_enoder=[2,2,2,2], 
                depths_decoder=[1,2,2,2],
                num_heads = [3,6,12,24],
                num_classes=20,
                dropout=0.1):
        super(SwinUnet, self).__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depths_enoder = depths_enoder
        self.depths_decoder = depths_decoder
        self.num_heads = num_heads
        self.dropout = dropout

        self.swin_unet = SwinTransformerSys(img_size=self.img_size,
                                            num_classes=self.num_classes,
                                            embed_dim=self.embed_dim,
                                            depths=self.depths_enoder,
                                            depths_decoder=self.depths_decoder,
                                            num_heads=self.num_heads,
                                            drop_rate=self.dropout)
        
        # self.desc_encoder = nn.Linear(clip_dim, clip_dim, bias=False).cuda()
                                
    def forward(self, x, desc_feat):
        # desc_feat = self.desc_encoder(desc_feat)
        logits, img_text_logits = self.swin_unet(x, desc_feat)
        return logits, img_text_logits
 