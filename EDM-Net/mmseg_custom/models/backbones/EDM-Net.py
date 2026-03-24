# --------------------------------------------------------
# InternImage + Vision Mamba 双分支骨干网络
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from mmcv.cnn import constant_init, trunc_normal_init
from mmcv.runner import _load_checkpoint
from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger
from ops_dcnv3 import modules as dcnv3
from timm.models.layers import DropPath, trunc_normal_
# 导入 Vision Mamba 模块（需提前安装：pip install vision-mamba）
from vision_mamba import ViMBlock  # Vision Mamba 核心模块


class to_channels_first(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class to_channels_last(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)


def build_norm_layer(dim,
                     norm_layer,
                     in_format='channels_last',
                     out_format='channels_last',
                     eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)


def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()
    raise NotImplementedError(f'build_act_layer does not support {act_layer}')


class CrossAttention(nn.Module):
    r""" Cross Attention Module
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 8
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: False.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        attn_head_dim (int, optional): Dimension of attention head.
        out_dim (int, optional): Dimension of output.
    """

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 attn_head_dim=None,
                 out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        assert all_head_dim == dim

        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.k = nn.Linear(dim, all_head_dim, bias=False)
        self.v = nn.Linear(dim, all_head_dim, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.k_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, k=None, v=None):
        B, N, C = x.shape
        N_k = k.shape[1]
        N_v = v.shape[1]

        q_bias, k_bias, v_bias = None, None, None
        if self.q_bias is not None:
            q_bias = self.q_bias
            k_bias = self.k_bias
            v_bias = self.v_bias

        q = F.linear(input=x, weight=self.q.weight, bias=q_bias)
        q = q.reshape(B, N, 1, self.num_heads,
                      -1).permute(2, 0, 3, 1,
                                  4).squeeze(0)  # (B, N_head, N_q, dim)

        k = F.linear(input=k, weight=self.k.weight, bias=k_bias)
        k = k.reshape(B, N_k, 1, self.num_heads, -1).permute(2, 0, 3, 1,
                                                             4).squeeze(0)

        v = F.linear(input=v, weight=self.v.weight, bias=v_bias)
        v = v.reshape(B, N_v, 1, self.num_heads, -1).permute(2, 0, 3, 1,
                                                             4).squeeze(0)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B, N_head, N_q, N_k)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class AttentiveBlock(nn.Module):
    r"""Attentive Block
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 8
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: False.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop (float, optional): Dropout rate. Default: 0.0.
        attn_drop (float, optional): Attention dropout rate. Default: 0.0.
        drop_path (float | tuple[float], optional): Stochastic depth rate.
            Default: 0.0.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm.
        attn_head_dim (int, optional): Dimension of attention head. Default: None.
        out_dim (int, optional): Dimension of output. Default: None.
    """

    def __init__(self,
                 dim,
                 num_heads,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer='LN',
                 attn_head_dim=None,
                 out_dim=None):
        super().__init__()

        self.norm1_q = build_norm_layer(dim, norm_layer, eps=1e-6)
        self.norm1_k = build_norm_layer(dim, norm_layer, eps=1e-6)
        self.norm1_v = build_norm_layer(dim, norm_layer, eps=1e-6)
        self.cross_dcn = CrossAttention(dim,
                                        num_heads=num_heads,
                                        qkv_bias=qkv_bias,
                                        qk_scale=qk_scale,
                                        attn_drop=attn_drop,
                                        proj_drop=drop,
                                        attn_head_dim=attn_head_dim,
                                        out_dim=out_dim)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self,
                x_q,
                x_kv,
                pos_q,
                pos_k,
                bool_masked_pos,
                rel_pos_bias=None):
        x_q = self.norm1_q(x_q + pos_q)
        x_k = self.norm1_k(x_kv + pos_k)
        x_v = self.norm1_v(x_kv)

        x = self.cross_dcn(x_q, k=x_k, v=x_v)

        return x


class AttentionPoolingBlock(AttentiveBlock):
    def forward(self, x):
        x_q = x.mean(1, keepdim=True)
        x_kv = x
        pos_q, pos_k = 0, 0
        x = super().forward(x_q, x_kv, pos_q, pos_k,
                            bool_masked_pos=None,
                            rel_pos_bias=None)
        x = x.squeeze(1)
        return x


class StemLayer(nn.Module):
    r""" Stem layer of InternImage
    Args:
        in_chans (int): number of input channels
        out_chans (int): number of output channels
        act_layer (str): activation layer
        norm_layer (str): normalization layer
    """

    def __init__(self,
                 in_chans=3,
                 out_chans=96,
                 act_layer='GELU',
                 norm_layer='BN'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans,
                               out_chans // 2,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.norm1 = build_norm_layer(out_chans // 2, norm_layer,
                                      'channels_first', 'channels_first')
        self.act = build_act_layer(act_layer)
        self.conv2 = nn.Conv2d(out_chans // 2,
                               out_chans,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.norm2 = build_norm_layer(out_chans, norm_layer, 'channels_first',
                                      'channels_last')

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x


class DownsampleLayer(nn.Module):
    r""" Downsample layer of InternImage
    Args:
        channels (int): number of input channels
        norm_layer (str): normalization layer
    """

    def __init__(self, channels, norm_layer='LN'):
        super().__init__()
        self.conv = nn.Conv2d(channels,
                              2 * channels,
                              kernel_size=3,
                              stride=2,
                              padding=1,
                              bias=False)
        self.norm = build_norm_layer(2 * channels, norm_layer,
                                     'channels_first', 'channels_last')

    def forward(self, x):
        x = self.conv(x.permute(0, 3, 1, 2))
        x = self.norm(x)
        return x


class MLPLayer(nn.Module):
    r""" MLP layer of InternImage
    Args:
        in_features (int): number of input features
        hidden_features (int): number of hidden features
        out_features (int): number of output features
        act_layer (str): activation layer
        drop (float): dropout rate
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer='GELU',
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = build_act_layer(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class InternImageLayer(nn.Module):
    r""" Basic layer of InternImage
    Args:
        core_op (nn.Module): core operation of InternImage
        channels (int): number of input channels
        groups (list): Groups of each block.
        mlp_ratio (float): ratio of mlp hidden features to input channels
        drop (float): dropout rate
        drop_path (float): drop path rate
        act_layer (str): activation layer
        norm_layer (str): normalization layer
        post_norm (bool): whether to use post normalization
        layer_scale (float): layer scale
        offset_scale (float): offset scale
        with_cp (bool): whether to use checkpoint
    """

    def __init__(self,
                 core_op,
                 channels,
                 groups,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer='GELU',
                 norm_layer='LN',
                 post_norm=False,
                 layer_scale=None,
                 offset_scale=1.0,
                 with_cp=False,
                 dw_kernel_size=None,  # for InternImage-H/G
                 res_post_norm=False,  # for InternImage-H/G
                 center_feature_scale=False,
                 use_dcn_v4_op=False):  # for InternImage-H/G
        super().__init__()
        self.channels = channels
        self.groups = groups
        self.mlp_ratio = mlp_ratio
        self.with_cp = with_cp

        self.norm1 = build_norm_layer(channels, 'LN')
        self.post_norm = post_norm
        self.dcn = core_op(
            channels=channels,
            kernel_size=3,
            stride=1,
            pad=1,
            dilation=1,
            group=groups,
            offset_scale=offset_scale,
            act_layer=act_layer,
            norm_layer=norm_layer,
            dw_kernel_size=dw_kernel_size,  # for InternImage-H/G
            center_feature_scale=center_feature_scale,
            use_dcn_v4_op=use_dcn_v4_op)  # for InternImage-H/G
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.norm2 = build_norm_layer(channels, 'LN')
        self.mlp = MLPLayer(in_features=channels,
                            hidden_features=int(channels * mlp_ratio),
                            act_layer=act_layer,
                            drop=drop)
        self.layer_scale = layer_scale is not None
        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(channels),
                                       requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(channels),
                                       requires_grad=True)
        self.res_post_norm = res_post_norm
        if res_post_norm:
            self.res_post_norm1 = build_norm_layer(channels, 'LN')
            self.res_post_norm2 = build_norm_layer(channels, 'LN')

    def forward(self, x):

        def _inner_forward(x):
            if not self.layer_scale:
                if self.post_norm:
                    x = x + self.drop_path(self.norm1(self.dcn(x)))
                    x = x + self.drop_path(self.norm2(self.mlp(x)))
                elif self.res_post_norm:  # for InternImage-H/G
                    x = x + self.drop_path(self.res_post_norm1(self.dcn(self.norm1(x))))
                    x = x + self.drop_path(self.res_post_norm2(self.mlp(self.norm2(x))))
                else:
                    x = x + self.drop_path(self.dcn(self.norm1(x)))
                    x = x + self.drop_path(self.mlp(self.norm2(x)))
                return x
            if self.post_norm:
                x = x + self.drop_path(self.gamma1 * self.norm1(self.dcn(x)))
                x = x + self.drop_path(self.gamma2 * self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.gamma1 * self.dcn(self.norm1(x)))
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
            return x

        if self.with_cp and x.requires_grad:
            x = checkpoint.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class InternImageBlock(nn.Module):
    r""" Block of InternImage
    Args:
        core_op (nn.Module): core operation of InternImage
        channels (int): number of input channels
        depths (list): Depth of each block.
        groups (list): Groups of each block.
        mlp_ratio (float): ratio of mlp hidden features to input channels
        drop (float): dropout rate
        drop_path (float): drop path rate
        act_layer (str): activation layer
        norm_layer (str): normalization layer
        post_norm (bool): whether to use post normalization
        layer_scale (float): layer scale
        offset_scale (float): offset scale
        with_cp (bool): whether to use checkpoint
    """

    def __init__(self,
                 core_op,
                 channels,
                 depth,
                 groups,
                 downsample=True,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer='GELU',
                 norm_layer='LN',
                 post_norm=False,
                 offset_scale=1.0,
                 layer_scale=None,
                 with_cp=False,
                 dw_kernel_size=None,  # for InternImage-H/G
                 post_norm_block_ids=None,  # for InternImage-H/G
                 res_post_norm=False,  # for InternImage-H/G
                 center_feature_scale=False,  # for InternImage-H/G
                 use_dcn_v4_op=False):
        super().__init__()
        self.channels = channels
        self.depth = depth
        self.post_norm = post_norm
        self.center_feature_scale = center_feature_scale

        self.blocks = nn.ModuleList([
            InternImageLayer(
                core_op=core_op,
                channels=channels,
                groups=groups,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(
                    drop_path, list) else drop_path,
                act_layer=act_layer,
                norm_layer=norm_layer,
                post_norm=post_norm,
                layer_scale=layer_scale,
                offset_scale=offset_scale,
                with_cp=with_cp,
                dw_kernel_size=dw_kernel_size,  # for InternImage-H/G
                res_post_norm=res_post_norm,  # for InternImage-H/G
                center_feature_scale=center_feature_scale,  # for InternImage-H/G
                use_dcn_v4_op=use_dcn_v4_op
            ) for i in range(depth)
        ])
        if not self.post_norm or center_feature_scale:
            self.norm = build_norm_layer(channels, 'LN')
        self.post_norm_block_ids = post_norm_block_ids
        if post_norm_block_ids is not None:  # for InternImage-H/G
            self.post_norms = nn.ModuleList(
                [build_norm_layer(channels, 'LN', eps=1e-6) for _ in post_norm_block_ids]
            )
        self.downsample = DownsampleLayer(
            channels=channels, norm_layer=norm_layer) if downsample else None

    def forward(self, x, return_wo_downsample=False):
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (self.post_norm_block_ids is not None) and (i in self.post_norm_block_ids):
                index = self.post_norm_block_ids.index(i)
                x = self.post_norms[index](x)  # for InternImage-H/G
        if not self.post_norm or self.center_feature_scale:
            x = self.norm(x)
        if return_wo_downsample:
            x_ = x
        if self.downsample is not None:
            x = self.downsample(x)

        if return_wo_downsample:
            return x, x_
        return x


# ------------------------------ Vision Mamba分支相关模块 ------------------------------
class VisionMambaBlock(nn.Module):
    r""" Vision Mamba Block（专为视觉任务优化）
    Args:
        dim (int): Number of input channels.
        drop (float): Dropout rate. Default: 0.0.
        n_div (int): Number of divisions for group convolution. Default: 8.
        mlp_ratio (float): Ratio of MLP hidden dim to input dim. Default: 4.0.
        drop_path (float): Stochastic depth rate. Default: 0.0.
    """

    def __init__(self, dim, drop=0., n_div=8, mlp_ratio=4.0, drop_path=0.):
        super().__init__()
        # 格式转换：channels_last (B,H,W,C) → channels_first (B,C,H,W)
        self.to_cf = to_channels_first()
        # 格式转换：channels_first → channels_last
        self.to_cl = to_channels_last()

        # Vision Mamba 核心模块（原生支持 2D 特征输入）
        self.vim_block = ViMBlock(
            d_model=dim,  # 输入通道数
            n_div=n_div,  # 分组数（视觉任务常用8）
            mlp_ratio=mlp_ratio,  # MLP扩张比
            drop=drop,  # dropout率
            drop_path=drop_path,  # 随机深度率
            norm_layer='LN',  # 归一化层
            act_layer='GELU'  # 激活函数
        )

    def forward(self, x):
        # x: (B, H, W, C) → 转为 Vision Mamba 输入格式 (B, C, H, W)
        x_cf = self.to_cf(x)
        # Vision Mamba 前向传播
        x_vim = self.vim_block(x_cf)
        # 转回原网络格式 (B, H, W, C)
        x_out = self.to_cl(x_vim)
        return x_out


class CrossAttentionFusion(nn.Module):
    r""" Cross Attention Fusion for dual-branch feature complement
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 8.
    """

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.norm = build_norm_layer(dim, 'LN')
        self.attn = CrossAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=0.1
        )
        self.proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(0.1) if 0.1 > 0. else nn.Identity()

    def forward(self, feat_intern, feat_mamba):
        # 特征融合前归一化
        feat_cat = self.norm(feat_intern + feat_mamba)
        # 交叉注意力融合
        feat_fused = self.attn(feat_cat, k=feat_intern, v=feat_mamba)
        # 残差连接与特征校准
        feat_fused = self.drop_path(self.proj(feat_fused)) + feat_cat
        return feat_fused


class MultiScaleFusion(nn.Module):
    r""" Multi-scale Feature Fusion to single output
    Args:
        in_dims (list): List of input channel dimensions for each scale.
        out_dim (int): Output channel dimension. Default: 512.
    """

    def __init__(self, in_dims=[64, 128, 256, 512], out_dim=512):
        super().__init__()
        self.in_dims = in_dims
        self.out_dim = out_dim
        # 通道统一卷积（将不同尺度通道转为out_dim）
        self.channel_proj = nn.ModuleList([
            nn.Conv2d(dim, out_dim, kernel_size=1, bias=False)
            for dim in in_dims
        ])
        # 自适应注意力权重
        self.attn_weight = nn.Parameter(torch.ones(len(in_dims)))
        # 输出校准层
        self.norm = nn.BatchNorm2d(out_dim)
        self.act = build_act_layer('GELU')
        self.final_proj = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False)

    def forward(self, feats):
        # feats: 列表，包含4个尺度的特征图 (B, C, H, W)
        B, _, H0, W0 = feats[0].shape  # 最精细尺度 (H/4, W/4)
        proj_feats = []

        for i, (feat, proj_conv) in enumerate(zip(feats, self.channel_proj)):
            # 上采样到最精细尺度
            feat_up = F.interpolate(
                feat, size=(H0, W0), mode='bilinear', align_corners=False
            )
            # 通道统一
            feat_proj = proj_conv(feat_up)
            proj_feats.append(feat_proj)

        # 注意力加权融合
        weight = F.softmax(self.attn_weight, dim=0)
        fused_feat = sum(w * feat for w, feat in zip(weight, proj_feats))

        # 输出校准
        fused_feat = self.act(self.norm(fused_feat))
        fused_feat = self.final_proj(fused_feat)

        return fused_feat


# ------------------------------ 双分支骨干网络 ------------------------------
@BACKBONES.register_module()
class InternImageVisionMamba(nn.Module):
    r""" InternImage + Vision Mamba Dual-branch Backbone
    Args:
        core_op (str): Core operator of InternImage. Default: 'DCNv3'.
        channels (int): Number of the first stage channels. Default: 64.
        depths (list): Depth of each block. Default: [3, 4, 18, 5].
        groups (list): Groups of each block. Default: [3, 6, 12, 24].
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        drop_rate (float): Probability of an element to be zeroed. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        act_layer (str): Activation layer. Default: 'GELU'.
        norm_layer (str): Normalization layer. Default: 'LN'.
        layer_scale (bool): Whether to use layer scale. Default: None.
        offset_scale (float): Offset scale of DCNv3. Default: 1.0.
        post_norm (bool): Whether to use post normalization. Default: False.
        with_cp (bool): Use checkpoint or not. Default: False.
        out_dim (int): Output channel dimension. Default: 512.
        vim_n_div (int): Number of divisions for Vision Mamba group conv. Default: 8.
        vim_mlp_ratio (float): MLP ratio for Vision Mamba. Default: 4.0.
    """

    def __init__(self,
                 core_op='DCNv3',
                 channels=64,
                 depths=[3, 4, 18, 5],
                 groups=[3, 6, 12, 24],
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 drop_path_type='linear',
                 act_layer='GELU',
                 norm_layer='LN',
                 layer_scale=None,
                 offset_scale=1.0,
                 post_norm=False,
                 with_cp=False,
                 dw_kernel_size=None,
                 level2_post_norm=False,
                 level2_post_norm_block_ids=None,
                 res_post_norm=False,
                 center_feature_scale=False,
                 use_dcn_v4_op=False,
                 out_dim=512,
                 vim_n_div=8,  # Vision Mamba 分组数
                 vim_mlp_ratio=4.0,  # Vision Mamba MLP扩张比
                 frozen_stages=-1,
                 init_cfg=None,
                 **kwargs):
        super().__init__()
        self.core_op = core_op
        self.num_levels = len(depths)
        self.depths = depths
        self.channels = channels
        self.out_dim = out_dim
        self.vim_n_div = vim_n_div
        self.vim_mlp_ratio = vim_mlp_ratio
        self.init_cfg = init_cfg
        self.frozen_stages = frozen_stages

        # 日志输出
        logger = get_root_logger()
        logger.info(f'Using dual-branch backbone: InternImage + Vision Mamba')
        logger.info(f'Using core type: {core_op}')
        logger.info(f'Vision Mamba n_div: {vim_n_div}, mlp_ratio: {vim_mlp_ratio}')
        logger.info(f'Output channel dimension: {out_dim}')

        # ------------------------------ 公共输入层 ------------------------------
        in_chans = 3
        self.patch_embed = StemLayer(
            in_chans=in_chans,
            out_chans=channels,
            act_layer=act_layer,
            norm_layer=norm_layer
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 计算drop path rate
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        if drop_path_type == 'uniform':
            for i in range(len(dpr)):
                dpr[i] = drop_path_rate

        # ------------------------------ 分支1：InternImage分支 ------------------------------
        self.intern_levels = nn.ModuleList()
        for i in range(self.num_levels):
            post_norm_block_ids = level2_post_norm_block_ids if level2_post_norm and (
                    i == 2) else None
            intern_level = InternImageBlock(
                core_op=getattr(dcnv3, core_op),
                channels=int(channels * 2 ** i),
                depth=depths[i],
                groups=groups[i],
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                act_layer=act_layer,
                norm_layer=norm_layer,
                post_norm=post_norm,
                downsample=(i < self.num_levels - 1),
                layer_scale=layer_scale,
                offset_scale=offset_scale,
                with_cp=with_cp,
                dw_kernel_size=dw_kernel_size,
                post_norm_block_ids=post_norm_block_ids,
                res_post_norm=res_post_norm,
                center_feature_scale=center_feature_scale,
                use_dcn_v4_op=use_dcn_v4_op,
            )
            self.intern_levels.append(intern_level)

        # ------------------------------ 分支2：Vision Mamba分支 ------------------------------
        self.vim_levels = nn.ModuleList()
        for i in range(self.num_levels):
            stage_dim = int(channels * 2 ** i)
            # 当前阶段的drop path率
            stage_dpr = dpr[sum(depths[:i]):sum(depths[:i + 1])]
            # 堆叠Vision Mamba Block
            vim_block = nn.Sequential(*[
                VisionMambaBlock(
                    dim=stage_dim,
                    drop=drop_rate,
                    n_div=vim_n_div,
                    mlp_ratio=vim_mlp_ratio,
                    drop_path=stage_dpr[j]  # 每个Block的随机深度率
                ) for j in range(depths[i])
            ])
            self.vim_levels.append(vim_block)

        # ------------------------------ 特征融合模块 ------------------------------
        # 跨阶段融合模块（每个阶段一个）
        self.stage_fusions = nn.ModuleList([
            CrossAttentionFusion(dim=int(channels * 2 ** i))
            for i in range(self.num_levels)
        ])
        # 最终多尺度融合模块
        self.multi_scale_fusion = MultiScaleFusion(
            in_dims=[int(channels * 2 ** i) for i in range(self.num_levels)],
            out_dim=out_dim
        )

        # ------------------------------ 初始化 ------------------------------
        self.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        self._freeze_stages()

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer frozen."""
        super(InternImageVisionMamba, self).train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for level in self.intern_levels[:self.frozen_stages]:
                level.eval()
                for param in level.parameters():
                    param.requires_grad = False
            for level in self.vim_levels[:self.frozen_stages]:
                level.eval()
                for param in level.parameters():
                    param.requires_grad = False

    def init_weights(self):
        logger = get_root_logger()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1.0)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = _load_checkpoint(self.init_cfg.checkpoint,
                                    logger=logger,
                                    map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            # 加载InternImage分支权重
            intern_state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    intern_state_dict[k[9:]] = v
                elif not k.startswith('vim_') and not k.startswith('stage_fusions.') and not k.startswith(
                        'multi_scale_fusion.'):
                    intern_state_dict[k] = v

            # 加载权重
            meg = self.load_state_dict(intern_state_dict, strict=False)
            logger.info(f'Loaded InternImage weights, missing keys: {meg.missing_keys}')
            logger.info(f'Unexpected keys: {meg.unexpected_keys}')

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_deform_weights(self, m):
        if isinstance(m, getattr(dcnv3, self.core_op)):
            m._reset_parameters()

    def forward(self, x):
        # 公共输入处理
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        # 各阶段特征存储列表
        stage_feats = []

        # 双分支并行处理与跨阶段融合
        for i in range(self.num_levels):
            # InternImage分支前向（返回下采样后结果 + 未下采样特征）
            intern_level = self.intern_levels[i]
            x_intern, x_intern_wo_down = intern_level(x, return_wo_downsample=True)
            # Vision Mamba分支前向（直接处理2D特征，无需序列转换）
            vim_level = self.vim_levels[i]
            x_vim = vim_level(x)
            # 跨阶段交叉注意力融合
            fusion = self.stage_fusions[i]
            x_fused = fusion(x_intern_wo_down, x_vim)
            # 转为channels_first格式存入列表（适配多尺度融合）
            x_fused = x_fused.permute(0, 3, 1, 2).contiguous()
            stage_feats.append(x_fused)
            # 更新下阶段输入（复用InternImage分支的下采样结果）
            if i < self.num_levels - 1:
                x = x_intern

        # 最终多尺度融合为单个输出特征图
        final_feat = self.multi_scale_fusion(stage_feats)

        return final_feat