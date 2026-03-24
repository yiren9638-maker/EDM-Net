# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
_base_ = [
    '../_base_/models/upernet_r50.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
pretrained = r'E:\renyi\InternImage\segmentation\work_dirs\mapilary\best_mIoU_iter_64000.pth'
model = dict(
    backbone=dict(
        _delete_=True,
        type='InternImage',
        core_op='DCNv3',
        channels=64,
        depths=[4, 4, 18, 4],
        groups=[4, 8, 16, 32],
        mlp_ratio=4.,
        drop_path_rate=0.2,
        norm_layer='LN',
        layer_scale=1.0,
        offset_scale=1.0,
        post_norm=False,
        with_cp=False,
        out_indices=(0, 1, 2, 3),
        danet_channels=None,  # 可选：指定输出通道数
        use_danet=False,  # 新增参数，启用 DANet
        use_edge_refine=True,  # 启用边缘细化
        edge_refine_channels=64,
        frozen_stages=3,  # 冻结前3个stage
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    decode_head=dict(num_classes=19, in_channels=[64, 128, 256, 512]),
    auxiliary_head=dict(num_classes=19, in_channels=256),
    test_cfg=dict(mode='whole')
)
optimizer = dict(
    _delete_=True, type='AdamW', lr=0.00002, betas=(0.9, 0.999), weight_decay=0.05,
    constructor='CustomLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=30, layer_decay_rate=1.0,
                       depths=[4, 4, 18, 4],
                       custom_keys={
                           'backbone': dict(lr_mult=0.5),  # 骨干网络学习率减半
                           'norm': dict(decay_mult=0.),  # 忽略norm层权重衰减
                            'head': dict(lr_mult=1.5),  # 提高解码头学习率
                       }
                       ))
lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(
    samples_per_gpu=2,
    train=dict(
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion',
                 brightness_delta=20,
                 contrast_range=(0.8, 1.2)),
            dict(type='RandomCrop',
                 crop_size=(768, 768),  # 更小的裁剪增强细节
                 cat_max_ratio=0.75),   # 控制类别比例
            dict(type='Normalize',
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375]),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]
    ))
runner = dict(type='IterBasedRunner',max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=3)
evaluation = dict(interval=4000, metric='mIoU', save_best='mIoU')
# fp16 = dict(loss_scale=dict(init_scale=512))
