# dataset settings
dataset_type = 'PotsdamDataset'
data_root = 'data/potsdam'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)
crop_size = (512, 512)

# 训练流水线 (旧版格式)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='Resize',
        img_scale=(512, 512),  # 直接使用img_scale参数
        ratio_range=(0.5, 2.0),
        keep_ratio=True
    ),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),  # 显式添加归一化
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),  # 添加填充操作
    dict(type='DefaultFormatBundle'),  # 旧版数据打包
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])  # 旧版数据收集
]

# 测试流水线 (旧版格式)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),  # 单尺度测试
        flip=False,  # 禁用翻转增强
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),  # 虽然flip=False，但保留结构
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),  # 只收集图像
        ])
]

# 旧版数据配置
data = dict(
    samples_per_gpu=4,  # 对应batch_size
    workers_per_gpu=4,  # 对应num_workers
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/train',
        ann_dir='ann_dir/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        pipeline=test_pipeline),  # 验证集使用测试流水线
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/val',  # 测试集路径（根据实际情况修改）
        ann_dir='ann_dir/val',
        pipeline=test_pipeline))